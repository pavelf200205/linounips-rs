//! Subgroup LayerNorm, FP32 register statistics and FP32 affine parameters.
use burn::tensor::Tensor as BurnTensor;
use burn_backend::{DType, ExecutionError};
use burn_cubecl::{
    CubeBackend,
    fusion::{CubeFusionHandle, FusionCubeRuntime},
    ops::numeric::empty_device_contiguous_dtype,
};
use burn_fusion::{
    Fusion,
    custom::{
        CustomOpIr, HandleContainer, Operation, OperationIr, OperationOutput, ScalarIr, StreamId,
        TensorIr,
    },
};
use cubecl::prelude::*;
type Gpu = Fusion<CubeBackend>;
#[cube(launch)]
fn normalize<F: Float>(
    x: &Tensor<F>,
    gamma: &Tensor<f32>,
    beta: &Tensor<f32>,
    out: &mut Tensor<F>,
    inv_c: f32,
    epsilon: f32,
    #[comptime] channels: usize,
    #[comptime] slots: usize,
    #[comptime] rows_per_group: usize,
) {
    let plane = usize::cast_from(PLANE_DIM);
    let lane = usize::cast_from(UNIT_POS_PLANE);
    let planes = usize::cast_from(CUBE_DIM) / plane;
    let group = usize::cast_from(CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X);
    for tile in 0..rows_per_group / planes {
        let row = group * rows_per_group + tile * planes + usize::cast_from(PLANE_POS);
        let valid = row < x.shape(0) * x.shape(1);
        let batch = row / x.shape(1);
        let seq = row % x.shape(1);
        let mut values = Array::<f32>::new(slots);
        let mut sum = f32::new(0.0_f32);
        #[unroll]
        for j in 0..slots {
            let channel = lane + j * plane;
            values[j] = 0.0_f32;
            if valid && channel < channels {
                let value = f32::cast_from(
                    x[batch * x.stride(0) + seq * x.stride(1) + channel * x.stride(2)],
                );
                values[j] = value;
                sum += value;
            }
        }
        let mean = plane_sum(sum) * inv_c;
        let mut variance = f32::new(0.0_f32);
        #[unroll]
        for j in 0..slots {
            let channel = lane + j * plane;
            if valid && channel < channels {
                let delta = values[j] - mean;
                variance += delta * delta;
            }
        }
        let inverse = 1.0_f32 / f32::sqrt(plane_sum(variance) * inv_c + epsilon);
        #[unroll]
        for j in 0..slots {
            let channel = lane + j * plane;
            if valid && channel < channels {
                out[row * channels + channel] =
                    F::cast_from((values[j] - mean) * inverse * gamma[channel] + beta[channel]);
            }
        }
    }
}
#[derive(Debug)]
struct NormOp {
    desc: CustomOpIr,
    epsilon: f32,
}
impl Operation<FusionCubeRuntime> for NormOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([x, gamma, beta], [out]) = self.desc.as_fixed();
        let [x, gamma, beta] =
            [x, gamma, beta].map(|ir| handles.get_float_tensor::<CubeBackend>(ir));
        let [b, n, c] = x.meta.shape().dims::<3>();
        let client = x.client.clone();
        let hw = &client.properties().hardware;
        let rows_per_group = 128 / hw.plane_size_min as usize;
        let slots = c.div_ceil(hw.plane_size_min as usize);
        let groups = (b * n).div_ceil(rows_per_group);
        let gx = groups.min(hw.max_cube_count.0 as usize).max(1);
        let gy = groups.div_ceil(gx);
        if gy > hw.max_cube_count.1 as usize {
            return Err(ExecutionError::generic("Norm dispatch exceeds GPU limits"));
        }
        let count = CubeCount::Static(gx as u32, gy as u32, 1);
        let dim = CubeDim::new_1d(128);
        let result = empty_device_contiguous_dtype(
            client.clone(),
            x.device.clone(),
            x.meta.shape().clone(),
            x.dtype,
        );
        macro_rules! launch {
            ($ty:ty) => {
                normalize::launch::<$ty>(
                    &client,
                    count,
                    dim,
                    x.into_tensor_arg(),
                    gamma.into_tensor_arg(),
                    beta.into_tensor_arg(),
                    result.clone().into_tensor_arg(),
                    1.0f32 / c as f32,
                    self.epsilon,
                    c,
                    slots,
                    rows_per_group,
                )
            };
        }
        match x.dtype {
            DType::F16 => launch!(half::f16),
            DType::BF16 => launch!(half::bf16),
            DType::F32 => launch!(f32),
            _ => return Err(ExecutionError::generic("Unsupported norm dtype")),
        };
        handles.register_float_tensor::<CubeBackend>(&out.id, result);
        Ok(())
    }
}
pub fn forward<const D: usize>(
    x: BurnTensor<D>,
    gamma: BurnTensor<1>,
    beta: BurnTensor<1>,
    epsilon: f64,
) -> Option<BurnTensor<D>> {
    let shape = x.dims();
    let c = shape[D - 1];
    if c > 1536 {
        return None;
    }
    let rows = shape.iter().product::<usize>() / c;
    let x = x
        .reshape([1, rows, c])
        .inner()
        .try_into_primitive::<Gpu>()
        .ok()?;
    let gamma = gamma.inner().try_into_primitive::<Gpu>().ok()?;
    let beta = beta.inner().try_into_primitive::<Gpu>().ok()?;
    let client = x.client.clone();
    let compute_client = client.device().client();
    let props = compute_client.properties();
    let hw = &props.hardware;
    if !props
        .features
        .plane
        .contains(cubecl::ir::features::Plane::Ops)
        || hw.plane_size_min < 2
        || hw.plane_size_max > 128
        || hw.max_units_per_cube < 128
        || c.div_ceil(hw.plane_size_min as usize) > 64
    {
        return None;
    }
    let out = TensorIr::uninit(client.create_empty_handle(), x.shape.clone(), x.dtype);
    let desc = CustomOpIr::with_scalars(
        "lino_subgroup_norm_v1",
        &[x.into_ir(), gamma.into_ir(), beta.into_ir()],
        &[out],
        vec![ScalarIr::UInt((epsilon as f32).to_bits() as u64)],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            NormOp {
                desc,
                epsilon: epsilon as f32,
            },
        )
        .outputs();
    Some(BurnTensor::<3>::from_primitive::<Gpu>(out).reshape(shape))
}
