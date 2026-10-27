//! Four PBR final heads in one subgroup reduction and FP32 output packing kernel.
use burn::tensor::Tensor as BurnTensor;
use burn_backend::{DType, ExecutionError, Shape};
use burn_cubecl::{
    CubeBackend,
    fusion::{CubeFusionHandle, FusionCubeRuntime},
    ops::numeric::empty_device_contiguous_dtype,
};
use burn_fusion::{
    Fusion,
    custom::{
        CustomOpIr, HandleContainer, Operation, OperationIr, OperationOutput, StreamId, TensorIr,
    },
};
use cubecl::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
type Gpu = Fusion<CubeBackend>;
static CALLS: AtomicU64 = AtomicU64::new(0);

#[cube(launch)]
fn project<F: Float>(
    hidden: &Tensor<F>,
    weights: &Tensor<F>,
    bias: &Tensor<F>,
    out: &mut Tensor<f32>,
    #[comptime] slots: usize,
    #[comptime] rows_per_group: usize,
) {
    let plane = usize::cast_from(PLANE_DIM);
    let lane = usize::cast_from(UNIT_POS_PLANE);
    let planes = usize::cast_from(CUBE_DIM) / plane;
    let group = usize::cast_from(CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X);
    let n = hidden.shape(1);
    for tile in 0..rows_per_group / planes {
        let row = group * rows_per_group + tile * planes + usize::cast_from(PLANE_POS);
        let valid = row < hidden.shape(0) * n;
        let base = (row / n) * hidden.stride(0) + (row % n) * hidden.stride(1);
        let mut accum = Array::<f32>::new(8usize);
        #[unroll]
        for k in 0..8usize {
            accum[k] = 0.0_f32;
        }
        #[unroll]
        for j in 0..slots {
            let channel = lane + j * plane;
            #[unroll]
            for k in 0..8usize {
                let head = if k < 3usize {
                    0usize
                } else if k < 6usize {
                    1usize
                } else if k == 6usize {
                    2usize
                } else {
                    3usize
                };
                if valid && channel < 192usize {
                    accum[k] += f32::cast_from(
                        hidden[base + (head * 192usize + channel) * hidden.stride(2)],
                    ) * f32::cast_from(
                        weights[channel * weights.stride(0) + k * weights.stride(1)],
                    );
                }
            }
        }
        let mut values = Array::<f32>::new(8usize);
        #[unroll]
        for k in 0..8usize {
            let product = F::cast_from(plane_sum(accum[k]));
            values[k] = f32::cast_from(product + bias[k]);
        }
        if valid && lane == 0usize {
            let length = f32::max(
                f32::sqrt(values[0] * values[0] + values[1] * values[1] + values[2] * values[2]),
                1e-12_f32,
            );
            for k in 0..8usize {
                let mut value = values[k];
                if k < 3usize {
                    value = value / length;
                }
                out[row * 8usize + k] = value;
            }
        }
    }
}

#[derive(Debug)]
struct ProjectionOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for ProjectionOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([h, w, b], [out]) = self.desc.as_fixed();
        let [h, w, b] = [h, w, b].map(|ir| handles.get_float_tensor::<CubeBackend>(ir));
        let [batch, n, _] = h.meta.shape().dims::<3>();
        let client = h.client.clone();
        let hw = &client.properties().hardware;
        let rows_per_group = 128 / hw.plane_size_min as usize;
        let groups = (batch * n).div_ceil(rows_per_group);
        let gy = groups.div_ceil(hw.max_cube_count.0 as usize).max(1);
        let gx = groups.div_ceil(gy).max(1);
        if gy > hw.max_cube_count.1 as usize {
            return Err(ExecutionError::generic(
                "PBR projection exceeds grid limits",
            ));
        }
        let result = empty_device_contiguous_dtype(
            client.clone(),
            h.device.clone(),
            Shape::new([batch, n, 8]),
            DType::F32,
        );
        let count = CubeCount::Static(gx as u32, gy as u32, 1);
        let dim = CubeDim::new_1d(128);
        macro_rules! launch {
            ($ty:ty) => {
                project::launch::<$ty>(
                    &client,
                    count,
                    dim,
                    h.into_tensor_arg(),
                    w.into_tensor_arg(),
                    b.into_tensor_arg(),
                    result.clone().into_tensor_arg(),
                    192usize.div_ceil(hw.plane_size_min as usize),
                    rows_per_group,
                )
            };
        }
        match h.dtype {
            DType::F16 => launch!(half::f16),
            DType::BF16 => launch!(half::bf16),
            DType::F32 => launch!(f32),
            _ => return Err(ExecutionError::generic("Unsupported PBR dtype")),
        };
        CALLS.fetch_add(1, Ordering::Relaxed);
        handles.register_float_tensor::<CubeBackend>(&out.id, result);
        Ok(())
    }
}
pub fn forward(
    hidden: BurnTensor<3>,
    weights: BurnTensor<2>,
    bias: BurnTensor<1>,
) -> Option<BurnTensor<3>> {
    let [b, n, c] = hidden.dims();
    if c != 768 || weights.dims() != [192, 8] || bias.dims() != [8] {
        return None;
    }
    let h = hidden.inner().try_into_primitive::<Gpu>().ok()?;
    let w = weights.inner().try_into_primitive::<Gpu>().ok()?;
    let bias = bias.inner().try_into_primitive::<Gpu>().ok()?;
    let client = h.client.clone();
    let compute = client.device().client();
    let props = compute.properties();
    let hw = &props.hardware;
    if !props
        .features
        .plane
        .contains(cubecl::ir::features::Plane::Ops)
        || hw.plane_size_min < 2
        || hw.plane_size_max > 128
        || hw.max_units_per_cube < 128
        || hw.max_cube_dim.0 < 128
    {
        return None;
    }
    let out = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([b, n, 8]),
        DType::F32,
    );
    let desc = CustomOpIr::with_scalars(
        "lino_pbr_projection_v1",
        &[h.into_ir(), w.into_ir(), bias.into_ir()],
        &[out],
        vec![],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            ProjectionOp { desc },
        )
        .outputs();
    Some(BurnTensor::from_primitive::<Gpu>(out))
}
pub fn stats() -> u64 {
    CALLS.load(Ordering::Relaxed)
}
