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
fn normalize_3d<F: Float>(
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
    let d0 = x.shape(0);
    let d1 = x.shape(1);
    let total_rows = d0 * d1;
    let s0 = x.stride(0);
    let s1 = x.stride(1);
    let s2 = x.stride(2);
    for tile in 0..rows_per_group / planes {
        let row = group * rows_per_group + tile * planes + usize::cast_from(PLANE_POS);
        let valid = row < total_rows;
        let batch = row / d1;
        let seq = row % d1;
        let base = batch * s0 + seq * s1;
        let out_base = row * channels;
        let mut values = Array::<f32>::new(slots);
        let mut sum = f32::new(0.0_f32);
        #[unroll]
        for j in 0..slots {
            let channel = lane + j * plane;
            values[j] = 0.0_f32;
            if valid && channel < channels {
                let value = f32::cast_from(x[base + channel * s2]);
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
                out[out_base + channel] =
                    F::cast_from((values[j] - mean) * inverse * gamma[channel] + beta[channel]);
            }
        }
    }
}

#[cube(launch)]
fn normalize_4d<F: Float>(
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
    let d0 = x.shape(0);
    let d1 = x.shape(1);
    let d2 = x.shape(2);
    let d12 = d1 * d2;
    let total_rows = d0 * d12;
    let s0 = x.stride(0);
    let s1 = x.stride(1);
    let s2 = x.stride(2);
    let s3 = x.stride(3);
    for tile in 0..rows_per_group / planes {
        let row = group * rows_per_group + tile * planes + usize::cast_from(PLANE_POS);
        let valid = row < total_rows;
        let i0 = row / d12;
        let rem = row % d12;
        let i1 = rem / d2;
        let i2 = rem % d2;
        let base = i0 * s0 + i1 * s1 + i2 * s2;
        let out_base = row * channels;
        let mut values = Array::<f32>::new(slots);
        let mut sum = f32::new(0.0_f32);
        #[unroll]
        for j in 0..slots {
            let channel = lane + j * plane;
            values[j] = 0.0_f32;
            if valid && channel < channels {
                let value = f32::cast_from(x[base + channel * s3]);
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
                out[out_base + channel] =
                    F::cast_from((values[j] - mean) * inverse * gamma[channel] + beta[channel]);
            }
        }
    }
}

#[derive(Debug)]
struct NormOp {
    desc: CustomOpIr,
    epsilon: f32,
    ndims: usize,
    reuse: bool,
}
impl Operation<FusionCubeRuntime> for NormOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([x, gamma, beta], [out]) = self.desc.as_fixed();
        let [x, gamma, beta] =
            [x, gamma, beta].map(|ir| handles.get_float_tensor::<CubeBackend>(ir));
        let (rows, c) = if self.ndims == 4 {
            let [d0, d1, d2, c] = x.meta.shape().dims::<4>();
            (d0 * d1 * d2, c)
        } else {
            let [b, n, c] = x.meta.shape().dims::<3>();
            (b * n, c)
        };
        let client = x.client.clone();
        let hw = &client.properties().hardware;
        let rows_per_group = 128 / hw.plane_size_min as usize;
        let slots = c.div_ceil(hw.plane_size_min as usize);
        let groups = rows.div_ceil(rows_per_group);
        let gx = groups.min(hw.max_cube_count.0 as usize).max(1);
        let gy = groups.div_ceil(gx);
        if gy > hw.max_cube_count.1 as usize {
            return Err(ExecutionError::generic("Norm dispatch exceeds GPU limits"));
        }
        let count = CubeCount::Static(gx as u32, gy as u32, 1);
        let dim = CubeDim::new_1d(128);
        let contiguous = if self.ndims == 4 {
            let [_, b, n, c] = x.meta.shape().dims::<4>();
            **x.meta.strides() == [b * n * c, n * c, c, 1]
        } else {
            let [_, n, c] = x.meta.shape().dims::<3>();
            **x.meta.strides() == [n * c, c, 1]
        };
        // Both kernels load their complete row into registers before writing.
        // Ownership is essential: residual tensors must never be overwritten.
        let result = if self.reuse && contiguous && x.can_mut() {
            x.clone()
        } else {
            empty_device_contiguous_dtype(
                client.clone(),
                x.device.clone(),
                x.meta.shape().clone(),
                x.dtype,
            )
        };
        macro_rules! launch {
            ($ty:ty) => {
                if self.ndims == 4 {
                    normalize_4d::launch::<$ty>(
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
                } else {
                    normalize_3d::launch::<$ty>(
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
                }
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
    reuse: bool,
) -> Option<BurnTensor<D>> {
    let shape = x.dims();
    let c = shape[D - 1];
    if c > 1536 {
        return None;
    }
    let reshaped = D != 3 && D != 4;
    let ndims = if reshaped { 3 } else { D };
    let x_prim = if reshaped {
        let rows = shape.iter().product::<usize>() / c;
        x.reshape([1, rows, c])
            .inner()
            .try_into_primitive::<Gpu>()
            .ok()?
    } else {
        x.inner().try_into_primitive::<Gpu>().ok()?
    };
    let gamma = gamma.inner().try_into_primitive::<Gpu>().ok()?;
    let beta = beta.inner().try_into_primitive::<Gpu>().ok()?;
    let client = x_prim.client.clone();
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
    let out = TensorIr::uninit(
        client.create_empty_handle(),
        x_prim.shape.clone(),
        x_prim.dtype,
    );
    let desc = CustomOpIr::with_scalars(
        "lino_subgroup_norm_v1",
        &[x_prim.into_ir(), gamma.into_ir(), beta.into_ir()],
        &[out],
        vec![
            ScalarIr::UInt((epsilon as f32).to_bits() as u64),
            ScalarIr::UInt(reuse as u64),
        ],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            NormOp {
                desc,
                epsilon: epsilon as f32,
                ndims,
                reuse,
            },
        )
        .outputs();
    if reshaped {
        Some(BurnTensor::<3>::from_primitive::<Gpu>(out).reshape(shape))
    } else {
        Some(BurnTensor::<D>::from_primitive::<Gpu>(out))
    }
}
