//! Short MAB attention over [B,N,C] Q and [B,M,2*C] packed KV.
//! Preserves pre-dot query scaling and online softmax; fuses Q residual.
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
use std::sync::atomic::{AtomicU64, Ordering};
type Gpu = Fusion<CubeBackend>;
static CALLS: AtomicU64 = AtomicU64::new(0);

#[cube(launch)]
fn attend<F: Float>(
    q: &Tensor<F>,
    kv: &Tensor<F>,
    out: &mut Tensor<F>,
    scale: f32,
    #[comptime] heads: usize,
    #[comptime] width: usize,
    #[comptime] slots: usize,
    #[comptime] rows_per_group: usize,
) {
    let plane = usize::cast_from(PLANE_DIM);
    let lane = usize::cast_from(UNIT_POS_PLANE);
    let planes = usize::cast_from(CUBE_DIM) / plane;
    let group = usize::cast_from(CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X);
    let queries = q.shape(1);
    let channels = heads * width;
    for tile in 0..rows_per_group / planes {
        let row = group * rows_per_group + tile * planes + usize::cast_from(PLANE_POS);
        let valid = row < q.shape(0) * heads * queries;
        let query = row % queries;
        let head = (row / queries) % heads;
        let batch = row / (queries * heads);
        let qb = batch * q.stride(0) + query * q.stride(1) + head * width * q.stride(2);
        let mut qq = Array::<f32>::new(slots);
        let mut acc = Array::<f32>::new(slots);
        #[unroll]
        for j in 0..slots {
            let ch = lane + j * plane;
            qq[j] = 0.0_f32;
            acc[j] = 0.0_f32;
            if valid && ch < width {
                qq[j] = f32::cast_from(q[qb + ch * q.stride(2)]) * scale;
            }
        }
        let mut maximum = f32::new(-3.402823e38_f32);
        let mut sum = f32::new(0.0_f32);
        for key in 0..kv.shape(1) {
            let base = batch * kv.stride(0) + key * kv.stride(1) + head * width * kv.stride(2);
            let mut dot = f32::new(0.0_f32);
            #[unroll]
            for j in 0..slots {
                let ch = lane + j * plane;
                if valid && ch < width {
                    dot += qq[j] * f32::cast_from(kv[base + ch * kv.stride(2)]);
                }
            }
            let score = plane_sum(dot);
            let next = f32::max(maximum, score);
            let correction = f32::exp(maximum - next);
            let probability = f32::exp(score - next);
            sum = sum * correction + probability;
            #[unroll]
            for j in 0..slots {
                let ch = lane + j * plane;
                if valid && ch < width {
                    acc[j] = acc[j] * correction
                        + probability * f32::cast_from(kv[base + (channels + ch) * kv.stride(2)]);
                }
            }
            maximum = next;
        }
        let inv_sum = 1.0_f32 / sum;
        #[unroll]
        for j in 0..slots {
            let ch = lane + j * plane;
            if valid && ch < width {
                let attention = F::cast_from(acc[j] * inv_sum);
                out[((batch * queries + query) * heads + head) * width + ch] =
                    q[qb + ch * q.stride(2)] + attention;
            }
        }
    }
}

#[derive(Debug)]
struct AttentionOp {
    desc: CustomOpIr,
    heads: usize,
    scale: f32,
}
impl Operation<FusionCubeRuntime> for AttentionOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([q, kv], [out]) = self.desc.as_fixed();
        let [q, kv] = [q, kv].map(|ir| handles.get_float_tensor::<CubeBackend>(ir));
        let [b, n, c] = q.meta.shape().dims::<3>();
        let width = c / self.heads;
        let client = q.client.clone();
        let hw = &client.properties().hardware;
        let rows_per_group = 128 / hw.plane_size_min as usize;
        let groups = (b * n * self.heads).div_ceil(rows_per_group);
        let gy = groups.div_ceil(hw.max_cube_count.0 as usize).max(1);
        let gx = groups.div_ceil(gy).max(1);
        if gy > hw.max_cube_count.1 as usize {
            return Err(ExecutionError::generic(
                "Packed attention exceeds grid limits",
            ));
        }
        let result = empty_device_contiguous_dtype(
            client.clone(),
            q.device.clone(),
            q.meta.shape().clone(),
            q.dtype,
        );
        let count = CubeCount::Static(gx as u32, gy as u32, 1);
        let dim = CubeDim::new_1d(128);
        macro_rules! launch {
            ($ty:ty) => {
                attend::launch::<$ty>(
                    &client,
                    count,
                    dim,
                    q.into_tensor_arg(),
                    kv.into_tensor_arg(),
                    result.clone().into_tensor_arg(),
                    self.scale,
                    self.heads,
                    width,
                    width.div_ceil(hw.plane_size_min as usize),
                    rows_per_group,
                )
            };
        }
        match q.dtype {
            DType::F16 => launch!(half::f16),
            DType::BF16 => launch!(half::bf16),
            DType::F32 => launch!(f32),
            _ => {
                return Err(ExecutionError::generic(
                    "Unsupported packed attention dtype",
                ));
            }
        };
        CALLS.fetch_add(1, Ordering::Relaxed);
        handles.register_float_tensor::<CubeBackend>(&out.id, result);
        Ok(())
    }
}

pub fn forward(
    q: BurnTensor<3>,
    kv: BurnTensor<3>,
    heads: usize,
    scale: f64,
    valid_m: usize,
) -> Option<BurnTensor<3>> {
    let [b, _, c] = q.dims();
    let [kb, m, kc] = kv.dims();
    if heads == 0
        || c % heads != 0
        || c / heads > 128
        || m == 0
        || m > 64
        || valid_m != m
        || kb != b
        || kc != 2 * c
    {
        return None;
    }
    let q = q.inner().try_into_primitive::<Gpu>().ok()?;
    let kv = kv.inner().try_into_primitive::<Gpu>().ok()?;
    let client = q.client.clone();
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
    let out = TensorIr::uninit(client.create_empty_handle(), q.shape.clone(), q.dtype);
    let scale = scale as f32;
    let desc = CustomOpIr::with_scalars(
        "lino_packed_mab_v1",
        &[q.into_ir(), kv.into_ir()],
        &[out],
        vec![
            ScalarIr::UInt(heads as u64),
            ScalarIr::UInt(scale.to_bits() as u64),
        ],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            AttentionOp { desc, heads, scale },
        )
        .outputs();
    Some(BurnTensor::from_primitive::<Gpu>(out))
}
pub fn stats() -> u64 {
    CALLS.load(Ordering::Relaxed)
}
