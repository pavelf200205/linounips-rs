//! Flash attention with explicit LINO scaling and no dense-score fallback.
use burn::tensor::Tensor as BurnTensor;
use burn_backend::{DType, ExecutionError, ops::AttentionModuleOptions};
use burn_cubecl::{
    CubeBackend,
    fusion::{CubeFusionHandle, FusionCubeRuntime},
    kernel::attention::flash_attention,
    ops::numeric::empty_device_contiguous_dtype,
};
use burn_fusion::{
    Fusion,
    custom::{
        CustomOpIr, HandleContainer, Operation, OperationIr, OperationOutput, StreamId, TensorIr,
    },
};
use cubecl::prelude::*;
use cubek::attention::forward::{
    launch::{BlueprintStrategy, Strategy},
    routines::blackbox_accelerated::BlackboxAcceleratedStrategy,
};
use std::{
    collections::HashMap,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};
type Gpu = Fusion<CubeBackend>;
#[cube(launch)]
fn online_plane<F: Float>(
    q: &Tensor<F>,
    k: &Tensor<F>,
    v: &Tensor<F>,
    out: &mut Tensor<F>,
    scale: f32,
    #[comptime] width: usize,
    #[comptime] slots: usize,
    #[comptime] rows_per_group: usize,
) {
    let plane = usize::cast_from(PLANE_DIM);
    let lane = usize::cast_from(UNIT_POS_PLANE);
    let planes = usize::cast_from(CUBE_DIM) / plane;
    let group = usize::cast_from(CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X);
    for tile in 0..rows_per_group / planes {
        let row = group * rows_per_group + tile * planes + usize::cast_from(PLANE_POS);
        let valid = row < q.shape(0) * q.shape(1) * q.shape(2);
        let query = row % q.shape(2);
        let head = (row / q.shape(2)) % q.shape(1);
        let batch = row / (q.shape(2) * q.shape(1));
        let qb = batch * q.stride(0) + head * q.stride(1) + query * q.stride(2);
        let kb = batch * k.stride(0) + head * k.stride(1);
        let vb = batch * v.stride(0) + head * v.stride(1);
        let mut qq = Array::<f32>::new(slots);
        let mut acc = Array::<f32>::new(slots);
        #[unroll]
        for j in 0..slots {
            let channel = lane + j * plane;
            qq[j] = 0.0_f32;
            acc[j] = 0.0_f32;
            if valid && channel < width {
                qq[j] = f32::cast_from(q[qb + channel * q.stride(3)]);
            }
        }
        let mut maximum = f32::new(-3.402823e38_f32);
        let mut sum = f32::new(0.0_f32);
        for key in 0..k.shape(2) {
            let mut dot = f32::new(0.0_f32);
            #[unroll]
            for j in 0..slots {
                let ch = lane + j * plane;
                if valid && ch < width {
                    dot += qq[j] * f32::cast_from(k[kb + key * k.stride(2) + ch * k.stride(3)]);
                }
            }
            let score = plane_sum(dot) * scale;
            let next = f32::max(maximum, score);
            let correction = f32::exp(maximum - next);
            let probability = f32::exp(score - next);
            sum = sum * correction + probability;
            #[unroll]
            for j in 0..slots {
                let ch = lane + j * plane;
                if valid && ch < width {
                    acc[j] = acc[j] * correction
                        + probability
                            * f32::cast_from(v[vb + key * v.stride(2) + ch * v.stride(3)]);
                }
            }
            maximum = next;
        }
        #[unroll]
        for j in 0..slots {
            let ch = lane + j * plane;
            if valid && ch < width {
                out[row * width + ch] = F::cast_from(acc[j] / sum);
            }
        }
    }
}
#[cube(launch)]
fn online_scalar<F: Float>(
    q: &Tensor<F>,
    k: &Tensor<F>,
    v: &Tensor<F>,
    out: &mut Tensor<F>,
    scale: f32,
    #[comptime] width: usize,
) {
    let row = usize::cast_from(ABSOLUTE_POS_X + ABSOLUTE_POS_Y * CUBE_COUNT_X * CUBE_DIM_X);
    if row < q.shape(0) * q.shape(1) * q.shape(2) {
        let query = row % q.shape(2);
        let head = (row / q.shape(2)) % q.shape(1);
        let batch = row / (q.shape(2) * q.shape(1));
        let qb = batch * q.stride(0) + head * q.stride(1) + query * q.stride(2);
        let kb = batch * k.stride(0) + head * k.stride(1);
        let vb = batch * v.stride(0) + head * v.stride(1);
        let mut qq = Array::<f32>::new(width);
        let mut acc = Array::<f32>::new(width);
        #[unroll]
        for ch in 0..width {
            qq[ch] = f32::cast_from(q[qb + ch * q.stride(3)]);
            acc[ch] = 0.0_f32;
        }
        let mut maximum = f32::new(-3.402823e38_f32);
        let mut sum = f32::new(0.0_f32);
        for key in 0..k.shape(2) {
            let mut dot = f32::new(0.0_f32);
            #[unroll]
            for ch in 0..width {
                dot += qq[ch] * f32::cast_from(k[kb + key * k.stride(2) + ch * k.stride(3)]);
            }
            let score = dot * scale;
            let next = f32::max(maximum, score);
            let correction = f32::exp(maximum - next);
            let probability = f32::exp(score - next);
            sum = sum * correction + probability;
            #[unroll]
            for ch in 0..width {
                acc[ch] = acc[ch] * correction
                    + probability * f32::cast_from(v[vb + key * v.stride(2) + ch * v.stride(3)]);
            }
            maximum = next;
        }
        #[unroll]
        for ch in 0..width {
            out[row * width + ch] = F::cast_from(acc[ch] / sum);
        }
    }
}
#[derive(Clone, Copy, Debug)]
enum Plan {
    Accelerated4,
    Accelerated2,
    Unit,
    Online,
}
fn plans() -> &'static Mutex<HashMap<String, Plan>> {
    static PLANS: OnceLock<Mutex<HashMap<String, Plan>>> = OnceLock::new();
    PLANS.get_or_init(|| Mutex::new(HashMap::new()))
}
#[derive(Debug)]
struct AttentionOp {
    desc: CustomOpIr,
    key: String,
    short: bool,
}
impl Operation<FusionCubeRuntime> for AttentionOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([q, k, v], [out]) = self.desc.as_fixed();
        let [q, k, v] = [q, k, v].map(|ir| handles.get_float_tensor::<CubeBackend>(ir));
        let _progress = crate::progress::phase(
            if self.short {
                "fused light attention"
            } else {
                "flash attention"
            },
            1,
            "launch",
        );
        let initial = plans().lock().unwrap().get(&self.key).copied();
        let options = AttentionModuleOptions {
            scale: None,
            softcap: None,
            is_causal: false,
        };
        let strategies = match initial {
            Some(Plan::Accelerated4) => vec![
                Plan::Accelerated4,
                Plan::Accelerated2,
                Plan::Unit,
                Plan::Online,
            ],
            Some(Plan::Accelerated2) => vec![Plan::Accelerated2, Plan::Unit, Plan::Online],
            Some(Plan::Unit) => vec![Plan::Unit, Plan::Online],
            Some(Plan::Online) => vec![Plan::Online],
            None if self.short => vec![Plan::Online],
            None => vec![
                Plan::Accelerated4,
                Plan::Accelerated2,
                Plan::Unit,
                Plan::Online,
            ],
        };
        for plan in strategies {
            let strategy = match plan {
                Plan::Accelerated4 => Some(Strategy::BlackboxAccelerated(
                    BlueprintStrategy::Inferred(BlackboxAcceleratedStrategy {
                        num_planes: 4,
                        seq_q: 1,
                        seq_kv: 1,
                    }),
                )),
                Plan::Accelerated2 => Some(Strategy::BlackboxAccelerated(
                    BlueprintStrategy::Inferred(BlackboxAcceleratedStrategy {
                        num_planes: 2,
                        seq_q: 1,
                        seq_kv: 1,
                    }),
                )),
                Plan::Unit => Some(Strategy::Unit(BlueprintStrategy::Inferred(()))),
                Plan::Online => None,
            };
            if let Some(strategy) = strategy {
                if let Ok(result) = flash_attention(
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    None,
                    None,
                    options,
                    strategy,
                ) {
                    plans().lock().unwrap().insert(self.key.clone(), plan);
                    record(if matches!(plan, Plan::Unit) { 1 } else { 0 });
                    handles.register_float_tensor::<CubeBackend>(&out.id, result);
                    return Ok(());
                }
                continue;
            }
            let client = q.client.clone();
            let width = q.meta.shape().dims::<4>()[3];
            let [bb, hh, nn, _] = q.meta.shape().dims::<4>();
            let rows = bb * hh * nn;
            let props = client.properties();
            let hw = &props.hardware;
            let plane = props
                .features
                .plane
                .contains(cubecl::ir::features::Plane::Ops)
                && hw.plane_size_min >= 2
                && hw.plane_size_max <= 128
                && hw.max_units_per_cube >= 128;
            let rows_per_group = if plane {
                128 / hw.plane_size_min as usize
            } else {
                128
            };
            let groups = rows.div_ceil(rows_per_group);
            let gx = groups.min(hw.max_cube_count.0 as usize).max(1);
            let gy = groups.div_ceil(gx);
            if gy > hw.max_cube_count.1 as usize {
                return Err(ExecutionError::generic(
                    "Attention dispatch exceeds GPU limits",
                ));
            }
            let count = CubeCount::Static(gx as u32, gy as u32, 1);
            let dim = CubeDim::new_1d(128);
            let scale = 1.0f32 / (width as f32).sqrt();
            let result = empty_device_contiguous_dtype(
                client.clone(),
                q.device.clone(),
                q.meta.shape().clone(),
                q.dtype,
            );
            macro_rules! launch {
                ($ty:ty) => {{
                    if plane {
                        online_plane::launch::<$ty>(
                            &client,
                            count,
                            dim,
                            q.into_tensor_arg(),
                            k.into_tensor_arg(),
                            v.into_tensor_arg(),
                            result.clone().into_tensor_arg(),
                            scale,
                            width,
                            width.div_ceil(hw.plane_size_min as usize),
                            rows_per_group,
                        );
                    } else {
                        online_scalar::launch::<$ty>(
                            &client,
                            count,
                            dim,
                            q.into_tensor_arg(),
                            k.into_tensor_arg(),
                            v.into_tensor_arg(),
                            result.clone().into_tensor_arg(),
                            scale,
                            width,
                        );
                    }
                }};
            }
            match q.dtype {
                DType::F16 => launch!(half::f16),
                DType::BF16 => launch!(half::bf16),
                DType::F32 => launch!(f32),
                _ => return Err(ExecutionError::generic("Unsupported attention dtype")),
            };
            plans()
                .lock()
                .unwrap()
                .insert(self.key.clone(), Plan::Online);
            record(if plane { 2 } else { 3 });
            handles.register_float_tensor::<CubeBackend>(&out.id, result);
            return Ok(());
        }
        // A cached strategy can be rejected for changed runtime capabilities. Evict it
        // and report a real execution error, never a dense unbounded fallback.
        plans().lock().unwrap().remove(&self.key);
        Err(ExecutionError::generic(
            "Previously selected flash attention strategy became unavailable",
        ))
    }
}
pub fn forward(
    q: BurnTensor<4>,
    k: BurnTensor<4>,
    v: BurnTensor<4>,
    scale: f64,
) -> Option<BurnTensor<4>> {
    let dims = q.dims();
    let keys = k.dims()[2];
    if dims[3] > 128 {
        return None;
    }
    // CubeK 0.3 flash ignores AttentionModuleOptions.scale. Correct the query
    // explicitly so BOTH Flash and the streaming fallback use LINO's exact scale.
    let correction = scale * (dims[3] as f64).sqrt();
    let q = if (correction - 1.).abs() > 1e-12 {
        q * correction
    } else {
        q
    };
    let original = dims[2];
    let short = keys <= 128;
    let q = if !short && original % 64 != 0 {
        let padded = original.div_ceil(64) * 64;
        let zeros = BurnTensor::<4>::zeros(
            [dims[0], dims[1], padded - original, dims[3]],
            burn::tensor::TensorCreationOptions::new(q.device()).with_dtype(q.dtype()),
        );
        BurnTensor::cat(vec![q, zeros], 2)
    } else {
        q
    };
    let key = format!(
        "{:?}/{:?}/{:?}/{:?}",
        q.device().identity(),
        q.dtype(),
        q.dims(),
        k.dims()
    );
    let q = q.inner().try_into_primitive::<Gpu>().ok()?;
    let k = k.inner().try_into_primitive::<Gpu>().ok()?;
    let v = v.inner().try_into_primitive::<Gpu>().ok()?;
    let client = q.client.clone();
    let out = TensorIr::uninit(client.create_empty_handle(), q.shape.clone(), q.dtype);
    let desc = CustomOpIr::with_scalars(
        "lino_bounded_attention_v1",
        &[q.into_ir(), k.into_ir(), v.into_ir()],
        &[out],
        vec![],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            AttentionOp { desc, key, short },
        )
        .outputs();
    let out = BurnTensor::from_primitive::<Gpu>(out);
    Some(out.slice([0..dims[0], 0..dims[1], 0..original, 0..dims[3]]))
}

static COUNTS: [AtomicU64; 4] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
fn record(index: usize) {
    if COUNTS[index].fetch_add(1, Ordering::Relaxed) == 0 {
        crate::progress::message(format!(
            "Attention path: {}",
            [
                "accelerated flash",
                "portable unit flash",
                "fused subgroup streaming",
                "scalar streaming fallback"
            ][index]
        ));
    }
}
pub fn stats() -> serde_json::Value {
    serde_json::json!({"accelerated":COUNTS[0].load(Ordering::Relaxed),"unit_flash":COUNTS[1].load(Ordering::Relaxed),"subgroup_streaming":COUNTS[2].load(Ordering::Relaxed),"scalar_streaming":COUNTS[3].load(Ordering::Relaxed)})
}
