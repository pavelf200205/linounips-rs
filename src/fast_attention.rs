//! Fused subgroup scale+softmax for padded GEMM attention and compact short-key attention.
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
        CustomOpIr, HandleContainer, Operation, OperationIr, OperationOutput, ScalarIr, StreamId,
        TensorIr,
    },
};
use cubecl::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

type Gpu = Fusion<CubeBackend>;

#[cube(launch)]
fn scale_softmax_plane<F: Float>(
    scores: &Tensor<F>,
    out: &mut Tensor<F>,
    scale: f32,
    valid_m: u32,
    #[comptime] rows_per_group: usize,
) {
    let plane = usize::cast_from(PLANE_DIM);
    let lane = usize::cast_from(UNIT_POS_PLANE);
    let planes = usize::cast_from(CUBE_DIM) / plane;
    let group = usize::cast_from(CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X);
    let b_dim = scores.shape(0);
    let q_dim = scores.shape(1);
    let m_dim = scores.shape(2);
    let active_m = usize::cast_from(valid_m);
    let total_rows = b_dim * q_dim;
    let s0 = scores.stride(0);
    let s1 = scores.stride(1);
    let s2 = scores.stride(2);
    let steps = (m_dim + plane - 1usize) / plane;
    for tile in 0..rows_per_group / planes {
        let row = group * rows_per_group + tile * planes + usize::cast_from(PLANE_POS);
        let valid = row < total_rows;
        let b_idx = row / q_dim;
        let q_idx = row % q_dim;
        let base = b_idx * s0 + q_idx * s1;
        let out_base = row * m_dim;
        let mut local_max = f32::new(-3.402823e38_f32);
        let mut local_sum = f32::new(0.0_f32);
        for step in 0..steps {
            let col = step * plane + lane;
            if valid && col < active_m {
                let val = f32::cast_from(scores[base + col * s2]) * scale;
                let next = f32::max(local_max, val);
                local_sum = local_sum * f32::exp(local_max - next) + f32::exp(val - next);
                local_max = next;
            }
        }
        let row_max = plane_max(local_max);
        let row_sum = plane_sum(local_sum * f32::exp(local_max - row_max));
        let inv_sum = 1.0_f32 / row_sum;
        for step in 0..steps {
            let col = step * plane + lane;
            if valid && col < m_dim {
                if col < active_m {
                    let val = f32::cast_from(scores[base + col * s2]) * scale;
                    out[out_base + col] = F::cast_from(f32::exp(val - row_max) * inv_sum);
                } else {
                    out[out_base + col] = F::cast_from(0.0_f32);
                }
            }
        }
    }
}

#[cube(launch)]
fn scale_softmax_scalar<F: Float>(
    scores: &Tensor<F>,
    out: &mut Tensor<F>,
    scale: f32,
    valid_m: u32,
) {
    let row = usize::cast_from(ABSOLUTE_POS_X + ABSOLUTE_POS_Y * CUBE_COUNT_X * CUBE_DIM_X);
    let b_dim = scores.shape(0);
    let q_dim = scores.shape(1);
    let m_dim = scores.shape(2);
    let active_m = usize::cast_from(valid_m);
    if row < b_dim * q_dim {
        let b_idx = row / q_dim;
        let q_idx = row % q_dim;
        let base = b_idx * scores.stride(0) + q_idx * scores.stride(1);
        let s2 = scores.stride(2);
        let out_base = row * m_dim;
        let mut maximum = f32::new(-3.402823e38_f32);
        let mut sum = f32::new(0.0_f32);
        for col in 0..active_m {
            let val = f32::cast_from(scores[base + col * s2]) * scale;
            let next = f32::max(maximum, val);
            sum = sum * f32::exp(maximum - next) + f32::exp(val - next);
            maximum = next;
        }
        let inv_sum = 1.0_f32 / sum;
        for col in 0..m_dim {
            if col < active_m {
                let val = f32::cast_from(scores[base + col * s2]) * scale;
                out[out_base + col] = F::cast_from(f32::exp(val - maximum) * inv_sum);
            } else {
                out[out_base + col] = F::cast_from(0.0_f32);
            }
        }
    }
}

#[cube(launch)]
fn online_short_plane<F: Float>(
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
    let b_dim = q.shape(0);
    let h_dim = q.shape(1);
    let q_dim = q.shape(2);
    let k_dim = k.shape(2);
    let total_rows = b_dim * h_dim * q_dim;
    let q_s0 = q.stride(0);
    let q_s1 = q.stride(1);
    let q_s2 = q.stride(2);
    let q_s3 = q.stride(3);
    let k_s0 = k.stride(0);
    let k_s1 = k.stride(1);
    let k_s2 = k.stride(2);
    let k_s3 = k.stride(3);
    let v_s0 = v.stride(0);
    let v_s1 = v.stride(1);
    let v_s2 = v.stride(2);
    let v_s3 = v.stride(3);
    for tile in 0..rows_per_group / planes {
        let row = group * rows_per_group + tile * planes + usize::cast_from(PLANE_POS);
        let valid = row < total_rows;
        let query = row % q_dim;
        let head = (row / q_dim) % h_dim;
        let batch = row / (q_dim * h_dim);
        let qb = batch * q_s0 + head * q_s1 + query * q_s2;
        let kb = batch * k_s0 + head * k_s1;
        let vb = batch * v_s0 + head * v_s1;
        let mut qq = Array::<f32>::new(slots);
        let mut acc = Array::<f32>::new(slots);
        #[unroll]
        for j in 0..slots {
            let channel = lane + j * plane;
            qq[j] = 0.0_f32;
            acc[j] = 0.0_f32;
            if valid && channel < width {
                qq[j] = f32::cast_from(q[qb + channel * q_s3]) * scale;
            }
        }
        let mut maximum = f32::new(-3.402823e38_f32);
        let mut sum = f32::new(0.0_f32);
        for key in 0..k_dim {
            let k_row = kb + key * k_s2;
            let v_row = vb + key * v_s2;
            let mut dot = f32::new(0.0_f32);
            #[unroll]
            for j in 0..slots {
                let ch = lane + j * plane;
                if valid && ch < width {
                    dot += qq[j] * f32::cast_from(k[k_row + ch * k_s3]);
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
                    acc[j] =
                        acc[j] * correction + probability * f32::cast_from(v[v_row + ch * v_s3]);
                }
            }
            maximum = next;
        }
        let inv_sum = 1.0_f32 / sum;
        let out_row = ((batch * q_dim + query) * h_dim + head) * width;
        #[unroll]
        for j in 0..slots {
            let ch = lane + j * plane;
            if valid && ch < width {
                out[out_row + ch] = F::cast_from(acc[j] * inv_sum);
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
    let b_dim = q.shape(0);
    let h_dim = q.shape(1);
    let q_dim = out.shape(1);
    let k_dim = k.shape(2);
    if row < b_dim * h_dim * q_dim {
        let query = row % q_dim;
        let head = (row / q_dim) % h_dim;
        let batch = row / (q_dim * h_dim);
        let qb = batch * q.stride(0) + head * q.stride(1) + query * q.stride(2);
        let kb = batch * k.stride(0) + head * k.stride(1);
        let vb = batch * v.stride(0) + head * v.stride(1);
        let q_s3 = q.stride(3);
        let k_s2 = k.stride(2);
        let k_s3 = k.stride(3);
        let v_s2 = v.stride(2);
        let v_s3 = v.stride(3);
        let mut qq = Array::<f32>::new(width);
        let mut acc = Array::<f32>::new(width);
        for ch in 0..width {
            qq[ch] = f32::cast_from(q[qb + ch * q_s3]) * scale;
            acc[ch] = 0.0_f32;
        }
        let mut maximum = f32::new(-3.402823e38_f32);
        let mut sum = f32::new(0.0_f32);
        for key in 0..k_dim {
            let k_row = kb + key * k_s2;
            let v_row = vb + key * v_s2;
            let mut dot = f32::new(0.0_f32);
            for ch in 0..width {
                dot += qq[ch] * f32::cast_from(k[k_row + ch * k_s3]);
            }
            let score = dot;
            let next = f32::max(maximum, score);
            let correction = f32::exp(maximum - next);
            let probability = f32::exp(score - next);
            sum = sum * correction + probability;
            for ch in 0..width {
                acc[ch] = acc[ch] * correction + probability * f32::cast_from(v[v_row + ch * v_s3]);
            }
            maximum = next;
        }
        let inv_sum = 1.0_f32 / sum;
        let out_row = ((batch * q_dim + query) * h_dim + head) * width;
        for ch in 0..width {
            out[out_row + ch] = F::cast_from(acc[ch] * inv_sum);
        }
    }
}

#[derive(Debug)]
struct SoftmaxOp {
    desc: CustomOpIr,
    scale: f32,
    valid_m: u32,
    reuse: bool,
}
impl Operation<FusionCubeRuntime> for SoftmaxOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([scores], [out]) = self.desc.as_fixed();
        let scores = handles.get_float_tensor::<CubeBackend>(scores);
        let [bb, qq, mm] = scores.meta.shape().dims::<3>();
        let rows = bb * qq;
        let client = scores.client.clone();
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
                "Softmax dispatch exceeds GPU limits",
            ));
        }
        let count = CubeCount::Static(gx as u32, gy as u32, 1);
        let dim = CubeDim::new_1d(128);
        // Each row finishes its reductions before writing; no other row reads it.
        let reuse = self.reuse && scores.can_mut() && **scores.meta.strides() == [qq * mm, mm, 1];
        let result = if reuse {
            scores.clone()
        } else {
            empty_device_contiguous_dtype(
                client.clone(),
                scores.device.clone(),
                scores.meta.shape().clone(),
                scores.dtype,
            )
        };
        macro_rules! launch {
            ($ty:ty) => {{
                if plane {
                    scale_softmax_plane::launch::<$ty>(
                        &client,
                        count,
                        dim,
                        scores.into_tensor_arg(),
                        result.clone().into_tensor_arg(),
                        self.scale,
                        self.valid_m,
                        rows_per_group,
                    );
                } else {
                    scale_softmax_scalar::launch::<$ty>(
                        &client,
                        count,
                        dim,
                        scores.into_tensor_arg(),
                        result.clone().into_tensor_arg(),
                        self.scale,
                        self.valid_m,
                    );
                }
            }};
        }
        match result.dtype {
            DType::F16 => launch!(half::f16),
            DType::BF16 => launch!(half::bf16),
            DType::F32 => launch!(f32),
            _ => return Err(ExecutionError::generic("Unsupported softmax dtype")),
        };
        record(if plane { 0 } else { 1 });
        handles.register_float_tensor::<CubeBackend>(&out.id, result);
        Ok(())
    }
}

#[derive(Debug)]
struct ShortAttentionOp {
    desc: CustomOpIr,
    scale: f32,
}
impl Operation<FusionCubeRuntime> for ShortAttentionOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([q, k, v], [out]) = self.desc.as_fixed();
        let [q, k, v] = [q, k, v].map(|ir| handles.get_float_tensor::<CubeBackend>(ir));
        let [bb, hh, nn, width] = q.meta.shape().dims::<4>();
        let client = q.client.clone();
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
        let rows = bb * hh * nn;
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
        let result = empty_device_contiguous_dtype(
            client.clone(),
            q.device.clone(),
            Shape::new([bb, nn, hh * width]),
            q.dtype,
        );
        macro_rules! launch {
            ($ty:ty) => {{
                if plane {
                    online_short_plane::launch::<$ty>(
                        &client,
                        count,
                        dim,
                        q.into_tensor_arg(),
                        k.into_tensor_arg(),
                        v.into_tensor_arg(),
                        result.clone().into_tensor_arg(),
                        self.scale,
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
                        self.scale,
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
        record(if plane { 2 } else { 3 });
        handles.register_float_tensor::<CubeBackend>(&out.id, result);
        Ok(())
    }
}

/// Fused subgroup scale + masked softmax on a 3D score tensor `[B, Q, M]`.
/// Columns `>= valid_m` are excluded from the softmax statistics and written as `0.0`.
pub fn scale_softmax(
    scores: BurnTensor<3>,
    scale: f64,
    valid_m: usize,
    reuse: bool,
) -> Option<BurnTensor<3>> {
    let scale_f32 = scale as f32;
    let s = scores.inner().try_into_primitive::<Gpu>().ok()?;
    let client = s.client.clone();
    let out = TensorIr::uninit(client.create_empty_handle(), s.shape.clone(), s.dtype);
    let desc = CustomOpIr::with_scalars(
        "lino_scale_softmax_v2",
        &[s.into_ir()],
        &[out],
        vec![
            ScalarIr::UInt(scale_f32.to_bits() as u64),
            ScalarIr::UInt(valid_m as u64),
            ScalarIr::UInt(reuse as u64),
        ],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            SoftmaxOp {
                desc,
                scale: scale_f32,
                valid_m: valid_m as u32,
                reuse,
            },
        )
        .outputs();
    Some(BurnTensor::<3>::from_primitive::<Gpu>(out))
}

/// Fused short-sequence attention (`M <= 64`) returning contiguous `[B, N, H * D]` directly.
pub fn forward(
    q: BurnTensor<4>,
    k: BurnTensor<4>,
    v: BurnTensor<4>,
    scale: f64,
) -> Option<BurnTensor<3>> {
    let [b, h, n, d] = q.dims();
    let keys = k.dims()[2];
    if d > 128 || keys > 64 {
        return None;
    }
    let scale_f32 = scale as f32;
    let q = q.inner().try_into_primitive::<Gpu>().ok()?;
    let k = k.inner().try_into_primitive::<Gpu>().ok()?;
    let v = v.inner().try_into_primitive::<Gpu>().ok()?;
    let client = q.client.clone();
    let out = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([b, n, h * d]),
        q.dtype,
    );
    let desc = CustomOpIr::with_scalars(
        "lino_short_attention_v2",
        &[q.into_ir(), k.into_ir(), v.into_ir()],
        &[out],
        vec![ScalarIr::UInt(scale_f32.to_bits() as u64)],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            ShortAttentionOp {
                desc,
                scale: scale_f32,
            },
        )
        .outputs();
    Some(BurnTensor::<3>::from_primitive::<Gpu>(out))
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
                "gemm + subgroup softmax",
                "gemm + scalar softmax",
                "fused subgroup streaming",
                "scalar streaming fallback"
            ][index]
        ));
    }
}
pub fn stats() -> serde_json::Value {
    serde_json::json!({"gemm_subgroup_softmax":COUNTS[0].load(Ordering::Relaxed),"gemm_scalar_softmax":COUNTS[1].load(Ordering::Relaxed),"subgroup_streaming":COUNTS[2].load(Ordering::Relaxed),"scalar_streaming":COUNTS[3].load(Ordering::Relaxed)})
}
