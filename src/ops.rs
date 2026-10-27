use crate::host::upload;
use anyhow::{Context, Result, ensure};
use burn::tensor::{
    Device, FloatDType, Int, Tensor, TensorCreationOptions, TensorData, activation, module,
    ops::{ConvOptions, ConvTransposeOptions},
};
use safetensors::{Dtype, SafeTensors};
use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, HashMap},
    path::Path,
};

pub struct Ops {
    pub device: Device,
    pub dtype: FloatDType,
    pub score_mib: usize,
    pub reference_kernels: bool,
    pub inflight_mib: usize,
    pending_bytes: Cell<usize>,
    pending_ops: Cell<usize>,
    constants: RefCell<HashMap<String, Tensor<1>>>,
    constant_bytes: Cell<usize>,
    timings: RefCell<BTreeMap<&'static str, (u64, f64)>>,
    weights: HashMap<String, (Vec<usize>, Tensor<1>)>,
}

impl Ops {
    pub fn load(
        path: &Path,
        device: Device,
        dtype: FloatDType,
        score_mib: usize,
        pbr: bool,
    ) -> Result<Self> {
        ensure!(
            device.supports_dtype(dtype),
            "Device does not support {dtype:?}"
        );
        let bytes = std::fs::read(path)?;
        let archive = SafeTensors::deserialize(&bytes)?;
        let material_checkpoint = archive
            .tensor("regressor.prediction_base.out_layer.weight")
            .is_ok();
        ensure!(
            material_checkpoint == pbr,
            "Checkpoint mode mismatch: select --mode {} for this checkpoint",
            if material_checkpoint {
                "pbr"
            } else {
                "standard"
            }
        );
        let expected = crate::schema::expected(pbr);
        let mut raw_linear: HashMap<String, (usize, usize, Vec<f32>)> = HashMap::new();
        let mut weights = HashMap::with_capacity(expected.len() + 16);
        for (name, mut shape) in expected {
            let t = archive.tensor(&name).with_context(|| {
                format!("Missing required tensor {name}; wrong checkpoint mode or architecture")
            })?;
            ensure!(
                t.shape() == shape,
                "{name}: expected {shape:?}, got {:?}",
                t.shape()
            );
            let mut v: Vec<f32> = match t.dtype() {
                Dtype::F32 => t
                    .data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
                Dtype::F16 => t
                    .data()
                    .chunks_exact(2)
                    .map(|b| {
                        half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32()
                    })
                    .collect(),
                Dtype::BF16 => t
                    .data()
                    .chunks_exact(2)
                    .map(|b| {
                        half::bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32()
                    })
                    .collect(),
                _ => anyhow::bail!("Unsupported dtype for {name}"),
            };
            ensure!(v.iter().all(|v| v.is_finite()), "Nonfinite tensor {name}");
            // Store linear weights in GEMM layout once, without a duplicate GPU copy.
            if shape.len() == 2 {
                let (o, i) = (shape[0], shape[1]);
                let mut transposed = vec![0.; v.len()];
                for row in 0..o {
                    let row_off = row * i;
                    for col in 0..i {
                        transposed[col * o + row] = v[row_off + col];
                    }
                }
                if name.ends_with(".fc_k.weight") || name.ends_with(".fc_v.weight") {
                    raw_linear.insert(name, (i, o, transposed));
                    continue;
                }
                v = transposed;
                shape = vec![i, o];
            }
            let n = v.len();
            let mut tensor = upload(v, [n], &device, dtype);
            // Match only LayerNorm parameters (.norm, _norm., .ln0., .ln1.), never `prediction_normal`!
            if shape.len() == 1
                && (name.contains(".norm")
                    || name.contains("_norm.")
                    || name.contains(".ln0.")
                    || name.contains(".ln1."))
            {
                tensor = tensor.cast(FloatDType::F32);
            }
            weights.insert(name, (shape, tensor));
        }

        // Pack paired MAB (fc_k, fc_v) weights into a single [i, 2 * o] GEMM weight tensor
        // with zero additional VRAM compared to storing fc_k and fc_v separately.
        let k_keys: Vec<String> = raw_linear
            .keys()
            .filter(|k| k.ends_with(".fc_k.weight"))
            .cloned()
            .collect();
        for k_name in k_keys {
            let prefix = k_name.strip_suffix(".fc_k.weight").unwrap();
            let v_name = format!("{prefix}.fc_v.weight");
            let (i, o, vk) = raw_linear.remove(&k_name).unwrap();
            let (iv, ov, vv) = raw_linear.remove(&v_name).unwrap();
            debug_assert_eq!((i, o), (iv, ov));
            let mut fused = Vec::with_capacity(i * 2 * o);
            for row in 0..i {
                fused.extend_from_slice(&vk[row * o..(row + 1) * o]);
                fused.extend_from_slice(&vv[row * o..(row + 1) * o]);
            }
            let n = fused.len();
            let tensor = upload(fused, [n], &device, dtype);
            weights.insert(format!("{prefix}.fc_kv.weight"), (vec![i, 2 * o], tensor));
        }

        // Precompute static DINO prefix tokens ([cls + pos[0], reg]) and patch pos_embed
        // to eliminate redundant slicing and concatenation on every encoder batch.
        let d_prefix = "image_encoder.backbone.aggregator.patch_embed";
        if let (Some((_, cls_t)), Some((_, pos_t)), Some((_, reg_t))) = (
            weights.remove(&format!("{d_prefix}.cls_token")),
            weights.remove(&format!("{d_prefix}.pos_embed")),
            weights.remove(&format!("{d_prefix}.register_tokens")),
        ) {
            let cls = cls_t.reshape([1, 1, 384]);
            let pos = pos_t.reshape([1, 1025, 384]);
            let reg = reg_t.reshape([1, 4, 384]);
            let cls_pos = cls + pos.clone().slice([0..1, 0..1, 0..384]);
            let prefix_tokens = Tensor::cat(vec![cls_pos, reg], 1).reshape([5 * 384]);
            let patch_pos = pos.slice([0..1, 1..1025, 0..384]).reshape([1024 * 384]);
            weights.insert(
                format!("{d_prefix}.prefix_tokens"),
                (vec![1, 5, 384], prefix_tokens),
            );
            weights.insert(
                format!("{d_prefix}.patch_pos"),
                (vec![1, 1024, 384], patch_pos),
            );
        }

        // Pre-project the constant decoder aggregation seed `S` by `dec.0.mab.fc_q` once at load
        // time instead of projecting `[2048, 1, 384]` on every pixel group.
        let seed_key = "glc_aggregation.aggregation.dec.0.S";
        let fc_q_key = "glc_aggregation.aggregation.dec.0.mab.fc_q.weight";
        if let (Some((_, s_t)), Some((_, w_t))) =
            (weights.get(seed_key).cloned(), weights.remove(fc_q_key))
        {
            let q_seed = module::linear(s_t.reshape([1, 384]), w_t.reshape([384, 384]), None)
                .reshape([384]);
            weights.insert(
                "glc_aggregation.aggregation.dec.0.S_q".into(),
                (vec![1, 1, 384], q_seed),
            );
        }

        Ok(Self {
            device,
            dtype,
            score_mib,
            weights,
            reference_kernels: false,
            inflight_mib: 64,
            pending_bytes: Cell::new(0),
            pending_ops: Cell::new(0),
            constants: RefCell::new(HashMap::new()),
            constant_bytes: Cell::new(0),
            timings: RefCell::new(BTreeMap::new()),
        })
    }

    pub fn options(&self) -> TensorCreationOptions {
        TensorCreationOptions::new(self.device.clone()).with_dtype(self.dtype.into())
    }

    pub fn w<const D: usize>(&self, key: &str) -> Tensor<D> {
        let (s, t) = &self.weights[key];
        let shape: [usize; D] = s.as_slice().try_into().expect("validated weight rank");
        t.clone().reshape(shape)
    }

    pub fn has(&self, key: &str) -> bool {
        self.weights.contains_key(key)
    }

    pub fn linear(&self, x: Tensor<3>, p: &str) -> Tensor<3> {
        let _timer = self.span("linear");
        let [b, n, c] = x.dims();
        let w: Tensor<2> = self.w(&format!("{p}.weight"));
        let o = w.dims()[1];
        let bias_key = format!("{p}.bias");
        let bias = self.has(&bias_key).then(|| self.w::<1>(&bias_key));
        module::linear(x.reshape([b * n, c]), w, bias).reshape([b, n, o])
    }

    pub fn norm<const D: usize>(&self, x: Tensor<D>, p: &str, eps: f64) -> Tensor<D> {
        let _timer = self.span("layer_norm");
        let weight_key = format!("{p}.weight");
        let bias_key = format!("{p}.bias");
        if !self.reference_kernels {
            if let Some(out) = crate::fast_norm::forward(
                x.clone(),
                self.w::<1>(&weight_key),
                self.w::<1>(&bias_key),
                eps,
            ) {
                return out;
            }
        }
        module::layer_norm(
            x.cast(FloatDType::F32),
            self.w::<1>(&weight_key),
            Some(self.w::<1>(&bias_key)),
            eps,
        )
        .cast(self.dtype)
    }

    pub fn conv(&self, x: Tensor<4>, p: &str, stride: usize, pad: usize) -> Tensor<4> {
        let _timer = self.span("conv");
        let bias_key = format!("{p}.bias");
        let bias = self.has(&bias_key).then(|| self.w::<1>(&bias_key));
        module::conv2d(
            x,
            self.w(&format!("{p}.weight")),
            bias,
            ConvOptions::new([stride; 2], [pad; 2], [1; 2], 1),
        )
    }

    pub fn deconv(&self, x: Tensor<4>, p: &str) -> Tensor<4> {
        let _timer = self.span("transpose_conv");
        module::conv_transpose2d(
            x,
            self.w(&format!("{p}.weight")),
            Some(self.w(&format!("{p}.bias"))),
            ConvTransposeOptions::new([2; 2], [0; 2], [0; 2], [1; 2], 1),
        )
    }

    pub fn indices(&self, v: Vec<i64>) -> Tensor<1, Int> {
        let n = v.len();
        Tensor::from_data(TensorData::new(v, [n]), &self.device)
    }

    pub fn attention(&self, q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, scale: f64) -> Tensor<3> {
        let m = k.dims()[2];
        self.attention_masked(q, k, v, scale, m)
    }

    // Exact attention over `valid_m` keys. For `m > 64`, pads sequence lengths to multiples of 32
    // so CubeK `matmul` uses vectorized `vec4` / Tensor Core tiles on every GEMM while
    // `scale_softmax` masks out padded keys (`col >= valid_m`) with zero effect on outputs.
    pub fn attention_masked(
        &self,
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        scale: f64,
        valid_m: usize,
    ) -> Tensor<3> {
        let _timer = self.span("attention");
        let [b, h, n, d] = q.dims();
        let m = k.dims()[2];
        if !self.reference_kernels && m <= 64 && valid_m == m {
            if let Some(out) =
                crate::fast_attention::forward(q.clone(), k.clone(), v.clone(), scale)
            {
                return out;
            }
        }
        let bh = b * h;
        let q = q.reshape([bh, n, d]);
        let k = k.reshape([bh, m, d]);
        let v = v.reshape([bh, m, d]);
        let elem_bytes = if self.dtype == FloatDType::F32 { 4 } else { 2 };
        let budget_bytes = self.score_mib.max(16) * 1024 * 1024;
        if !self.reference_kernels {
            let pad_m = m.div_ceil(32) * 32;
            let (k_pad, v_pad) = if pad_m > m {
                let z = Tensor::<3>::zeros([bh, pad_m - m, d], self.options());
                (
                    Tensor::cat(vec![k.clone(), z.clone()], 1),
                    Tensor::cat(vec![v.clone(), z], 1),
                )
            } else {
                (k.clone(), v.clone())
            };
            let kt_pad = k_pad.swap_dims(1, 2);
            // Fast path 1: Full [bh, pad_n, pad_m] score matrix fits within `score_mib`
            // (covers DINO 1029->1056, frame_blocks 1031->1056, and regressor.comm.enc 2048).
            let pad_n_32 = n.div_ceil(32) * 32;
            if bh * pad_n_32 * pad_m * elem_bytes <= budget_bytes {
                let q_pad = if pad_n_32 > n {
                    let z = Tensor::<3>::zeros([bh, pad_n_32 - n, d], self.options());
                    Tensor::cat(vec![q.clone(), z], 1)
                } else {
                    q.clone()
                };
                let scores = q_pad.matmul(kt_pad.clone());
                if let Some(probs) = crate::fast_attention::scale_softmax(scores, scale, valid_m) {
                    let out = probs.matmul(v_pad.clone());
                    let out = if pad_n_32 > n {
                        out.slice([0..bh, 0..n, 0..d])
                    } else {
                        out
                    };
                    return out
                        .reshape([b, h, n, d])
                        .permute([0, 2, 1, 3])
                        .reshape([b, n, h * d]);
                }
            }
            // Fast path 2: Uniformly aligned query-chunked GEMM + fused subgroup softmax
            // (covers global_blocks while keeping temporary score buffer <= `score_mib`).
            let bs = if bh % 2 == 0 && (budget_bytes / (2 * pad_m * elem_bytes)) >= 256 {
                2
            } else if (budget_bytes / (bh * pad_m * elem_bytes)) >= 128 {
                bh
            } else {
                1
            };
            let raw_qs = budget_bytes / (bs * pad_m.max(1) * elem_bytes);
            if raw_qs >= 64 {
                let qs = (raw_qs / 64 * 64).clamp(64, 1024);
                let pad_n = n.div_ceil(qs) * qs;
                let q_pad = if pad_n > n {
                    let z = Tensor::<3>::zeros([bh, pad_n - n, d], self.options());
                    Tensor::cat(vec![q.clone(), z], 1)
                } else {
                    q.clone()
                };
                let mut head_groups = Vec::with_capacity(bh.div_ceil(bs));
                let mut ok = true;
                for bi in (0..bh).step_by(bs) {
                    let be = (bi + bs).min(bh);
                    let kt_b = kt_pad.clone().slice([bi..be, 0..d, 0..pad_m]);
                    let v_b = v_pad.clone().slice([bi..be, 0..pad_m, 0..d]);
                    let mut q_chunks = Vec::with_capacity(pad_n / qs);
                    for qi in (0..pad_n).step_by(qs) {
                        let qe = qi + qs;
                        let qq = q_pad.clone().slice([bi..be, qi..qe, 0..d]);
                        let scores = qq.matmul(kt_b.clone());
                        if let Some(probs) =
                            crate::fast_attention::scale_softmax(scores, scale, valid_m)
                        {
                            q_chunks.push(probs.matmul(v_b.clone()));
                        } else {
                            ok = false;
                            break;
                        }
                    }
                    if !ok {
                        break;
                    }
                    let head_out = Tensor::cat(q_chunks, 1);
                    head_groups.push(if pad_n > n {
                        head_out.slice([0..(be - bi), 0..n, 0..d])
                    } else {
                        head_out
                    });
                    self.boundary();
                }
                if ok {
                    let out = if head_groups.len() == 1 {
                        head_groups.pop().unwrap()
                    } else {
                        Tensor::cat(head_groups, 0)
                    };
                    return out
                        .reshape([b, h, n, d])
                        .permute([0, 2, 1, 3])
                        .reshape([b, n, h * d]);
                }
            }
        }
        let m = valid_m.min(m);
        let qs = n.min(256);
        let ks = m.min(1024);
        let bs = (self.score_mib * 1024 * 1024 / (16 * qs * ks))
            .max(1)
            .min(bh);
        let mut output = Tensor::<3>::empty([bh, n, d], self.options());
        for bi in (0..bh).step_by(bs) {
            let be = (bi + bs).min(bh);
            let batch = be - bi;
            for qi in (0..n).step_by(qs) {
                let qe = (qi + qs).min(n);
                let count = qe - qi;
                let qq = q.clone().slice([bi..be, qi..qe, 0..d]);
                let fp = TensorCreationOptions::new(self.device.clone())
                    .with_dtype(FloatDType::F32.into());
                let mut max = Tensor::<3>::full([batch, count, 1], f32::NEG_INFINITY, fp.clone());
                let mut sum = Tensor::<3>::zeros([batch, count, 1], fp.clone());
                let mut acc = Tensor::<3>::zeros([batch, count, d], fp);
                for ki in (0..m).step_by(ks) {
                    let ke = (ki + ks).min(m);
                    let kk = k.clone().slice([bi..be, ki..ke, 0..d]);
                    let vv = v
                        .clone()
                        .slice([bi..be, ki..ke, 0..d])
                        .cast(FloatDType::F32);
                    let scores =
                        qq.clone().matmul(kk.swap_dims(1, 2)).cast(FloatDType::F32) * scale;
                    let next = max.clone().max_pair(scores.clone().max_dim(2));
                    let factor = (max - next.clone()).exp();
                    let prob = (scores - next.clone()).exp();
                    acc = acc * factor.clone() + prob.clone().matmul(vv);
                    sum = sum * factor + prob.sum_dim(2);
                    max = next;
                }
                output = output.slice_assign(
                    [bi..be, qi..qe, 0..d],
                    (acc / sum.clamp_min(1e-20)).cast(self.dtype),
                );
                // Prevent asynchronous queues from retaining unbounded tile intermediates.
                self.schedule(batch * count * (ks * 24 + d * 16));
            }
        }
        output
            .reshape([b, h, n, d])
            .permute([0, 2, 1, 3])
            .reshape([b, n, h * d])
    }

    // FP32 bilinear interpolation in bounded channel slices, then cast immediately.
    pub fn resize(&self, x: Tensor<4>, size: usize, align: bool) -> Tensor<4> {
        let _timer = self.span("resize");
        let [b, c, h, w] = x.dims();
        if h == size && w == size {
            return x;
        }
        if !self.reference_kernels {
            if let Some(out) = crate::geometry::resize(x.clone(), size, align) {
                return out;
            }
        }
        let axis = |n: usize| {
            let mut lo = vec![];
            let mut hi = vec![];
            let mut a = vec![];
            for j in 0..size {
                let f = if align {
                    if size == 1 {
                        0.
                    } else {
                        j as f32 * (n - 1) as f32 / (size - 1) as f32
                    }
                } else {
                    ((j as f32 + 0.5) * n as f32 / size as f32 - 0.5).max(0.)
                };
                let l = (f.floor() as usize).min(n - 1);
                lo.push(l as i64);
                hi.push((l + 1).min(n - 1) as i64);
                a.push(f - l as f32);
            }
            (self.indices(lo), self.indices(hi), a)
        };
        let (yl, yh, ya) = axis(h);
        let (xl, xh, xa) = axis(w);
        let wy = upload(ya, [1, 1, size, 1], &self.device, FloatDType::F32);
        let wx = upload(xa, [1, 1, 1, size], &self.device, FloatDType::F32);
        let mut out = Tensor::<4>::empty([b, c, size, size], self.options());
        for i in (0..c).step_by(16) {
            let e = (i + 16).min(c);
            let z = x
                .clone()
                .slice([0..b, i..e, 0..h, 0..w])
                .cast(FloatDType::F32);
            let a = z.clone().select(3, xl.clone());
            let z = a.clone() + (z.select(3, xh.clone()) - a) * wx.clone();
            let a = z.clone().select(2, yl.clone());
            let z = a.clone() + (z.select(2, yh.clone()) - a) * wy.clone();
            out = out.slice_assign([0..b, i..e, 0..size, 0..size], z.cast(self.dtype));
            self.schedule(b * (e - i) * size * size * 20);
        }
        out
    }

    pub fn leaky(&self, x: Tensor<4>) -> Tensor<4> {
        activation::leaky_relu(x, 0.01)
    }

    // Budget queued temporary work; live model/output tensors are separate.
    pub fn schedule(&self, bytes: usize) {
        let pending = self.pending_bytes.get().saturating_add(bytes);
        let ops = self.pending_ops.get() + 1;
        self.pending_bytes.set(pending);
        self.pending_ops.set(ops);
        if pending >= self.inflight_mib * 1024 * 1024 || ops >= 256 {
            let _wait = self.span("queue_wait");
            self.device.sync().expect("bounded queue completion");
            self.pending_bytes.set(0);
            self.pending_ops.set(0);
        } else if ops % 64 == 0 {
            self.device.flush().expect("GPU submission");
        }
    }

    pub fn boundary(&self) {
        self.device.flush().expect("GPU submission");
    }

    pub fn trim_after_readback(&self) {
        self.pending_bytes.set(0);
        self.pending_ops.set(0);
        if let Some(u) = self.device.memory_pool_usage() {
            let unused = u
                .bytes_reserved
                .saturating_sub(u.bytes_in_use)
                .saturating_sub(u.bytes_padding);
            if unused > 256 * 1024 * 1024 {
                self.device.sync().expect("allocator boundary");
                self.device.memory_cleanup();
            }
        }
    }

    pub fn constant(&self, key: String, values: impl FnOnce() -> Vec<f32>) -> Tensor<1> {
        self.constant_as(key, FloatDType::F32, values)
    }

    pub fn constant_as(
        &self,
        key: String,
        dtype: FloatDType,
        values: impl FnOnce() -> Vec<f32>,
    ) -> Tensor<1> {
        if let Some(t) = self.constants.borrow().get(&key) {
            return t.clone();
        }
        let v = values();
        let bytes = v.len() * if dtype == FloatDType::F32 { 4 } else { 2 };
        let n = v.len();
        let t = upload(v, [n], &self.device, dtype);
        if bytes <= 16 * 1024 * 1024 {
            if self.constant_bytes.get() + bytes > 16 * 1024 * 1024 {
                self.constants.borrow_mut().clear();
                self.constant_bytes.set(0);
            }
            self.constants.borrow_mut().insert(key, t.clone());
            self.constant_bytes.set(self.constant_bytes.get() + bytes);
        }
        t
    }

    pub fn position(&self, x: Tensor<4>) -> Tensor<4> {
        let _timer = self.span("position_embedding");
        let [b, c, h, w] = x.dims();
        let length = h.max(w);
        let lut = self.constant(format!("position/{c}/{h}/{w}"), || {
            let mut lut = Vec::with_capacity(c * length);
            for ch in 0..c {
                let quarter = c / 4;
                let axis = ch / (c / 2);
                let cosine = ch % (c / 2) >= quarter;
                let n = if axis == 0 { w } else { h };
                let frequency = 100f64.powf(-((ch % quarter) as f64) / quarter as f64);
                for j in 0..length {
                    let coordinate = (2. * j as f64 + 1. - n as f64) / (n as f64 * 2f64.sqrt());
                    let coordinate = match self.dtype {
                        FloatDType::BF16 => half::bf16::from_f32(coordinate as f32).to_f32() as f64,
                        FloatDType::F16 => half::f16::from_f32(coordinate as f32).to_f32() as f64,
                        _ => coordinate,
                    };
                    let angle = coordinate * frequency;
                    lut.push(0.1 * (if cosine { angle.cos() } else { angle.sin() }) as f32);
                }
            }
            lut
        });
        if !self.reference_kernels {
            if let Some(out) = crate::geometry::add_position(x.clone(), lut.clone()) {
                return out;
            }
        }
        let mut out = Tensor::<4>::empty([b, c, h, w], self.options());
        for ch in (0..c).step_by(16) {
            let end = (ch + 16).min(c);
            let axis = if ch < c / 2 { w } else { h };
            let values =
                lut.clone()
                    .slice([ch * length..end * length])
                    .reshape([1, end - ch, 1, axis]);
            let values = if ch < c / 2 {
                values
            } else {
                values.swap_dims(2, 3)
            };
            let part = x
                .clone()
                .slice([0..b, ch..end, 0..h, 0..w])
                .cast(FloatDType::F32)
                + values;
            out = out.slice_assign([0..b, ch..end, 0..h, 0..w], part.cast(self.dtype));
            self.schedule(b * (end - ch) * h * w * 8);
        }
        out
    }

    pub fn rotary_cached(
        &self,
        x: Tensor<4>,
        pos_key: Option<&str>,
        max_hint: Option<usize>,
        positions: &[[usize; 2]],
    ) -> Tensor<4> {
        let _timer = self.span("rotary");
        let [b, _, n, d] = x.dims();
        let half = d / 2;
        let quarter = d / 4;
        let maximum = max_hint
            .unwrap_or_else(|| positions.iter().flatten().copied().max().unwrap_or(0));
        let lut = self.constant(format!("rope/lut/{d}/{maximum}"), || {
            let mut v = Vec::with_capacity((maximum + 1) * d);
            for p in 0..=maximum {
                for j in 0..half {
                    let angle = p as f32 / 100f32.powf(2. * (j % quarter) as f32 / half as f32);
                    let angle = match self.dtype {
                        FloatDType::BF16 => half::bf16::from_f32(angle).to_f32(),
                        FloatDType::F16 => half::f16::from_f32(angle).to_f32(),
                        _ => angle,
                    };
                    let round = |v: f32| match self.dtype {
                        FloatDType::BF16 => half::bf16::from_f32(v).to_f32(),
                        FloatDType::F16 => half::f16::from_f32(v).to_f32(),
                        _ => v,
                    };
                    v.extend([round(angle.cos()), round(angle.sin())]);
                }
            }
            v
        });
        let key = if let Some(k) = pos_key {
            format!("rope/positions/{b}/{n}/{k}")
        } else {
            let mut hash = 0xcbf29ce484222325u64;
            for p in positions {
                for v in p {
                    hash = (hash ^ (*v as u64)).wrapping_mul(0x100000001b3);
                }
            }
            format!("rope/positions/{b}/{n}/{hash}")
        };
        let pos = self.constant(key, || {
            positions.iter().flat_map(|p| p.map(|x| x as f32)).collect()
        });
        if !self.reference_kernels {
            if let Some(out) = crate::geometry::rope(x.clone(), lut.clone(), pos) {
                return out;
            }
        }
        let lookup = lut.into_data().try_into_vec::<f32>().expect("rope LUT");
        let mut cos = Vec::with_capacity(positions.len() * d);
        let mut sin = Vec::with_capacity(positions.len() * d);
        for p in positions {
            for axis in 0..2 {
                for j in 0..half {
                    let at = (p[axis] * half + j) * 2;
                    cos.push(lookup[at]);
                    sin.push(lookup[at + 1]);
                }
            }
        }
        let cos = upload(cos, [b, 1, n, d], &self.device, self.dtype);
        let sin = upload(sin, [b, 1, n, d], &self.device, self.dtype);
        let heads = x.dims()[1];
        let mut parts = Vec::new();
        for axis in 0..2 {
            let start = axis * half;
            parts.push(
                -x.clone()
                    .slice([0..b, 0..heads, 0..n, start + quarter..start + half]),
            );
            parts.push(
                x.clone()
                    .slice([0..b, 0..heads, 0..n, start..start + quarter]),
            );
        }
        x * cos + Tensor::cat(parts, 3) * sin
    }

    pub fn rotary(&self, x: Tensor<4>, positions: &[[usize; 2]]) -> Tensor<4> {
        self.rotary_cached(x, None, None, positions)
    }

    pub fn capture<const D: usize>(&self, x: Tensor<D>) -> crate::host::Packed {
        let _timer = self.span("feature_readback");
        let data = crate::host::Packed::capture(x, self.dtype);
        self.pending_bytes.set(0);
        self.pending_ops.set(0);
        data
    }

    pub fn span(&self, label: &'static str) -> Timer<'_> {
        Timer {
            ops: self,
            label,
            start: std::time::Instant::now(),
        }
    }

    pub fn reset_timings(&self) {
        self.timings.borrow_mut().clear();
    }

    pub fn timing_report(&self) -> serde_json::Value {
        let entries = self
            .timings
            .borrow()
            .iter()
            .map(|(name, (calls, seconds))| {
                (
                    (*name).to_owned(),
                    serde_json::json!({"calls":calls,"inclusive_wall_seconds":seconds}),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        serde_json::Value::Object(entries)
    }
}

pub struct Timer<'a> {
    ops: &'a Ops,
    label: &'static str,
    start: std::time::Instant,
}
impl Drop for Timer<'_> {
    fn drop(&mut self) {
        let mut timings = self.ops.timings.borrow_mut();
        let entry = timings.entry(self.label).or_insert((0, 0.));
        entry.0 += 1;
        entry.1 += self.start.elapsed().as_secs_f64();
    }
}
