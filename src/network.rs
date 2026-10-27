//! Released LINO graph, with CPU-backed stage features and batched per-light fusion.
use crate::{host::Packed, ops::Ops};
use burn::tensor::{
    FloatDType, Tensor, module,
    ops::{ConvOptions, ConvTransposeOptions},
};

const A: &str = "image_encoder.backbone.aggregator";
const F: &str = "image_encoder.fusion";

pub struct Network {
    pub ops: Ops,
    pub pbr: bool,
    pub encoder_batch: usize,
    pub verbose: bool,
}

pub struct Predictions {
    pub tensor: Tensor<3>,
    pub maps: Vec<(String, usize)>,
}

impl Network {
    fn block(
        &self,
        x: Tensor<3>,
        p: &str,
        pos: Option<(&str, &[[usize; 2]])>,
        eps: f64,
    ) -> Tensor<3> {
        let [b, n, c] = x.dims();
        let z = self.ops.norm(x.clone(), &format!("{p}.norm1"), eps);
        let qkv = self
            .ops
            .linear(z, &format!("{p}.attn.qkv"))
            .reshape([b, n, 18, 64])
            .swap_dims(1, 2);
        let mut q = qkv.clone().slice([0..b, 0..6, 0..n, 0..64]);
        let mut k = qkv.clone().slice([0..b, 6..12, 0..n, 0..64]);
        let v = qkv.slice([0..b, 12..18, 0..n, 0..64]);
        if let Some((pos_key, pos_slice)) = pos {
            q = self.ops.norm(q, &format!("{p}.attn.q_norm"), 1e-5);
            k = self.ops.norm(k, &format!("{p}.attn.k_norm"), 1e-5);
            q = self
                .ops
                .rotary_cached(q, Some(pos_key), Some(32), pos_slice);
            k = self
                .ops
                .rotary_cached(k, Some(pos_key), Some(32), pos_slice);
        }
        let attn = self.ops.attention(q, k, v, 1.0 / 8.0);
        let ls1 = self
            .ops
            .w::<1>(&format!("{p}.ls1.gamma"))
            .reshape([1, 1, c]);
        let x = x + self.ops.linear(attn, &format!("{p}.attn.proj")) * ls1;
        let ls2 = self
            .ops
            .w::<1>(&format!("{p}.ls2.gamma"))
            .reshape([1, 1, c]);
        let step = ((self.ops.inflight_mib * 1024 * 1024 / (b.max(1) * c * 16)) / 512 * 512)
            .clamp(512, 4096);
        let step = if self.ops.mlp_chunk_tokens > 0 {
            (self.ops.mlp_chunk_tokens / 512 * 512).max(512)
        } else {
            step
        };
        if n <= step {
            let y = self.ops.norm(x.clone(), &format!("{p}.norm2"), eps);
            let y = self.ops.linear_gelu(y, &format!("{p}.mlp.fc1"));
            let y = self.ops.linear(y, &format!("{p}.mlp.fc2")) * ls2;
            return x + y;
        }
        // Pad `n` to a multiple of `step` so every MLP chunk uses the exact same aligned GEMM shape.
        let pad_n = n.div_ceil(step) * step;
        let x_pad = if pad_n > n {
            let z = Tensor::<3>::zeros([b, pad_n - n, c], self.ops.options());
            Tensor::cat(vec![x, z], 1)
        } else {
            x
        };
        let mut chunks = Vec::with_capacity(pad_n / step);
        let mut output = self
            .ops
            .memory_optimizations
            .then(|| Tensor::<3>::empty([b, pad_n, c], self.ops.options()));
        for start in (0..pad_n).step_by(step) {
            let end = start + step;
            let orig = x_pad.clone().slice([0..b, start..end, 0..c]);
            let y = self.ops.norm(orig.clone(), &format!("{p}.norm2"), eps);
            let y = self.ops.linear_gelu(y, &format!("{p}.mlp.fc1"));
            let y = self.ops.linear(y, &format!("{p}.mlp.fc2")) * ls2.clone();
            let chunk = orig + y;
            if let Some(out) = output.take() {
                output = Some(out.slice_assign([0..b, start..end, 0..c], chunk));
            } else {
                chunks.push(chunk);
            }
        }
        let out = output.unwrap_or_else(|| Tensor::cat(chunks, 1));
        if pad_n > n {
            out.slice([0..b, 0..n, 0..c])
        } else {
            out
        }
    }

    fn blocks_batched(
        &self,
        x: Tensor<3>,
        p: &str,
        key_prefix: &str,
        pos: &[[usize; 2]],
        batch: usize,
    ) -> Tensor<3> {
        let [b, n, c] = x.dims();
        if b <= batch {
            return self.block(
                x,
                p,
                Some((&format!("{key_prefix}/0/{b}"), &pos[0..b * n])),
                1e-5,
            );
        }
        let mut chunks = Vec::with_capacity(b.div_ceil(batch));
        let mut output = self
            .ops
            .memory_optimizations
            .then(|| Tensor::<3>::empty([b, n, c], self.ops.options()));
        for start in (0..b).step_by(batch) {
            let end = (start + batch).min(b);
            let count = end - start;
            let (slice_x, pos_vec, key) = if count < batch {
                // Pad tail batch to `batch` so CubeCL reuses the already-tuned batch kernel.
                let pad = batch - count;
                let z = Tensor::<3>::zeros([pad, n, c], self.ops.options());
                let mut padded_pos = Vec::with_capacity(batch * n);
                padded_pos.extend_from_slice(&pos[start * n..end * n]);
                padded_pos.resize(batch * n, [0, 0]);
                (
                    Tensor::cat(vec![x.clone().slice([start..end, 0..n, 0..c]), z], 0),
                    padded_pos,
                    format!("{key_prefix}/tail/{start}/{batch}"),
                )
            } else {
                (
                    x.clone().slice([start..end, 0..n, 0..c]),
                    pos[start * n..end * n].to_vec(),
                    format!("{key_prefix}/{start}/{end}"),
                )
            };
            let y = self.block(slice_x, p, Some((&key, &pos_vec)), 1e-5);
            let chunk = if count < batch {
                y.slice([0..count, 0..n, 0..c])
            } else {
                y
            };
            if let Some(out) = output.take() {
                output = Some(out.slice_assign([start..end, 0..n, 0..c], chunk));
            } else {
                chunks.push(chunk);
            }
            self.ops.boundary();
        }
        output.unwrap_or_else(|| Tensor::cat(chunks, 0))
    }

    fn dino(&self, input: Tensor<4>) -> Tensor<3> {
        let p = format!("{A}.patch_embed");
        let b = input.dims()[0];
        let patches = self
            .ops
            .conv(input, &format!("{p}.patch_embed.proj"), 8, 0)
            .reshape([b, 384, 1024])
            .swap_dims(1, 2)
            + self.ops.w::<3>(&format!("{p}.patch_pos"));
        let prefix = self
            .ops
            .w::<3>(&format!("{p}.prefix_tokens"))
            .expand([b, 5, 384]);
        let mut x = Tensor::cat(vec![prefix, patches], 1);
        for i in 0..12 {
            x = self.block(x, &format!("{p}.blocks.{i}"), None, 1e-6);
        }
        let x = self.ops.norm(x, &format!("{p}.norm"), 1e-6);
        Tensor::cat(
            vec![
                x.clone().slice([0..b, 1..4, 0..384]),
                x.slice([0..b, 1..1029, 0..384]),
            ],
            1,
        )
    }

    // Return four CPU-backed [stream, 384, 64, 64] pre-normalized spatial feature stages.
    fn backbone(&self, streams: &[Packed], branch: &str) -> Vec<Packed> {
        let _timer = self.ops.span("backbone");
        let _branch_timer = self.ops.span(&format!("backbone/{branch}"));
        let n = streams.len();
        crate::progress::message(format!(
            "  [{branch}] running DINO backbone on {n} streams..."
        ));
        let needs_resize = streams.first().is_some_and(|s| s.len() == 3 * 512 * 512);
        let batch_size = self.encoder_batch.max(1);
        let mut dino_chunks = Vec::with_capacity(n.div_ceil(batch_size));
        let mut dino_output = (self.ops.memory_optimizations && n > batch_size)
            .then(|| Tensor::<3>::empty([n, 1031, 384], self.ops.options()));
        for start in (0..n).step_by(batch_size) {
            let end = (start + batch_size).min(n);
            let count = end - start;
            let batch_input = if needs_resize {
                let mut data = Packed::empty(self.ops.dtype, batch_size * 3 * 512 * 512);
                for image in &streams[start..end] {
                    data.extend_range(image, 0..image.len());
                }
                for _ in count..batch_size {
                    data.extend_range(&streams[end - 1], 0..streams[end - 1].len());
                }
                let full = data.into_tensor([batch_size, 3, 512, 512], &self.ops.device);
                self.ops.resize(full, 256, true)
            } else {
                let mut data = Packed::empty(self.ops.dtype, batch_size * 3 * 256 * 256);
                for image in &streams[start..end] {
                    data.extend_range(image, 0..image.len());
                }
                for _ in count..batch_size {
                    data.extend_range(&streams[end - 1], 0..streams[end - 1].len());
                }
                data.into_tensor([batch_size, 3, 256, 256], &self.ops.device)
            };
            let y = self.dino(batch_input);
            let chunk = if count < batch_size {
                y.slice([0..count, 0..1031, 0..384])
            } else {
                y
            };
            if let Some(out) = dino_output.take() {
                dino_output = Some(out.slice_assign([start..end, 0..1031, 0..384], chunk));
            } else {
                dino_chunks.push(chunk);
            }
            self.ops.boundary();
        }
        let mut x = if let Some(out) = dino_output {
            out
        } else if dino_chunks.len() == 1 {
            dino_chunks.pop().unwrap()
        } else {
            Tensor::cat(dino_chunks, 0)
        };
        let one: Vec<[usize; 2]> = (0..1031)
            .map(|i| {
                if i < 7 {
                    [0, 0]
                } else {
                    [(i - 7) / 32 + 1, (i - 7) % 32 + 1]
                }
            })
            .collect();
        let frame_pos = one.repeat(n);
        let light_pos: Vec<_> = one
            .iter()
            .flat_map(|p| std::iter::repeat_n(*p, n))
            .collect();
        let light_batch = if n <= 18 { 1031 } else { 516 };
        let capture_batch = (self.ops.inflight_mib / 4).clamp(1, 18);
        let mut stages = Vec::with_capacity(4);
        for i in 0..4 {
            if self.verbose {
                crate::progress::message(format!(
                    "  [{branch}] alternating-attention stage {}/4 ({n} streams)...",
                    i + 1
                ));
            }
            let frame = self.blocks_batched(
                x,
                &format!("{A}.frame_blocks.{i}"),
                &format!("frame/{n}"),
                &frame_pos,
                batch_size,
            );
            let y = frame.clone().swap_dims(0, 1);
            let y = self.blocks_batched(
                y,
                &format!("{A}.light_axis_1_blocks.{i}"),
                &format!("light/{n}"),
                &light_pos,
                light_batch,
            );
            let light1 = y.swap_dims(0, 1);
            let global = self
                .block(
                    light1.clone().reshape([1, n * 1031, 384]),
                    &format!("{A}.global_blocks.{i}"),
                    Some((&format!("global/{n}"), &frame_pos)),
                    1e-5,
                )
                .reshape([n, 1031, 384]);
            let y = global.clone().swap_dims(0, 1);
            let y = self.blocks_batched(
                y,
                &format!("{A}.light_axis_2_blocks.{i}"),
                &format!("light/{n}"),
                &light_pos,
                light_batch,
            );
            x = y.swap_dims(0, 1);
            // Pre-normalize with fusion.norm and reshape to [stream, 384, 64, 64] in batches
            // before capturing to CPU so fusion uploads ready-to-convolve spatial maps.
            let mut packed = Packed::empty(self.ops.dtype, n * 384 * 64 * 64);
            for f in (0..n).step_by(capture_batch) {
                let end = (f + capture_batch).min(n);
                let b_cat = end - f;
                let cat = Tensor::cat(
                    [&frame, &light1, &global, &x]
                        .iter()
                        .map(|z| (*z).clone().slice([f..end, 7..1031, 0..384]))
                        .collect(),
                    2,
                );
                let spatial = self
                    .ops
                    .norm(cat, &format!("{F}.norm"), 1e-5)
                    .reshape([b_cat, 32, 32, 384, 2, 2])
                    .permute([0, 3, 1, 4, 2, 5])
                    .reshape([b_cat, 384, 64, 64]);
                packed.append(self.ops.capture(spatial));
            }
            stages.push(packed);
        }
        stages
    }

    fn residual(&self, x: Tensor<4>, p: &str) -> Tensor<4> {
        let y = self
            .ops
            .conv_leaky(self.ops.leaky(x.clone()), &format!("{p}.conv1"), 1, 1);
        let y = self.ops.conv(y, &format!("{p}.conv2"), 1, 1);
        x + y
    }

    fn refine(
        &self,
        x: Tensor<4>,
        skip: Option<Tensor<4>>,
        level: usize,
        size: usize,
    ) -> Tensor<4> {
        let p = format!("{F}.scratch.refinenet{level}");
        let x = if let Some(skip) = skip {
            x + self.residual(skip, &format!("{p}.resConfUnit1"))
        } else {
            x
        };
        let x = self.residual(x, &format!("{p}.resConfUnit2"));
        // Commute 1x1 `out_conv` before bilinear `resize` (mathematically identical since
        // bilinear weights sum to 1.0) to execute `out_conv` on 4x fewer pixels in NCHW GEMM.
        let x = self.ops.conv1x1(x, &format!("{p}.out_conv"));
        self.ops.resize(x, size, true)
    }

    fn fusion_pair(&self, stages: &[Packed], s0: usize, s1: usize) -> Tensor<4> {
        self.fusion_streams(stages, &[s0, s1])
    }

    fn fusion_streams(&self, stages: &[Packed], streams: &[usize]) -> Tensor<4> {
        let _timer = self.ops.span("fusion");
        let batch = streams.len();
        let stride = 384 * 64 * 64;
        let mut features = Vec::with_capacity(4);
        for (layer, data) in stages.iter().enumerate() {
            let x = if streams.windows(2).all(|s| s[1] == s[0] + 1) {
                let off = streams[0] * stride;
                data.upload_range(
                    off..off + batch * stride,
                    [batch, 384, 64, 64],
                    &self.ops.device,
                )
            } else {
                let mut packed = Packed::empty(self.ops.dtype, batch * stride);
                for &stream in streams {
                    packed.extend_range(data, stream * stride..(stream + 1) * stride);
                }
                packed.into_tensor([batch, 384, 64, 64], &self.ops.device)
            };
            let x = self
                .ops
                .leaky(self.ops.conv1x1(x, &format!("{F}.projects.{layer}.0")));
            let x = self.ops.position(x);
            let x = if layer < 2 {
                let x = self
                    .ops
                    .leaky(self.ops.deconv(x, &format!("{F}.resize_layers.{layer}.0")));
                if layer == 0 {
                    self.ops
                        .leaky(self.ops.deconv(x, &format!("{F}.resize_layers.0.2")))
                } else {
                    x
                }
            } else if layer == 2 {
                self.ops
                    .leaky(self.ops.conv1x1(x, &format!("{F}.resize_layers.2.0")))
            } else {
                self.ops
                    .leaky(self.ops.conv2x2s2(x, &format!("{F}.resize_layers.3.0")))
            };
            features.push(self.ops.conv_leaky(
                x,
                &format!("{F}.scratch.layer{}_rn.0", layer + 1),
                1,
                1,
            ));
        }
        let l4 = features.pop().unwrap();
        let l3 = features.pop().unwrap();
        let l2 = features.pop().unwrap();
        let l1 = features.pop().unwrap();
        let x = self.refine(l4, None, 4, 64);
        let x = self.refine(x, Some(l3), 3, 128);
        let x = self.refine(x, Some(l2), 2, 256);
        let x = if self.ops.fused_resize_conv {
            let x = self.refine(x, Some(l1), 1, 256);
            self.ops
                .conv_upsampled(x, &format!("{F}.scratch.output_conv1"), 512, 2, 1)
        } else {
            let x = self.refine(x, Some(l1), 1, 512);
            self.ops.conv(x, &format!("{F}.scratch.output_conv1"), 2, 1)
        };
        self.ops.position(x)
    }

    fn inverse(&self, bands: Vec<Tensor<4>>) -> Tensor<4> {
        let weight = self
            .ops
            .constant_as("haar/inverse".into(), self.ops.dtype, || {
                (0..256)
                    .flat_map(|_| {
                        [
                            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5,
                            -0.5, -0.5, 0.5,
                        ]
                    })
                    .collect()
            });
        let x = Tensor::stack::<5>(bands, 2).reshape([1, 1024, 256, 256]);
        module::conv_transpose2d(
            x,
            weight.reshape([1024, 1, 2, 2]),
            None,
            ConvTransposeOptions::new([2; 2], [0; 2], [0; 2], [1; 2], 256),
        )
    }

    fn smooth(&self, x: Tensor<4>) -> Tensor<4> {
        let weight = self
            .ops
            .constant_as("gaussian/21".into(), self.ops.dtype, || {
                let mut g = Vec::new();
                for y in -10..=10 {
                    for x in -10..=10 {
                        g.push((-0.5 * ((y * y + x * x) as f32)).exp());
                    }
                }
                let sum: f32 = g.iter().sum();
                for v in &mut g {
                    *v /= sum;
                }
                g.repeat(256)
            })
            .reshape([256, 1, 21, 21]);
        module::conv2d(
            x,
            weight,
            None,
            ConvOptions::new([1; 2], [10; 2], [1; 2], 256),
        )
    }

    fn smooth_nhwc_fast(
        &self,
        b01: Tensor<4>,
        b23: Tensor<4>,
        canonical_small: Tensor<4>,
        foreground: Option<Tensor<1>>,
    ) -> (Tensor<4>, bool) {
        let _timer = self.ops.span("smooth");
        if !self.ops.reference_kernels {
            let weights_1d = self.ops.constant("gaussian/1d/21".into(), || {
                let mut g = Vec::with_capacity(21);
                for d in -10..=10 {
                    g.push((-0.5 * ((d * d) as f32)).exp());
                }
                let sum: f32 = g.iter().sum();
                for v in &mut g {
                    *v /= sum;
                }
                g
            });
            if let Some(combined) = crate::geometry::inverse_haar_add_pair(
                b01.clone(),
                b23.clone(),
                canonical_small.clone(),
            ) {
                let compact = foreground.is_some();
                if let Some(nhwc) = crate::geometry::smooth_nhwc(
                    combined,
                    weights_1d,
                    self.ops.memory_optimizations,
                    foreground,
                ) {
                    return (nhwc, compact);
                }
            }
        }
        let bands = vec![
            b01.clone().slice([0..1, 0..256, 0..256, 0..256]),
            b01.slice([1..2, 0..256, 0..256, 0..256]),
            b23.clone().slice([0..1, 0..256, 0..256, 0..256]),
            b23.slice([1..2, 0..256, 0..256, 0..256]),
        ];
        let canonical = self.ops.resize(canonical_small, 512, true);
        let x = self.inverse(bands);
        let x = self.smooth(x + canonical);
        (x.permute([0, 2, 3, 1]), false)
    }

    pub fn encode(&self, images: &[Packed], foreground: &[usize]) -> Vec<Packed> {
        let _timer = self.ops.span("encoder");
        let n_lights = images.len();
        let stages = self.backbone(images, "canonical");
        crate::progress::message(format!(
            "  [canonical] running multi-scale fusion ({n_lights} lights)..."
        ));
        // Keep canonical fusion outputs at 256x256 (32 MiB/light instead of 128 MiB/light) across
        // the CPU backing boundary, and upsample to 512x512 inside `haar_inverse_add_pair`.
        let c_stride = 256 * 256 * 256;
        let mut canonical_256 = Packed::empty(self.ops.dtype, n_lights * c_stride);
        for f in (0..n_lights).step_by(2) {
            let _light_timer = self.ops.span("canonical/fusion_and_readback");
            let s1 = (f + 1).min(n_lights - 1);
            let pair = self.fusion_pair(&stages, f, s1);
            if f + 1 < n_lights {
                canonical_256.append(self.ops.capture(pair));
            } else {
                canonical_256.append(self.ops.capture(pair.slice([0..1, 0..256, 0..256, 0..256])));
            }
        }
        drop(stages);
        let wavelet_timer = self.ops.span("wavelet/cpu_transform");
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(n_lights.max(1));
        let chunk_size = n_lights.div_ceil(workers);
        let mut wavelets = Vec::with_capacity(n_lights * 4);
        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for chunk in images.chunks(chunk_size) {
                handles.push(s.spawn(move || chunk.iter().flat_map(wavelet).collect::<Vec<_>>()));
            }
            for h in handles {
                wavelets.extend(h.join().expect("wavelet worker panicked"));
            }
        });
        drop(wavelet_timer);
        let stages = self.backbone(&wavelets, "wavelet");
        drop(wavelets);
        crate::progress::message(format!(
            "  [wavelet] running fusion, inverse Haar & Gaussian smoothing ({n_lights} lights)..."
        ));
        let mut result = Vec::with_capacity(n_lights);
        // This changes only the storage representation, after all spatial arithmetic.
        // Indices are in raster order; decoder shuffles and context groups stay unchanged.
        let compact = self.ops.memory_optimizations && foreground.len() < 512 * 512;
        let indices = compact.then(|| {
            self.ops
                .indices(foreground.iter().map(|&i| i as i64).collect())
        });
        let gaussian_indices = (compact && !self.ops.reference_kernels).then(|| {
            crate::host::upload(
                foreground.iter().map(|&i| i as f32).collect(),
                [foreground.len()],
                &self.ops.device,
                FloatDType::F32,
            )
        });
        for f in 0..n_lights {
            let _light_timer = self.ops.span("wavelet/fusion_smooth_and_readback");
            let (b01, b23) = if self.ops.wide_wavelet_fusion {
                let all = self.fusion_streams(&stages, &[f * 4, f * 4 + 1, f * 4 + 2, f * 4 + 3]);
                (
                    all.clone().slice([0..2, 0..256, 0..256, 0..256]),
                    all.slice([2..4, 0..256, 0..256, 0..256]),
                )
            } else {
                (
                    self.fusion_pair(&stages, f * 4, f * 4 + 1),
                    self.fusion_pair(&stages, f * 4 + 2, f * 4 + 3),
                )
            };
            let can_off = f * c_stride;
            let canonical_small = canonical_256.upload_range(
                can_off..can_off + c_stride,
                [1, 256, 256, 256],
                &self.ops.device,
            );
            let (nhwc, already_compact) =
                self.smooth_nhwc_fast(b01, b23, canonical_small, gaussian_indices.clone());
            // Pixel-major host layout makes sample gathering contiguous across channels.
            if already_compact {
                result.push(self.ops.capture(nhwc));
            } else if let Some(indices) = &indices {
                let _gather_timer = self.ops.span("foreground/gather_and_readback");
                let dense = nhwc.reshape([512 * 512, 256]);
                let mut packed = Packed::empty(self.ops.dtype, foreground.len() * 256);
                let bytes = if self.ops.dtype == FloatDType::F32 {
                    4
                } else {
                    2
                };
                let chunk = 8 * 1024 * 1024 / (256 * bytes);
                for start in (0..foreground.len()).step_by(chunk) {
                    let end = (start + chunk).min(foreground.len());
                    let selected = dense.clone().select(0, indices.clone().slice([start..end]));
                    packed.append(self.ops.capture(selected));
                }
                result.push(packed);
            } else {
                result.push(self.ops.capture(nhwc));
            }
        }
        result
    }

    fn mab(&self, x: Tensor<3>, y: Tensor<3>, p: &str, normal: bool, valid_m: usize) -> Tensor<3> {
        let [b, n, _] = x.dims();
        let x = if self.ops.has(&format!("{p}.ln0.weight")) {
            self.ops.norm(x, &format!("{p}.ln0"), 1e-5)
        } else {
            x
        };
        let q = if p == "glc_aggregation.aggregation.dec.0.mab"
            && self.ops.has("glc_aggregation.aggregation.dec.0.S_q")
        {
            self.ops
                .w::<3>("glc_aggregation.aggregation.dec.0.S_q")
                .expand([b, n, 384])
        } else {
            self.ops.linear(x, &format!("{p}.fc_q"))
        };
        let c = q.dims()[2];
        let d = c / 8;
        let kv = self.ops.linear(y, &format!("{p}.fc_kv"));
        let o = self.ops.mab_attention(
            q,
            kv,
            8,
            1.0 / (if normal { c } else { d } as f64).sqrt(),
            valid_m,
        );
        let z = if self.ops.has(&format!("{p}.ln1.weight")) {
            self.ops.norm(o.clone(), &format!("{p}.ln1"), 1e-5)
        } else {
            o.clone()
        };
        let z = self.ops.linear_gelu(z, &format!("{p}.fc_o1"));
        o + self.ops.linear(z, &format!("{p}.fc_o2"))
    }

    pub fn decode(&self, rgb: Tensor<3>, features: Tensor<3>, pixels: usize) -> Predictions {
        let _timer = self.ops.span("decoder");
        let [uploaded_pixels, n_lights, _] = rgb.dims();
        // Pad pixel count up to a multiple of 64 (e.g. 2031/2032 -> 2048) so all pixel batches
        // share one 64-aligned GEMM shape, while `regressor.comm.enc` masks `valid_m = pixels`.
        let pad_pixels = pixels.div_ceil(64) * 64;
        let (rgb, features) = if pad_pixels > uploaded_pixels {
            let zr = Tensor::<3>::zeros(
                [pad_pixels - uploaded_pixels, n_lights, 3],
                self.ops.options(),
            );
            let zf = Tensor::<3>::zeros(
                [pad_pixels - uploaded_pixels, n_lights, 256],
                self.ops.options(),
            );
            (
                Tensor::cat(vec![rgb, zr], 0),
                Tensor::cat(vec![features, zf], 0),
            )
        } else {
            (rgb, features)
        };
        let rgb = self.ops.linear(
            self.ops.linear_leaky(rgb, "img_embedding.0"),
            "img_embedding.2",
        );
        let mut x = rgb.clone() + features;
        for i in 0..2 {
            x = self.mab(
                x.clone(),
                x,
                &format!("glc_upsample.comm.enc.{i}.mab"),
                true,
                n_lights,
            );
        }
        x = rgb + x;
        for i in 0..3 {
            x = self.mab(
                x.clone(),
                x,
                &format!("glc_aggregation.aggregation.enc.{i}.mab"),
                true,
                n_lights,
            );
        }
        let seed = self
            .ops
            .w::<3>("glc_aggregation.aggregation.dec.0.S")
            .expand([pad_pixels, 1, 384]);
        x = self
            .mab(
                seed,
                x,
                "glc_aggregation.aggregation.dec.0.mab",
                true,
                n_lights,
            )
            .reshape([1, pad_pixels, 384]);
        for i in 0..2 {
            x = self.mab(
                x.clone(),
                x,
                &format!("regressor.comm.enc.{i}.mab"),
                false,
                pixels,
            );
        }
        if pad_pixels > pixels {
            x = x.slice([0..1, 0..pixels, 0..384]);
        }
        let packed_hidden = if self.pbr && self.ops.packed_pbr_heads {
            Some(
                self.ops
                    .linear_relu(x.clone(), "regressor.prediction_packed"),
            )
        } else {
            None
        };
        if self.ops.fused_pbr_projection && !self.ops.reference_kernels {
            if let Some(hidden) = &packed_hidden {
                if let Some(tensor) = crate::pbr_projection::forward(
                    hidden.clone(),
                    self.ops.w("regressor.prediction_packed.output_weight"),
                    self.ops.w("regressor.prediction_packed.output_bias"),
                ) {
                    return Predictions {
                        tensor,
                        maps: vec![
                            ("normal".into(), 3),
                            ("diffuse".into(), 3),
                            ("roughness".into(), 1),
                            ("metallic".into(), 1),
                        ],
                    };
                }
            }
        }
        let mut out: Vec<(String, Tensor<3>)> = Vec::with_capacity(if self.pbr { 4 } else { 1 });
        for (head_index, (head, name)) in [
            ("normal", "normal"),
            ("base", "diffuse"),
            ("rough", "roughness"),
            ("metal", "metallic"),
        ]
        .into_iter()
        .enumerate()
        {
            if head != "normal" && !self.pbr {
                continue;
            }
            let p = format!("regressor.prediction_{head}");
            let y = if let Some(hidden) = &packed_hidden {
                hidden
                    .clone()
                    .slice([0..1, 0..pixels, head_index * 192..(head_index + 1) * 192])
            } else {
                self.ops
                    .linear_relu(x.clone(), &format!("{p}.regression.0"))
            };
            let mut y = self
                .ops
                .linear(y, &format!("{p}.out_layer"))
                .cast(FloatDType::F32);
            if head == "normal" {
                let norm = y.clone().square().sum_dim(2).sqrt().clamp_min(1e-12);
                y = y / norm;
            }
            out.push((name.into(), y));
        }
        let maps = out
            .iter()
            .map(|(name, t)| (name.clone(), t.dims()[2]))
            .collect();
        let mut tensors: Vec<_> = out.into_iter().map(|(_, t)| t).collect();
        let tensor = if tensors.len() == 1 {
            tensors.pop().unwrap()
        } else {
            Tensor::cat(tensors, 2)
        };
        Predictions { tensor, maps }
    }
}

// Haar signs follow PyWavelets db1: LL, high-Y/low-X, low-Y/high-X, HH.
fn wavelet(x: &Packed) -> Vec<Packed> {
    let len = 3 * 256 * 256;
    match x {
        Packed::F16(src) => {
            let mut b0 = vec![half::f16::ZERO; len];
            let mut b1 = vec![half::f16::ZERO; len];
            let mut b2 = vec![half::f16::ZERO; len];
            let mut b3 = vec![half::f16::ZERO; len];
            for c in 0..3 {
                for y in 0..256 {
                    let row = (c * 512 + 2 * y) * 512;
                    let dst = (c * 256 + y) * 256;
                    for xx in 0..256 {
                        let i = row + 2 * xx;
                        let a = src[i].to_f32();
                        let b = src[i + 1].to_f32();
                        let cc = src[i + 512].to_f32();
                        let d = src[i + 513].to_f32();
                        let j = dst + xx;
                        b0[j] = half::f16::from_f32((a + b + cc + d) * 0.5);
                        b1[j] = half::f16::from_f32((a + b - cc - d) * 0.5);
                        b2[j] = half::f16::from_f32((a - b + cc - d) * 0.5);
                        b3[j] = half::f16::from_f32((a - b - cc + d) * 0.5);
                    }
                }
            }
            vec![
                Packed::F16(b0),
                Packed::F16(b1),
                Packed::F16(b2),
                Packed::F16(b3),
            ]
        }
        Packed::BF16(src) => {
            let mut b0 = vec![half::bf16::ZERO; len];
            let mut b1 = vec![half::bf16::ZERO; len];
            let mut b2 = vec![half::bf16::ZERO; len];
            let mut b3 = vec![half::bf16::ZERO; len];
            for c in 0..3 {
                for y in 0..256 {
                    let row = (c * 512 + 2 * y) * 512;
                    let dst = (c * 256 + y) * 256;
                    for xx in 0..256 {
                        let i = row + 2 * xx;
                        let a = src[i].to_f32();
                        let b = src[i + 1].to_f32();
                        let cc = src[i + 512].to_f32();
                        let d = src[i + 513].to_f32();
                        let j = dst + xx;
                        b0[j] = half::bf16::from_f32((a + b + cc + d) * 0.5);
                        b1[j] = half::bf16::from_f32((a + b - cc - d) * 0.5);
                        b2[j] = half::bf16::from_f32((a - b + cc - d) * 0.5);
                        b3[j] = half::bf16::from_f32((a - b - cc + d) * 0.5);
                    }
                }
            }
            vec![
                Packed::BF16(b0),
                Packed::BF16(b1),
                Packed::BF16(b2),
                Packed::BF16(b3),
            ]
        }
        Packed::F32(src) => {
            let mut b0 = vec![0.0f32; len];
            let mut b1 = vec![0.0f32; len];
            let mut b2 = vec![0.0f32; len];
            let mut b3 = vec![0.0f32; len];
            for c in 0..3 {
                for y in 0..256 {
                    let row = (c * 512 + 2 * y) * 512;
                    let dst = (c * 256 + y) * 256;
                    for xx in 0..256 {
                        let i = row + 2 * xx;
                        let a = src[i];
                        let b = src[i + 1];
                        let cc = src[i + 512];
                        let d = src[i + 513];
                        let j = dst + xx;
                        b0[j] = (a + b + cc + d) * 0.5;
                        b1[j] = (a + b - cc - d) * 0.5;
                        b2[j] = (a - b + cc - d) * 0.5;
                        b3[j] = (a - b - cc + d) * 0.5;
                    }
                }
            }
            vec![
                Packed::F32(b0),
                Packed::F32(b1),
                Packed::F32(b2),
                Packed::F32(b3),
            ]
        }
    }
}
