//! Released LINO graph, with CPU-backed stage features and per-light fusion.
use crate::{host::Packed, ops::Ops};
use burn::tensor::{
    FloatDType, Tensor, activation, module,
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
impl Network {
    fn block(&self, x: Tensor<3>, p: &str, pos: Option<&[[usize; 2]]>, eps: f64) -> Tensor<3> {
        let [b, n, c] = x.dims();
        let z = self.ops.norm(x.clone(), &format!("{p}.norm1"), eps);
        let qkv = self.ops.linear(z, &format!("{p}.attn.qkv"));
        let part = |a: usize| {
            qkv.clone()
                .slice([0..b, 0..n, a * c..(a + 1) * c])
                .reshape([b, n, 6, 64])
                .permute([0, 2, 1, 3])
        };
        let mut q = part(0);
        let mut k = part(1);
        let v = part(2);
        drop(qkv);
        if pos.is_some() {
            q = self.ops.norm(q, &format!("{p}.attn.q_norm"), 1e-5);
            k = self.ops.norm(k, &format!("{p}.attn.k_norm"), 1e-5);
        }
        if let Some(pos) = pos {
            q = self.ops.rotary(q, pos);
            k = self.ops.rotary(k, pos);
        }
        let attn = self
            .ops
            .attention(q, k, v, 1.0 / 8.0)
            .permute([0, 2, 1, 3])
            .reshape([b, n, c]);
        let x = x + self.ops.linear(attn, &format!("{p}.attn.proj"))
            * self
                .ops
                .w::<1>(&format!("{p}.ls1.gamma"))
                .reshape([1, 1, c]);
        let mut out = Tensor::<3>::empty([b, n, c], self.ops.options());
        let progress =
            (n >= 4096).then(|| crate::progress::phase("MLP", n.div_ceil(512), "chunks"));
        for start in (0..n).step_by(512) {
            let end = (start + 512).min(n);
            let orig = x.clone().slice([0..b, start..end, 0..c]);
            let y = self.ops.norm(orig.clone(), &format!("{p}.norm2"), eps);
            let y = activation::gelu(self.ops.linear(y, &format!("{p}.mlp.fc1")));
            let y = self.ops.linear(y, &format!("{p}.mlp.fc2"))
                * self
                    .ops
                    .w::<1>(&format!("{p}.ls2.gamma"))
                    .reshape([1, 1, c]);
            out = out.slice_assign([0..b, start..end, 0..c], orig + y);
            self.ops.schedule(b * (end - start) * c * 16);
            if let Some(progress) = &progress {
                progress.set(start / 512 + 1);
            }
        }
        out
    }
    fn blocks_batched(&self, x: Tensor<3>, p: &str, pos: &[[usize; 2]], batch: usize) -> Tensor<3> {
        let [b, n, c] = x.dims();
        let mut out = Tensor::<3>::empty([b, n, c], self.ops.options());
        let progress = crate::progress::phase("transformer batches", b.div_ceil(batch), "batches");
        for start in (0..b).step_by(batch) {
            let end = (start + batch).min(b);
            let y = self.block(
                x.clone().slice([start..end, 0..n, 0..c]),
                p,
                Some(&pos[start * n..end * n]),
                1e-5,
            );
            out = out.slice_assign([start..end, 0..n, 0..c], y);
            self.ops.boundary();
            progress.set(start / batch + 1);
        }
        out
    }
    fn dino(&self, input: Tensor<4>) -> Tensor<3> {
        let p = format!("{A}.patch_embed");
        let b = input.dims()[0];
        let patches = self
            .ops
            .conv(input, &format!("{p}.patch_embed.proj"), 8, 0)
            .reshape([b, 384, 1024])
            .swap_dims(1, 2);
        let cls = self
            .ops
            .w::<3>(&format!("{p}.cls_token"))
            .expand([b, 1, 384]);
        let tokens =
            Tensor::cat(vec![cls, patches], 1) + self.ops.w::<3>(&format!("{p}.pos_embed"));
        let reg = self
            .ops
            .w::<3>(&format!("{p}.register_tokens"))
            .expand([b, 4, 384]);
        let mut x = Tensor::cat(
            vec![
                tokens.clone().slice([0..b, 0..1, 0..384]),
                reg,
                tokens.slice([0..b, 1..1025, 0..384]),
            ],
            1,
        );
        let progress = crate::progress::phase("ViT", 12, "layers");
        for i in 0..12 {
            x = self.block(x, &format!("{p}.blocks.{i}"), None, 1e-6);
            progress.set(i + 1);
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
    // Return four CPU-backed [stream,1031,1536] feature stages; no GPU stage list.
    fn backbone(&self, streams: &[Packed], branch: &str) -> Vec<Packed> {
        let _timer = self.ops.span("backbone");
        let n = streams.len();
        let mut x = Tensor::<3>::empty([n, 1031, 384], self.ops.options());
        let dino_progress = crate::progress::phase(
            format!("{branch}/DINO"),
            n.div_ceil(self.encoder_batch),
            "batches",
        );
        for start in (0..n).step_by(self.encoder_batch) {
            let end = (start + self.encoder_batch).min(n);
            let mut data = Packed::empty(self.ops.dtype, (end - start) * 3 * 256 * 256);
            for image in &streams[start..end] {
                data.extend_range(image, 0..image.len());
            }
            let y = self.dino(data.into_tensor([end - start, 3, 256, 256], &self.ops.device));
            x = x.slice_assign([start..end, 0..1031, 0..384], y);
            self.ops.boundary();
            dino_progress.set(start / self.encoder_batch + 1);
        }
        drop(dino_progress);
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
        let mut stages = Vec::new();
        for i in 0..4 {
            if self.verbose {
                crate::progress::message(format!(
                    "  {branch}: alternating-attention stage {}/4, {n} streams",
                    i + 1
                ));
            }
            let stage = crate::progress::phase(format!("{branch}/frame {}/4", i + 1), 1, "stage");
            let frame = self.blocks_batched(
                x,
                &format!("{A}.frame_blocks.{i}"),
                &frame_pos,
                self.encoder_batch,
            );
            drop(stage);
            let stage =
                crate::progress::phase(format!("{branch}/light-axis-1 {}/4", i + 1), 1, "stage");
            let y = frame.clone().swap_dims(0, 1);
            let y = self.blocks_batched(y, &format!("{A}.light_axis_1_blocks.{i}"), &light_pos, 32);
            let light1 = y.swap_dims(0, 1);
            drop(stage);
            let stage = crate::progress::phase(format!("{branch}/global {}/4", i + 1), 1, "stage");
            let global = self
                .block(
                    light1.clone().reshape([1, n * 1031, 384]),
                    &format!("{A}.global_blocks.{i}"),
                    Some(&frame_pos),
                    1e-5,
                )
                .reshape([n, 1031, 384]);
            drop(stage);
            let stage =
                crate::progress::phase(format!("{branch}/light-axis-2 {}/4", i + 1), 1, "stage");
            let y = global.clone().swap_dims(0, 1);
            let y = self.blocks_batched(y, &format!("{A}.light_axis_2_blocks.{i}"), &light_pos, 32);
            x = y.swap_dims(0, 1);
            drop(stage);
            // Copy one stream at a time; never allocate a full four-way GPU concatenation.
            let mut packed = Packed::empty(self.ops.dtype, n * 1031 * 1536);
            let progress =
                crate::progress::phase(format!("{branch}/save features {}/4", i + 1), n, "streams");
            for f in 0..n {
                let cat = Tensor::cat(
                    [&frame, &light1, &global, &x]
                        .iter()
                        .map(|z| (*z).clone().slice([f..f + 1, 0..1031, 0..384]))
                        .collect(),
                    2,
                );
                packed.append(self.ops.capture(cat));
                progress.set(f + 1);
            }
            stages.push(packed);
        }
        stages
    }
    fn residual(&self, x: Tensor<4>, p: &str) -> Tensor<4> {
        let y = self
            .ops
            .conv(self.ops.leaky(x.clone()), &format!("{p}.conv1"), 1, 1);
        let y = self
            .ops
            .conv(self.ops.leaky(y), &format!("{p}.conv2"), 1, 1);
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
        let x = self.ops.resize(x, size, true);
        self.ops.conv(x, &format!("{p}.out_conv"), 1, 0)
    }
    fn fusion(&self, stages: &[Packed], stream: usize) -> Tensor<4> {
        let _timer = self.ops.span("fusion");
        let progress = crate::progress::phase("fusion", 5, "layers");
        let mut features = Vec::new();
        for (layer, data) in stages.iter().enumerate() {
            let off = stream * 1031 * 1536 + 7 * 1536;
            let x = data.upload_range(off..off + 1024 * 1536, [1, 1024, 1536], &self.ops.device);
            let x = self
                .ops
                .norm(x, &format!("{F}.norm"), 1e-5)
                .swap_dims(1, 2)
                .reshape([1, 384, 2, 2, 32, 32])
                .permute([0, 1, 4, 2, 5, 3])
                .reshape([1, 384, 64, 64]);
            let x = self
                .ops
                .leaky(self.ops.conv(x, &format!("{F}.projects.{layer}.0"), 1, 0));
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
            } else {
                self.ops.leaky(self.ops.conv(
                    x,
                    &format!("{F}.resize_layers.{layer}.0"),
                    if layer == 3 { 2 } else { 1 },
                    0,
                ))
            };
            features.push(self.ops.leaky(self.ops.conv(
                x,
                &format!("{F}.scratch.layer{}_rn.0", layer + 1),
                1,
                1,
            )));
            self.ops.schedule(x_bytes_for_fusion(layer));
            progress.set(layer + 1);
        }
        let l4 = features.pop().unwrap();
        let l3 = features.pop().unwrap();
        let l2 = features.pop().unwrap();
        let l1 = features.pop().unwrap();
        let x = self.refine(l4, None, 4, 64);
        let x = self.refine(x, Some(l3), 3, 128);
        let x = self.refine(x, Some(l2), 2, 256);
        let x = self.refine(x, Some(l1), 1, 512);
        let x = self.ops.conv(x, &format!("{F}.scratch.output_conv1"), 2, 1);
        let x = self.ops.position(x);
        progress.set(5);
        x
    }
    // Haar signs follow PyWavelets db1: LL, high-Y/low-X, low-Y/high-X, HH.
    fn wavelet(&self, x: &Packed) -> Vec<Packed> {
        let mut bands: Vec<Vec<f32>> = (0..4).map(|_| vec![0.; 3 * 256 * 256]).collect();
        for c in 0..3 {
            for y in 0..256 {
                for xx in 0..256 {
                    let i = (c * 512 + 2 * y) * 512 + 2 * xx;
                    let a = x.get(i);
                    let b = x.get(i + 1);
                    let cc = x.get(i + 512);
                    let d = x.get(i + 513);
                    let j = (c * 256 + y) * 256 + xx;
                    bands[0][j] = (a + b + cc + d) * 0.5;
                    bands[1][j] = (a + b - cc - d) * 0.5;
                    bands[2][j] = (a - b + cc - d) * 0.5;
                    bands[3][j] = (a - b - cc + d) * 0.5;
                }
            }
        }
        bands
            .into_iter()
            .map(|v| Packed::new(v, self.ops.dtype))
            .collect()
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
    pub fn encode(&self, images: &[Packed]) -> Vec<Packed> {
        let _timer = self.ops.span("encoder");
        let preparation = crate::progress::phase("canonical/resize inputs", 1, "stage");
        let small: Vec<_> = images
            .iter()
            .map(|im| {
                Packed::capture(
                    self.ops.resize(
                        im.upload_range(0..im.len(), [1, 3, 512, 512], &self.ops.device),
                        256,
                        true,
                    ),
                    self.ops.dtype,
                )
            })
            .collect();
        drop(preparation);
        let stages = self.backbone(&small, "canonical");
        drop(small);
        let mut result = Vec::with_capacity(images.len());
        let fusion_progress = crate::progress::phase("canonical/fusion", images.len(), "lights");
        for f in 0..images.len() {
            let x = self.fusion(&stages, f);
            let x = self.ops.resize(x, 512, true);
            result.push(self.ops.capture(x));
            fusion_progress.set(f + 1);
        }
        drop(fusion_progress);
        drop(stages);
        let preparation = crate::progress::phase("wavelet/prepare inputs", 1, "stage");
        let wavelets: Vec<_> = images.iter().flat_map(|im| self.wavelet(im)).collect();
        drop(preparation);
        let stages = self.backbone(&wavelets, "wavelet");
        drop(wavelets);
        let fusion_progress =
            crate::progress::phase("wavelet/fusion and smoothing", images.len(), "lights");
        for f in 0..images.len() {
            let bands = (0..4).map(|k| self.fusion(&stages, f * 4 + k)).collect();
            let x = self.inverse(bands);
            let canonical =
                result[f].upload_range(0..result[f].len(), [1, 256, 512, 512], &self.ops.device);
            let x = self.smooth(x + canonical);
            // Pixel-major host layout makes sample gathering contiguous across channels.
            result[f] = self.ops.capture(x.permute([0, 2, 3, 1]));
            fusion_progress.set(f + 1);
        }
        result
    }
    fn mab(&self, x: Tensor<3>, y: Tensor<3>, p: &str, normal: bool) -> Tensor<3> {
        let x = if self.ops.has(&format!("{p}.ln0.weight")) {
            self.ops.norm(x, &format!("{p}.ln0"), 1e-5)
        } else {
            x
        };
        let q = self.ops.linear(x, &format!("{p}.fc_q"));
        let k = self.ops.linear(y.clone(), &format!("{p}.fc_k"));
        let v = self.ops.linear(y, &format!("{p}.fc_v"));
        let [b, n, c] = q.dims();
        let m = k.dims()[1];
        let d = c / 8;
        let a = self
            .ops
            .attention(
                q.clone().reshape([b, n, 8, d]).permute([0, 2, 1, 3]),
                k.reshape([b, m, 8, d]).permute([0, 2, 1, 3]),
                v.reshape([b, m, 8, d]).permute([0, 2, 1, 3]),
                1.0 / (if normal { c } else { d } as f64).sqrt(),
            )
            .permute([0, 2, 1, 3])
            .reshape([b, n, c]);
        let o = q + a;
        let z = if self.ops.has(&format!("{p}.ln1.weight")) {
            self.ops.norm(o.clone(), &format!("{p}.ln1"), 1e-5)
        } else {
            o.clone()
        };
        let z = activation::gelu(self.ops.linear(z, &format!("{p}.fc_o1")));
        o + self.ops.linear(z, &format!("{p}.fc_o2"))
    }
    pub fn decode(&self, rgb: Tensor<3>, features: Tensor<3>) -> Vec<(String, Tensor<3>)> {
        let _timer = self.ops.span("decoder");
        let [pixels, _, _] = rgb.dims();
        let rgb = self.ops.linear(
            activation::leaky_relu(self.ops.linear(rgb, "img_embedding.0"), 0.01),
            "img_embedding.2",
        );
        let mut x = rgb.clone() + features;
        for i in 0..2 {
            x = self.mab(
                x.clone(),
                x,
                &format!("glc_upsample.comm.enc.{i}.mab"),
                true,
            );
        }
        x = rgb + x;
        for i in 0..3 {
            x = self.mab(
                x.clone(),
                x,
                &format!("glc_aggregation.aggregation.enc.{i}.mab"),
                true,
            );
        }
        let seed = self
            .ops
            .w::<3>("glc_aggregation.aggregation.dec.0.S")
            .expand([pixels, 1, 384]);
        x = self
            .mab(seed, x, "glc_aggregation.aggregation.dec.0.mab", true)
            .reshape([1, pixels, 384]);
        for i in 0..2 {
            x = self.mab(x.clone(), x, &format!("regressor.comm.enc.{i}.mab"), false);
        }
        let mut out = Vec::new();
        for (head, name) in [
            ("normal", "normal"),
            ("base", "diffuse"),
            ("rough", "roughness"),
            ("metal", "metallic"),
        ] {
            if head != "normal" && !self.pbr {
                continue;
            }
            let p = format!("regressor.prediction_{head}");
            let y = activation::relu(self.ops.linear(x.clone(), &format!("{p}.regression.0")));
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
        out
    }
}

fn x_bytes_for_fusion(layer: usize) -> usize {
    match layer {
        0 => 256 * 256 * 256 * 4,
        1 => 256 * 128 * 128 * 4,
        _ => 256 * 64 * 64 * 4,
    }
}
