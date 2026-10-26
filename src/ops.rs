use crate::host::upload;
use anyhow::{Context, Result, ensure};
use burn::tensor::{
    Device, FloatDType, Int, Tensor, TensorCreationOptions, TensorData, activation, module,
    ops::{ConvOptions, ConvTransposeOptions},
};
use safetensors::{Dtype, SafeTensors};
use std::{collections::BTreeMap, path::Path};
pub struct Ops {
    pub device: Device,
    pub dtype: FloatDType,
    pub score_mib: usize,
    weights: BTreeMap<String, (Vec<usize>, Tensor<1>)>,
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
        let mut weights = BTreeMap::new();
        for (name, shape) in expected {
            let t = archive.tensor(&name).with_context(|| {
                format!("Missing required tensor {name}; wrong checkpoint mode or architecture")
            })?;
            ensure!(
                t.shape() == shape,
                "{name}: expected {shape:?}, got {:?}",
                t.shape()
            );
            let v: Vec<f32> = match t.dtype() {
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
            let n = v.len();
            weights.insert(name, (shape, upload(v, [n], &device, dtype)));
        }
        Ok(Self {
            device,
            dtype,
            score_mib,
            weights,
        })
    }
    pub fn options(&self) -> TensorCreationOptions {
        TensorCreationOptions::new(self.device.clone()).with_dtype(self.dtype.into())
    }
    pub fn w<const D: usize>(&self, key: &str) -> Tensor<D> {
        let (s, t) = &self.weights[key];
        let shape: [usize; D] = s.clone().try_into().expect("validated weight rank");
        t.clone().reshape(shape)
    }
    pub fn has(&self, key: &str) -> bool {
        self.weights.contains_key(key)
    }
    pub fn linear(&self, x: Tensor<3>, p: &str) -> Tensor<3> {
        let [b, n, c] = x.dims();
        let w: Tensor<2> = self.w(&format!("{p}.weight"));
        let o = w.dims()[0];
        let bias = self
            .has(&format!("{p}.bias"))
            .then(|| self.w::<1>(&format!("{p}.bias")));
        module::linear(x.reshape([b * n, c]), w.transpose(), bias).reshape([b, n, o])
    }
    pub fn norm<const D: usize>(&self, x: Tensor<D>, p: &str, eps: f64) -> Tensor<D> {
        module::layer_norm(
            x.cast(FloatDType::F32),
            self.w::<1>(&format!("{p}.weight")).cast(FloatDType::F32),
            Some(self.w::<1>(&format!("{p}.bias")).cast(FloatDType::F32)),
            eps,
        )
        .cast(self.dtype)
    }
    pub fn conv(&self, x: Tensor<4>, p: &str, stride: usize, pad: usize) -> Tensor<4> {
        let bias = self
            .has(&format!("{p}.bias"))
            .then(|| self.w::<1>(&format!("{p}.bias")));
        module::conv2d(
            x,
            self.w(&format!("{p}.weight")),
            bias,
            ConvOptions::new([stride; 2], [pad; 2], [1; 2], 1),
        )
    }
    pub fn deconv(&self, x: Tensor<4>, p: &str) -> Tensor<4> {
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
    // Online softmax over ALL keys. Query/key blocking never changes the receptive field.
    // FP32 statistics and accumulators; score_mib reserves headroom for score temporaries.
    pub fn attention(&self, q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, scale: f64) -> Tensor<4> {
        let [b, h, n, d] = q.dims();
        let m = k.dims()[2];
        let bh = b * h;
        let q = q.reshape([bh, n, d]);
        let k = k.reshape([bh, m, d]);
        let v = v.reshape([bh, m, d]);
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
                self.device.sync().expect("attention completion");
            }
        }
        output.reshape([b, h, n, d])
    }
    // FP32 bilinear interpolation in bounded channel slices, then cast immediately.
    pub fn resize(&self, x: Tensor<4>, size: usize, align: bool) -> Tensor<4> {
        let [b, c, h, w] = x.dims();
        if h == size && w == size {
            return x;
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
            self.device.sync().expect("resize completion");
        }
        out
    }
    pub fn leaky(&self, x: Tensor<4>) -> Tensor<4> {
        activation::leaky_relu(x, 0.01)
    }
}
