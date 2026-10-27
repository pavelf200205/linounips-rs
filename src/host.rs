//! Compact native-dtype transfers and host feature gathering.
use burn::tensor::{Device, FloatDType, Tensor, TensorCreationOptions, TensorData};

#[derive(Clone)]
pub enum Packed {
    F16(Vec<half::f16>),
    BF16(Vec<half::bf16>),
    F32(Vec<f32>),
}

impl Packed {
    pub fn new(v: Vec<f32>, dtype: FloatDType) -> Self {
        match dtype {
            FloatDType::F16 => Self::F16(v.into_iter().map(half::f16::from_f32).collect()),
            FloatDType::BF16 => Self::BF16(v.into_iter().map(half::bf16::from_f32).collect()),
            _ => Self::F32(v),
        }
    }

    pub fn empty(dtype: FloatDType, capacity: usize) -> Self {
        match dtype {
            FloatDType::F16 => Self::F16(Vec::with_capacity(capacity)),
            FloatDType::BF16 => Self::BF16(Vec::with_capacity(capacity)),
            _ => Self::F32(Vec::with_capacity(capacity)),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::F16(v) => v.len(),
            Self::BF16(v) => v.len(),
            Self::F32(v) => v.len(),
        }
    }

    /// Zero out masked-off pixels in a planar [3, plane_len] buffer without f32 round-trips.
    pub fn zero_masked_planar(&self, mask: &[bool]) -> Self {
        let plane = mask.len();
        match self {
            Self::F16(v) => {
                let mut out = v.clone();
                for c in 0..3 {
                    let row = &mut out[c * plane..(c + 1) * plane];
                    for (dst, &m) in row.iter_mut().zip(mask.iter()) {
                        if !m {
                            *dst = half::f16::ZERO;
                        }
                    }
                }
                Self::F16(out)
            }
            Self::BF16(v) => {
                let mut out = v.clone();
                for c in 0..3 {
                    let row = &mut out[c * plane..(c + 1) * plane];
                    for (dst, &m) in row.iter_mut().zip(mask.iter()) {
                        if !m {
                            *dst = half::bf16::ZERO;
                        }
                    }
                }
                Self::BF16(out)
            }
            Self::F32(v) => {
                let mut out = v.clone();
                for c in 0..3 {
                    let row = &mut out[c * plane..(c + 1) * plane];
                    for (dst, &m) in row.iter_mut().zip(mask.iter()) {
                        if !m {
                            *dst = 0.0;
                        }
                    }
                }
                Self::F32(out)
            }
        }
    }

    /// Convert planar [3, plane_len] RGB into pixel-major [plane_len, 3] in native dtype.
    pub fn planar_to_interleaved_rgb(&self, plane: usize) -> Self {
        match self {
            Self::F16(v) => {
                let mut out = vec![half::f16::ZERO; plane * 3];
                let (r, rest) = v.split_at(plane);
                let (g, b) = rest.split_at(plane);
                for i in 0..plane {
                    let dst = i * 3;
                    out[dst] = r[i];
                    out[dst + 1] = g[i];
                    out[dst + 2] = b[i];
                }
                Self::F16(out)
            }
            Self::BF16(v) => {
                let mut out = vec![half::bf16::ZERO; plane * 3];
                let (r, rest) = v.split_at(plane);
                let (g, b) = rest.split_at(plane);
                for i in 0..plane {
                    let dst = i * 3;
                    out[dst] = r[i];
                    out[dst + 1] = g[i];
                    out[dst + 2] = b[i];
                }
                Self::BF16(out)
            }
            Self::F32(v) => {
                let mut out = vec![0.0; plane * 3];
                let (r, rest) = v.split_at(plane);
                let (g, b) = rest.split_at(plane);
                for i in 0..plane {
                    let dst = i * 3;
                    out[dst] = r[i];
                    out[dst + 1] = g[i];
                    out[dst + 2] = b[i];
                }
                Self::F32(out)
            }
        }
    }

    pub fn capture<const D: usize>(x: Tensor<D>, dtype: FloatDType) -> Self {
        let data = x.cast(dtype).into_data();
        match dtype {
            FloatDType::F16 => Self::F16(data.try_into_vec::<half::f16>().expect("FP16 readback")),
            FloatDType::BF16 => {
                Self::BF16(data.try_into_vec::<half::bf16>().expect("BF16 readback"))
            }
            _ => Self::F32(data.try_into_vec::<f32>().expect("FP32 readback")),
        }
    }

    pub fn extend_range(&mut self, source: &Self, r: std::ops::Range<usize>) {
        match (self, source) {
            (Self::F16(a), Self::F16(b)) => a.extend_from_slice(&b[r]),
            (Self::BF16(a), Self::BF16(b)) => a.extend_from_slice(&b[r]),
            (Self::F32(a), Self::F32(b)) => a.extend_from_slice(&b[r]),
            _ => panic!("Host feature dtype mismatch"),
        }
    }

    pub fn append(&mut self, other: Self) {
        match (self, other) {
            (Self::F16(a), Self::F16(mut b)) => a.append(&mut b),
            (Self::BF16(a), Self::BF16(mut b)) => a.append(&mut b),
            (Self::F32(a), Self::F32(mut b)) => a.append(&mut b),
            _ => panic!("Host feature dtype mismatch"),
        }
    }

    pub fn upload_range<const D: usize>(
        &self,
        r: std::ops::Range<usize>,
        shape: [usize; D],
        device: &Device,
    ) -> Tensor<D> {
        match self {
            Self::F16(v) => Tensor::from_data(
                TensorData::new(v[r].to_vec(), shape),
                TensorCreationOptions::new(device.clone()).with_dtype(FloatDType::F16.into()),
            ),
            Self::BF16(v) => Tensor::from_data(
                TensorData::new(v[r].to_vec(), shape),
                TensorCreationOptions::new(device.clone()).with_dtype(FloatDType::BF16.into()),
            ),
            Self::F32(v) => Tensor::from_data(TensorData::new(v[r].to_vec(), shape), device),
        }
    }

    pub fn into_tensor<const D: usize>(self, shape: [usize; D], device: &Device) -> Tensor<D> {
        match self {
            Self::F16(v) => Tensor::from_data(
                TensorData::new(v, shape),
                TensorCreationOptions::new(device.clone()).with_dtype(FloatDType::F16.into()),
            ),
            Self::BF16(v) => Tensor::from_data(
                TensorData::new(v, shape),
                TensorCreationOptions::new(device.clone()).with_dtype(FloatDType::BF16.into()),
            ),
            Self::F32(v) => Tensor::from_data(TensorData::new(v, shape), device),
        }
    }
}

/// Gather both pixel-major RGB colors `[pixels, n, 3]` and features `[pixels, n, 256]`
/// directly in native dtype with a single outer enum dispatch.
pub fn gather_samples(
    rgb_pixel_major: &[Packed],
    features: &[Packed],
    pixels: &[usize],
    feature_rows: Option<&[usize]>,
    dtype: FloatDType,
) -> (Packed, Packed) {
    let n = rgb_pixel_major.len();
    let count = pixels.len();
    match dtype {
        FloatDType::F16 => {
            let rgb_slices: Vec<&[half::f16]> = rgb_pixel_major
                .iter()
                .map(|p| match p {
                    Packed::F16(v) => v.as_slice(),
                    _ => unreachable!(),
                })
                .collect();
            let feat_slices: Vec<&[half::f16]> = features
                .iter()
                .map(|p| match p {
                    Packed::F16(v) => v.as_slice(),
                    _ => unreachable!(),
                })
                .collect();
            let mut colors = Vec::with_capacity(count * n * 3);
            let mut feats = Vec::with_capacity(count * n * 256);
            for &pixel in pixels {
                let c_off = pixel * 3;
                let f_off = feature_rows.map_or(pixel, |rows| rows[pixel]) * 256;
                for light in 0..n {
                    colors.extend_from_slice(&rgb_slices[light][c_off..c_off + 3]);
                    feats.extend_from_slice(&feat_slices[light][f_off..f_off + 256]);
                }
            }
            (Packed::F16(colors), Packed::F16(feats))
        }
        FloatDType::BF16 => {
            let rgb_slices: Vec<&[half::bf16]> = rgb_pixel_major
                .iter()
                .map(|p| match p {
                    Packed::BF16(v) => v.as_slice(),
                    _ => unreachable!(),
                })
                .collect();
            let feat_slices: Vec<&[half::bf16]> = features
                .iter()
                .map(|p| match p {
                    Packed::BF16(v) => v.as_slice(),
                    _ => unreachable!(),
                })
                .collect();
            let mut colors = Vec::with_capacity(count * n * 3);
            let mut feats = Vec::with_capacity(count * n * 256);
            for &pixel in pixels {
                let c_off = pixel * 3;
                let f_off = feature_rows.map_or(pixel, |rows| rows[pixel]) * 256;
                for light in 0..n {
                    colors.extend_from_slice(&rgb_slices[light][c_off..c_off + 3]);
                    feats.extend_from_slice(&feat_slices[light][f_off..f_off + 256]);
                }
            }
            (Packed::BF16(colors), Packed::BF16(feats))
        }
        _ => {
            let rgb_slices: Vec<&[f32]> = rgb_pixel_major
                .iter()
                .map(|p| match p {
                    Packed::F32(v) => v.as_slice(),
                    _ => unreachable!(),
                })
                .collect();
            let feat_slices: Vec<&[f32]> = features
                .iter()
                .map(|p| match p {
                    Packed::F32(v) => v.as_slice(),
                    _ => unreachable!(),
                })
                .collect();
            let mut colors = Vec::with_capacity(count * n * 3);
            let mut feats = Vec::with_capacity(count * n * 256);
            for &pixel in pixels {
                let c_off = pixel * 3;
                let f_off = feature_rows.map_or(pixel, |rows| rows[pixel]) * 256;
                for light in 0..n {
                    colors.extend_from_slice(&rgb_slices[light][c_off..c_off + 3]);
                    feats.extend_from_slice(&feat_slices[light][f_off..f_off + 256]);
                }
            }
            (Packed::F32(colors), Packed::F32(feats))
        }
    }
}

pub fn read<const D: usize>(x: Tensor<D>) -> Vec<f32> {
    x.cast(FloatDType::F32)
        .into_data()
        .try_into_vec::<f32>()
        .expect("FP32 readback")
}

pub fn upload<const D: usize>(
    v: Vec<f32>,
    shape: [usize; D],
    device: &Device,
    dtype: FloatDType,
) -> Tensor<D> {
    Packed::new(v, dtype).into_tensor(shape, device)
}
