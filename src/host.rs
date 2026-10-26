//! Compact CPU backing storage. No full-resolution multi-light tensor lives on GPU.
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
    pub fn get(&self, i: usize) -> f32 {
        match self {
            Self::F16(v) => v[i].to_f32(),
            Self::BF16(v) => v[i].to_f32(),
            Self::F32(v) => v[i],
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::F16(v) => v.len(),
            Self::BF16(v) => v.len(),
            Self::F32(v) => v.len(),
        }
    }
    pub fn range(&self, r: std::ops::Range<usize>) -> Vec<f32> {
        r.map(|i| self.get(i)).collect()
    }
    pub fn capture<const D: usize>(x: Tensor<D>, dtype: FloatDType) -> Self {
        Self::new(read(x), dtype)
    }
}
pub fn read<const D: usize>(x: Tensor<D>) -> Vec<f32> {
    x.cast(FloatDType::F32)
        .into_data()
        .try_to_vec::<f32>()
        .expect("FP32 readback")
}
pub fn upload<const D: usize>(
    v: Vec<f32>,
    shape: [usize; D],
    device: &Device,
    dtype: FloatDType,
) -> Tensor<D> {
    let options = TensorCreationOptions::new(device.clone()).with_dtype(dtype.into());
    match dtype {
        FloatDType::BF16 => Tensor::from_data(
            TensorData::new(
                v.into_iter().map(half::bf16::from_f32).collect::<Vec<_>>(),
                shape,
            ),
            options,
        ),
        FloatDType::F16 => Tensor::from_data(
            TensorData::new(
                v.into_iter().map(half::f16::from_f32).collect::<Vec<_>>(),
                shape,
            ),
            options,
        ),
        _ => Tensor::from_data(TensorData::new(v, shape), options),
    }
}
