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
