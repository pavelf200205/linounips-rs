//! Portable element kernels: compact LUT embeddings, rotary embeddings and resize.
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
type Gpu = Fusion<CubeBackend>;
#[cube(launch)]
fn position<F: Float>(x: &Tensor<F>, lut: &Tensor<f32>, out: &mut Tensor<F>) {
    let i = usize::cast_from(ABSOLUTE_POS_X + ABSOLUTE_POS_Y * CUBE_COUNT_X * CUBE_DIM_X);
    if i < out.len() {
        let w = out.shape(3);
        let h = out.shape(2);
        let c = out.shape(1);
        let xx = i % w;
        let y = (i / w) % h;
        let ch = (i / (h * w)) % c;
        let b = i / (h * w * c);
        let mut coordinate = xx;
        if ch >= c / 2 {
            coordinate = y;
        }
        let mut length = h;
        if w > h {
            length = w;
        }
        let value = f32::cast_from(
            x[b * x.stride(0) + ch * x.stride(1) + y * x.stride(2) + xx * x.stride(3)],
        );
        out[i] = F::cast_from(value + lut[ch * length + coordinate]);
    }
}
#[cube(launch)]
fn rotary<F: Float>(x: &Tensor<F>, lut: &Tensor<f32>, pos: &Tensor<f32>, out: &mut Tensor<F>) {
    let i = usize::cast_from(ABSOLUTE_POS_X + ABSOLUTE_POS_Y * CUBE_COUNT_X * CUBE_DIM_X);
    if i < out.len() {
        let d = out.shape(3);
        let n = out.shape(2);
        let heads = out.shape(1);
        let ch = i % d;
        let q = (i / d) % n;
        let head = (i / (d * n)) % heads;
        let b = i / (d * n * heads);
        let half = d / 2;
        let quarter = d / 4;
        let axis = ch / half;
        let local = ch % half;
        let coordinate = usize::cast_from(pos[(b * n + q) * 2 + axis]);
        let lut_i = (coordinate * half + local) * 2;
        let mut paired = ch;
        let mut sign = f32::new(1.0_f32);
        if local < quarter {
            paired = ch + quarter;
            sign = -1.0_f32;
        } else {
            paired = ch - quarter;
        }
        let base = b * x.stride(0) + head * x.stride(1) + q * x.stride(2);
        let value = f32::cast_from(x[base + ch * x.stride(3)]);
        let rotated = f32::cast_from(x[base + paired * x.stride(3)]) * sign;
        // Match the reference's half-precision multiplies before the addition.
        out[i] = F::cast_from(value * lut[lut_i]) + F::cast_from(rotated * lut[lut_i + 1]);
    }
}
#[cube(launch)]
fn bilinear<F: Float>(x: &Tensor<F>, out: &mut Tensor<F>, #[comptime] align: bool) {
    let i = usize::cast_from(ABSOLUTE_POS_X + ABSOLUTE_POS_Y * CUBE_COUNT_X * CUBE_DIM_X);
    if i < out.len() {
        let w = out.shape(3);
        let h = out.shape(2);
        let c = out.shape(1);
        let ox = i % w;
        let oy = (i / w) % h;
        let ch = (i / (h * w)) % c;
        let b = i / (h * w * c);
        let iw = x.shape(3);
        let ih = x.shape(2);
        let mut fy = f32::new(0.0_f32);
        let mut fx = f32::new(0.0_f32);
        if align {
            if h > 1 {
                fy = f32::cast_from(oy) * f32::cast_from(ih - 1) / f32::cast_from(h - 1);
            }
            if w > 1 {
                fx = f32::cast_from(ox) * f32::cast_from(iw - 1) / f32::cast_from(w - 1);
            }
        } else {
            fy = f32::max(
                (f32::cast_from(oy) + 0.5_f32) * f32::cast_from(ih) / f32::cast_from(h) - 0.5_f32,
                0.0_f32,
            );
            fx = f32::max(
                (f32::cast_from(ox) + 0.5_f32) * f32::cast_from(iw) / f32::cast_from(w) - 0.5_f32,
                0.0_f32,
            );
        }
        let mut y0 = usize::cast_from(f32::floor(fy));
        if y0 >= ih {
            y0 = ih - 1;
        }
        let mut x0 = usize::cast_from(f32::floor(fx));
        if x0 >= iw {
            x0 = iw - 1;
        }
        let mut y1 = y0 + 1;
        if y1 >= ih {
            y1 = ih - 1;
        }
        let mut x1 = x0 + 1;
        if x1 >= iw {
            x1 = iw - 1;
        }
        let ay = fy - f32::cast_from(y0);
        let ax = fx - f32::cast_from(x0);
        let base = b * x.stride(0) + ch * x.stride(1);
        let a = f32::cast_from(x[base + y0 * x.stride(2) + x0 * x.stride(3)]);
        let bb = f32::cast_from(x[base + y0 * x.stride(2) + x1 * x.stride(3)]);
        let cc = f32::cast_from(x[base + y1 * x.stride(2) + x0 * x.stride(3)]);
        let dd = f32::cast_from(x[base + y1 * x.stride(2) + x1 * x.stride(3)]);
        let top = a + (bb - a) * ax;
        let bottom = cc + (dd - cc) * ax;
        out[i] = F::cast_from(top + (bottom - top) * ay);
    }
}
#[derive(Debug)]
struct GeometryOp {
    desc: CustomOpIr,
    kind: u8,
    shape: [usize; 4],
    align: bool,
}
impl Operation<FusionCubeRuntime> for GeometryOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let x = handles.get_float_tensor::<CubeBackend>(&self.desc.inputs[0]);
        let client = x.client.clone();
        let result = empty_device_contiguous_dtype(
            client.clone(),
            x.device.clone(),
            Shape::new(self.shape),
            x.dtype,
        );
        let rows = self.shape.iter().product::<usize>();
        let hw = &client.properties().hardware;
        let groups = rows.div_ceil(256);
        let gx = groups.min(hw.max_cube_count.0 as usize).max(1);
        let gy = groups.div_ceil(gx);
        if gy > hw.max_cube_count.1 as usize {
            return Err(ExecutionError::generic(
                "Geometry dispatch exceeds GPU limits",
            ));
        }
        let count = CubeCount::Static(gx as u32, gy as u32, 1);
        let dim = CubeDim::new_1d(256);
        macro_rules! launch {
            ($ty:ty) => {{
                match self.kind {
                    0 => {
                        let lut = handles.get_float_tensor::<CubeBackend>(&self.desc.inputs[1]);
                        position::launch::<$ty>(
                            &client,
                            count,
                            dim,
                            x.into_tensor_arg(),
                            lut.into_tensor_arg(),
                            result.clone().into_tensor_arg(),
                        );
                    }
                    1 => {
                        let lut = handles.get_float_tensor::<CubeBackend>(&self.desc.inputs[1]);
                        let pos = handles.get_float_tensor::<CubeBackend>(&self.desc.inputs[2]);
                        rotary::launch::<$ty>(
                            &client,
                            count,
                            dim,
                            x.into_tensor_arg(),
                            lut.into_tensor_arg(),
                            pos.into_tensor_arg(),
                            result.clone().into_tensor_arg(),
                        );
                    }
                    _ => {
                        bilinear::launch::<$ty>(
                            &client,
                            count,
                            dim,
                            x.into_tensor_arg(),
                            result.clone().into_tensor_arg(),
                            self.align,
                        );
                    }
                }
            }};
        }
        match x.dtype {
            DType::F16 => launch!(half::f16),
            DType::BF16 => launch!(half::bf16),
            DType::F32 => launch!(f32),
            _ => return Err(ExecutionError::generic("Unsupported geometry dtype")),
        };
        handles.register_float_tensor::<CubeBackend>(&self.desc.outputs[0].id, result);
        Ok(())
    }
}
fn register(
    x: BurnTensor<4>,
    extras: Vec<BurnTensor<1>>,
    kind: u8,
    shape: [usize; 4],
    align: bool,
) -> Option<BurnTensor<4>> {
    let x = x.inner().try_into_primitive::<Gpu>().ok()?;
    let client = x.client.clone();
    let mut inputs = vec![x.into_ir()];
    for t in extras {
        inputs.push(t.inner().try_into_primitive::<Gpu>().ok()?.into_ir());
    }
    let out = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new(shape),
        inputs[0].dtype,
    );
    // Shapes/align are semantic scalars: fusion must not deduplicate different resizes.
    let mut scalars = vec![ScalarIr::UInt(kind as u64), ScalarIr::UInt(align as u64)];
    scalars.extend(shape.map(|n| ScalarIr::UInt(n as u64)));
    let desc = CustomOpIr::with_scalars("lino_geometry_v1", &inputs, &[out], scalars);
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            GeometryOp {
                desc,
                kind,
                shape,
                align,
            },
        )
        .outputs();
    Some(BurnTensor::from_primitive::<Gpu>(out))
}
pub fn add_position(x: BurnTensor<4>, lut: BurnTensor<1>) -> Option<BurnTensor<4>> {
    let shape = x.dims();
    register(x, vec![lut], 0, shape, false)
}
pub fn rope(x: BurnTensor<4>, lut: BurnTensor<1>, pos: BurnTensor<1>) -> Option<BurnTensor<4>> {
    let shape = x.dims();
    register(x, vec![lut, pos], 1, shape, false)
}
pub fn resize(x: BurnTensor<4>, size: usize, align: bool) -> Option<BurnTensor<4>> {
    let [b, c, _, _] = x.dims();
    register(x, vec![], 2, [b, c, size, size], align)
}
