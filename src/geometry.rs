//! Portable element kernels: compact LUT embeddings, rotary embeddings, resize,
//! fused inverse Haar + bilinear canonical add, and coalesced separable 1D Gaussian smoothing.
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
        let hw = h * w;
        let xx = i % w;
        let y = (i / w) % h;
        let ch = (i / hw) % c;
        let b = i / (hw * c);
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
        let dn = d * n;
        let ch = i % d;
        let q = (i / d) % n;
        let head = (i / dn) % heads;
        let b = i / (dn * heads);
        let half = d / 2;
        let quarter = d / 4;
        let axis = ch / half;
        let local = ch % half;
        let coordinate = usize::cast_from(pos[(b * n + q) * 2 + axis]);
        let lut_i = (coordinate * half + local) * 2;
        let mut paired = ch - quarter;
        let mut sign = f32::new(1.0_f32);
        if local < quarter {
            paired = ch + quarter;
            sign = -1.0_f32;
        }
        let base = b * x.stride(0) + head * x.stride(1) + q * x.stride(2);
        let s3 = x.stride(3);
        let value = f32::cast_from(x[base + ch * s3]);
        let rotated = f32::cast_from(x[base + paired * s3]) * sign;
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
        let hw = h * w;
        let ox = i % w;
        let oy = (i / w) % h;
        let ch = (i / hw) % c;
        let b = i / (hw * c);
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
        let s2 = x.stride(2);
        let s3 = x.stride(3);
        let row0 = base + y0 * s2;
        let row1 = base + y1 * s2;
        let a = f32::cast_from(x[row0 + x0 * s3]);
        let bb = f32::cast_from(x[row0 + x1 * s3]);
        let cc = f32::cast_from(x[row1 + x0 * s3]);
        let dd = f32::cast_from(x[row1 + x1 * s3]);
        let top = a + (bb - a) * ax;
        let bottom = cc + (dd - cc) * ax;
        out[i] = F::cast_from(top + (bottom - top) * ay);
    }
}

/// Fused inverse Haar transform from paired `[2, 256, 256, 256]` wavelet outputs (`b01`, `b23`)
/// plus on-the-fly bilinear upsampling of `canonical_256` (`[1, 256, 256, 256]` -> `512x512`).
#[cube(launch)]
fn haar_inverse_add_pair<F: Float>(
    b01: &Tensor<F>,
    b23: &Tensor<F>,
    canonical_256: &Tensor<F>,
    out: &mut Tensor<F>,
) {
    let i = usize::cast_from(ABSOLUTE_POS_X + ABSOLUTE_POS_Y * CUBE_COUNT_X * CUBE_DIM_X);
    if i < out.len() {
        let ox = i % 512usize;
        let oy = (i / 512usize) % 512usize;
        let ch = i / 262144usize;
        let xx = ox / 2usize;
        let yy = oy / 2usize;
        let dx = ox % 2usize;
        let dy = oy % 2usize;
        let c_base = ch * 65536usize;
        let idx = c_base + yy * 256usize + xx;
        let v0 = f32::cast_from(b01[idx]);
        let v1 = f32::cast_from(b01[16777216usize + idx]);
        let v2 = f32::cast_from(b23[idx]);
        let v3 = f32::cast_from(b23[16777216usize + idx]);
        let mut s1 = f32::new(1.0_f32);
        if dy != 0usize {
            s1 = -1.0_f32;
        }
        let mut s2 = f32::new(1.0_f32);
        if dx != 0usize {
            s2 = -1.0_f32;
        }
        let inv = F::cast_from((v0 + v1 * s1 + v2 * s2 + v3 * s1 * s2) * 0.5_f32);

        let fy = f32::cast_from(oy) * (255.0_f32 / 511.0_f32);
        let fx = f32::cast_from(ox) * (255.0_f32 / 511.0_f32);
        let mut y0 = usize::cast_from(f32::floor(fy));
        if y0 >= 256usize {
            y0 = 255usize;
        }
        let mut x0 = usize::cast_from(f32::floor(fx));
        if x0 >= 256usize {
            x0 = 255usize;
        }
        let mut y1 = y0 + 1usize;
        if y1 >= 256usize {
            y1 = 255usize;
        }
        let mut x1 = x0 + 1usize;
        if x1 >= 256usize {
            x1 = 255usize;
        }
        let ay = fy - f32::cast_from(y0);
        let ax = fx - f32::cast_from(x0);
        let r0 = c_base + y0 * 256usize;
        let r1 = c_base + y1 * 256usize;
        let ca = f32::cast_from(canonical_256[r0 + x0]);
        let cb = f32::cast_from(canonical_256[r0 + x1]);
        let cc = f32::cast_from(canonical_256[r1 + x0]);
        let cd = f32::cast_from(canonical_256[r1 + x1]);
        let top = ca + (cb - ca) * ax;
        let bottom = cc + (cd - cc) * ax;
        let can_val = F::cast_from(top + (bottom - top) * ay);

        out[i] = F::cast_from(f32::cast_from(inv) + f32::cast_from(can_val));
    }
}

#[cube(launch)]
fn gaussian_h<F: Float>(x: &Tensor<F>, weights: &Tensor<f32>, out: &mut Tensor<F>) {
    let i = usize::cast_from(ABSOLUTE_POS_X + ABSOLUTE_POS_Y * CUBE_COUNT_X * CUBE_DIM_X);
    if i < out.len() {
        let ox = i % 512usize;
        let row_base = i - ox;
        let mut sum = f32::new(0.0_f32);
        #[unroll]
        for k in 0..21usize {
            let shifted = ox + k;
            if shifted >= 10usize {
                let ix = shifted - 10usize;
                if ix < 512usize {
                    sum += f32::cast_from(x[row_base + ix]) * weights[k];
                }
            }
        }
        out[i] = F::cast_from(sum);
    }
}

// Reuse the horizontal halo in shared memory. Keep the original boundary tests,
// FP32 products and accumulation order (including skipped taps at image edges).
#[cube(launch)]
fn gaussian_h_tiled<F: Float>(x: &Tensor<F>, weights: &Tensor<f32>, out: &mut Tensor<F>) {
    let group = usize::cast_from(CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X);
    let unit = usize::cast_from(UNIT_POS);
    let mut tile = Shared::<[F]>::new_slice(276usize);
    if group * 256usize < out.len() {
        let start = group * 256usize;
        let x_base = start % 512usize;
        let row_base = start - x_base;
        let shifted = x_base + unit;
        tile[unit] = F::cast_from(0.0_f32);
        if shifted >= 10usize && shifted < 522usize {
            tile[unit] = x[row_base + shifted - 10usize];
        }
        if unit < 20usize {
            let shifted = x_base + unit + 256usize;
            tile[unit + 256usize] = F::cast_from(0.0_f32);
            if shifted >= 10usize && shifted < 522usize {
                tile[unit + 256usize] = x[row_base + shifted - 10usize];
            }
        }
        sync_cube();
        let ox = x_base + unit;
        let mut sum = f32::new(0.0_f32);
        #[unroll]
        for k in 0..21usize {
            let shifted = ox + k;
            if shifted >= 10usize {
                let ix = shifted - 10usize;
                if ix < 512usize {
                    sum += f32::cast_from(tile[unit + k]) * weights[k];
                }
            }
        }
        out[start + unit] = F::cast_from(sum);
    }
}

/// Coalesced vertical 21-tap Gaussian pass: threads in a warp index consecutive `ox` (`0..31`)
/// at the same `(ch, oy)` so all 21 reads hit a single contiguous L1 cache line per tap.
#[cube(launch)]
fn gaussian_v_nhwc<F: Float>(x: &Tensor<F>, weights: &Tensor<f32>, out: &mut Tensor<F>) {
    let i = usize::cast_from(ABSOLUTE_POS_X + ABSOLUTE_POS_Y * CUBE_COUNT_X * CUBE_DIM_X);
    if i < out.len() {
        let ox = i % 512usize;
        let oy = (i / 512usize) % 512usize;
        let ch = i / 262144usize;
        let col_base = ch * 262144usize + ox;
        let mut sum = f32::new(0.0_f32);
        #[unroll]
        for k in 0..21usize {
            let shifted = oy + k;
            if shifted >= 10usize {
                let iy = shifted - 10usize;
                if iy < 512usize {
                    sum += f32::cast_from(x[col_base + iy * 512usize]) * weights[k];
                }
            }
        }
        let dst = (oy * 512usize + ox) * 256usize + ch;
        out[dst] = F::cast_from(sum);
    }
}

// Same 21 products, accumulation order and final cast as gaussian_v_nhwc.
// A 32-channel x 32-pixel tile transposes only the already-rounded results.
// Neighboring lanes read adjacent X positions and then write adjacent channels.
#[cube(launch)]
fn gaussian_v_tiled<F: Float>(x: &Tensor<F>, weights: &Tensor<f32>, out: &mut Tensor<F>) {
    let group = usize::cast_from(CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X);
    let unit = usize::cast_from(UNIT_POS);
    let mut tile = Shared::<[F]>::new_slice(1056usize);
    if group < 65536usize {
        let x_base = (group % 16usize) * 32usize;
        let oy = (group / 16usize) % 512usize;
        let c_base = (group / 8192usize) * 32usize;
        for part in 0..4usize {
            let local = unit + part * 256usize;
            let xx = local % 32usize;
            let cc = local / 32usize;
            let col_base = (c_base + cc) * 262144usize + x_base + xx;
            let mut sum = f32::new(0.0_f32);
            #[unroll]
            for k in 0..21usize {
                let shifted = oy + k;
                if shifted >= 10usize {
                    let iy = shifted - 10usize;
                    if iy < 512usize {
                        sum += f32::cast_from(x[col_base + iy * 512usize]) * weights[k];
                    }
                }
            }
            tile[cc * 33usize + xx] = F::cast_from(sum);
        }
        sync_cube();
        for part in 0..4usize {
            let local = unit + part * 256usize;
            let cc = local % 32usize;
            let xx = local / 32usize;
            let dst = (oy * 512usize + x_base + xx) * 256usize + c_base + cc;
            out[dst] = tile[cc * 33usize + xx];
        }
    }
}

// Only the final vertical pass is sparse. Horizontal filtering and all upstream
// spatial dependencies remain dense. Indices are exact FP32 integers < 262144.
#[cube(launch)]
fn gaussian_v_foreground<F: Float>(
    x: &Tensor<F>,
    weights: &Tensor<f32>,
    pixels: &Tensor<f32>,
    out: &mut Tensor<F>,
) {
    let group = usize::cast_from(CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X);
    let unit = usize::cast_from(UNIT_POS);
    let pixel_tiles = (pixels.len() + 31usize) / 32usize;
    let mut tile = Shared::<[F]>::new_slice(1056usize);
    if group < pixel_tiles * 8usize {
        let pixel_base = (group % pixel_tiles) * 32usize;
        let c_base = (group / pixel_tiles) * 32usize;
        for part in 0..4usize {
            let local = unit + part * 256usize;
            let xx = local % 32usize;
            let cc = local / 32usize;
            if pixel_base + xx < pixels.len() {
                let pixel = usize::cast_from(pixels[pixel_base + xx]);
                let ox = pixel % 512usize;
                let oy = pixel / 512usize;
                let col_base = (c_base + cc) * 262144usize + ox;
                let mut sum = f32::new(0.0_f32);
                #[unroll]
                for k in 0..21usize {
                    let shifted = oy + k;
                    if shifted >= 10usize {
                        let iy = shifted - 10usize;
                        if iy < 512usize {
                            sum += f32::cast_from(x[col_base + iy * 512usize]) * weights[k];
                        }
                    }
                }
                tile[cc * 33usize + xx] = F::cast_from(sum);
            }
        }
        sync_cube();
        for part in 0..4usize {
            let local = unit + part * 256usize;
            let cc = local % 32usize;
            let xx = local / 32usize;
            if pixel_base + xx < pixels.len() {
                out[(pixel_base + xx) * 256usize + c_base + cc] = tile[cc * 33usize + xx];
            }
        }
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
        let [_, c, h, w] = x.meta.shape().dims::<4>();
        let reuse = self.kind == 9 && x.can_mut() && **x.meta.strides() == [c * h * w, h * w, w, 1];
        let result = if reuse {
            x.clone()
        } else {
            empty_device_contiguous_dtype(
                client.clone(),
                x.device.clone(),
                Shape::new(self.shape),
                x.dtype,
            )
        };
        let rows = self.shape.iter().product::<usize>();
        let hw = &client.properties().hardware;
        let groups = if self.kind == 7 {
            self.shape[2].div_ceil(32) * 8
        } else {
            rows.div_ceil(if self.kind == 6 { 1024 } else { 256 })
        };
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
                    0 | 9 => {
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
                    2 => {
                        bilinear::launch::<$ty>(
                            &client,
                            count,
                            dim,
                            x.into_tensor_arg(),
                            result.clone().into_tensor_arg(),
                            self.align,
                        );
                    }
                    3 => {
                        let b23 = handles.get_float_tensor::<CubeBackend>(&self.desc.inputs[1]);
                        let can = handles.get_float_tensor::<CubeBackend>(&self.desc.inputs[2]);
                        haar_inverse_add_pair::launch::<$ty>(
                            &client,
                            count,
                            dim,
                            x.into_tensor_arg(),
                            b23.into_tensor_arg(),
                            can.into_tensor_arg(),
                            result.clone().into_tensor_arg(),
                        );
                    }
                    4 => {
                        let weights = handles.get_float_tensor::<CubeBackend>(&self.desc.inputs[1]);
                        gaussian_h::launch::<$ty>(
                            &client,
                            count,
                            dim,
                            x.into_tensor_arg(),
                            weights.into_tensor_arg(),
                            result.clone().into_tensor_arg(),
                        );
                    }
                    8 => {
                        let weights = handles.get_float_tensor::<CubeBackend>(&self.desc.inputs[1]);
                        gaussian_h_tiled::launch::<$ty>(
                            &client,
                            count,
                            dim,
                            x.into_tensor_arg(),
                            weights.into_tensor_arg(),
                            result.clone().into_tensor_arg(),
                        );
                    }
                    7 => {
                        let weights = handles.get_float_tensor::<CubeBackend>(&self.desc.inputs[1]);
                        let pixels = handles.get_float_tensor::<CubeBackend>(&self.desc.inputs[2]);
                        gaussian_v_foreground::launch::<$ty>(
                            &client,
                            count,
                            dim,
                            x.into_tensor_arg(),
                            weights.into_tensor_arg(),
                            pixels.into_tensor_arg(),
                            result.clone().into_tensor_arg(),
                        );
                    }
                    6 => {
                        let weights = handles.get_float_tensor::<CubeBackend>(&self.desc.inputs[1]);
                        gaussian_v_tiled::launch::<$ty>(
                            &client,
                            count,
                            dim,
                            x.into_tensor_arg(),
                            weights.into_tensor_arg(),
                            result.clone().into_tensor_arg(),
                        );
                    }
                    _ => {
                        let weights = handles.get_float_tensor::<CubeBackend>(&self.desc.inputs[1]);
                        gaussian_v_nhwc::launch::<$ty>(
                            &client,
                            count,
                            dim,
                            x.into_tensor_arg(),
                            weights.into_tensor_arg(),
                            result.clone().into_tensor_arg(),
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
    extras_4d: Vec<BurnTensor<4>>,
    extras_1d: Vec<BurnTensor<1>>,
    kind: u8,
    shape: [usize; 4],
    align: bool,
) -> Option<BurnTensor<4>> {
    let x = x.inner().try_into_primitive::<Gpu>().ok()?;
    let client = x.client.clone();
    let mut inputs = vec![x.into_ir()];
    for t in extras_4d {
        inputs.push(t.inner().try_into_primitive::<Gpu>().ok()?.into_ir());
    }
    for t in extras_1d {
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
    let desc = CustomOpIr::with_scalars("lino_geometry_v2", &inputs, &[out], scalars);
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

pub fn add_position(x: BurnTensor<4>, lut: BurnTensor<1>, reuse: bool) -> Option<BurnTensor<4>> {
    let shape = x.dims();
    register(
        x,
        vec![],
        vec![lut],
        if reuse { 9 } else { 0 },
        shape,
        false,
    )
}

pub fn rope(x: BurnTensor<4>, lut: BurnTensor<1>, pos: BurnTensor<1>) -> Option<BurnTensor<4>> {
    let shape = x.dims();
    register(x, vec![], vec![lut, pos], 1, shape, false)
}

pub fn resize(x: BurnTensor<4>, size: usize, align: bool) -> Option<BurnTensor<4>> {
    let [b, c, _, _] = x.dims();
    register(x, vec![], vec![], 2, [b, c, size, size], align)
}

pub fn inverse_haar_add_pair(
    b01: BurnTensor<4>,
    b23: BurnTensor<4>,
    canonical_256: BurnTensor<4>,
) -> Option<BurnTensor<4>> {
    register(
        b01,
        vec![b23, canonical_256],
        vec![],
        3,
        [1, 256, 512, 512],
        false,
    )
}

pub fn smooth_nhwc(
    x: BurnTensor<4>,
    weights_1d: BurnTensor<1>,
    tiled: bool,
    foreground: Option<BurnTensor<1>>,
) -> Option<BurnTensor<4>> {
    let h = register(
        x,
        vec![],
        vec![weights_1d.clone()],
        if tiled { 8 } else { 4 },
        [1, 256, 512, 512],
        false,
    )?;
    if let Some(pixels) = foreground {
        let count = pixels.dims()[0];
        return register(
            h,
            vec![],
            vec![weights_1d, pixels],
            7,
            [1, 1, count, 256],
            false,
        );
    }
    register(
        h,
        vec![],
        vec![weights_1d],
        if tiled { 6 } else { 5 },
        [1, 512, 512, 256],
        false,
    )
}
