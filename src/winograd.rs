//! Optional F(2x2,3x3) Winograd convolution. No spatial/context approximation.
//! Transform-domain rounding differs from direct convolution; validate outputs.
use burn::tensor::Tensor as BurnTensor;
use burn_backend::{DType, ExecutionError, Shape};
use burn_cubecl::{
    CubeBackend,
    fusion::{CubeFusionHandle, FusionCubeRuntime},
    kernel::matmul::{MatmulStrategy, matmul},
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
static LAYERS: AtomicU64 = AtomicU64::new(0);
static CHUNKS: AtomicU64 = AtomicU64::new(0);

#[cube(launch)]
fn transform_weights<F: Float>(x: &Tensor<F>, out: &mut Tensor<F>) {
    let i = usize::cast_from(ABSOLUTE_POS_X + ABSOLUTE_POS_Y * CUBE_COUNT_X * CUBE_DIM_X);
    let channels = x.shape(1);
    let outputs = x.shape(0);
    if i < channels * outputs {
        let oc = i / channels;
        let ic = i % channels;
        let mut g = Array::<f32>::new(9usize);
        let mut t = Array::<f32>::new(12usize);
        for j in 0..9usize {
            g[j] = f32::cast_from(
                x[oc * x.stride(0)
                    + ic * x.stride(1)
                    + (j / 3usize) * x.stride(2)
                    + (j % 3usize) * x.stride(3)],
            );
        }
        for j in 0..3usize {
            t[j] = g[j];
            t[3usize + j] = (g[j] + g[3usize + j] + g[6usize + j]) * 0.5_f32;
            t[6usize + j] = (g[j] - g[3usize + j] + g[6usize + j]) * 0.5_f32;
            t[9usize + j] = g[6usize + j];
        }
        for r in 0..4usize {
            let a = t[r * 3usize];
            let b = t[r * 3usize + 1usize];
            let c = t[r * 3usize + 2usize];
            out[((r * 4usize) * outputs + oc) * channels + ic] = F::cast_from(a);
            out[((r * 4usize + 1usize) * outputs + oc) * channels + ic] =
                F::cast_from((a + b + c) * 0.5_f32);
            out[((r * 4usize + 2usize) * outputs + oc) * channels + ic] =
                F::cast_from((a - b + c) * 0.5_f32);
            out[((r * 4usize + 3usize) * outputs + oc) * channels + ic] = F::cast_from(c);
        }
    }
}

#[cube(launch)]
fn transform_input<F: Float>(x: &Tensor<F>, out: &mut Tensor<F>, start: usize) {
    let i = usize::cast_from(ABSOLUTE_POS_X + ABSOLUTE_POS_Y * CUBE_COUNT_X * CUBE_DIM_X);
    let channels = x.shape(1);
    let rows = out.shape(2);
    if i < channels * rows {
        let ch = i / rows;
        let column = i % rows;
        let tile = start + column;
        let h = x.shape(2);
        let w = x.shape(3);
        let th = (h + 1usize) / 2usize;
        let tw = (w + 1usize) / 2usize;
        let batch = tile / (th * tw);
        let ty = (tile / tw) % th;
        let tx = tile % tw;
        let mut d = Array::<f32>::new(16usize);
        let mut t = Array::<f32>::new(16usize);
        for j in 0..16usize {
            d[j] = 0.0_f32;
            let yy = ty * 2usize + j / 4usize;
            let xx = tx * 2usize + j % 4usize;
            if batch < x.shape(0)
                && yy >= 1usize
                && yy < h + 1usize
                && xx >= 1usize
                && xx < w + 1usize
            {
                d[j] = f32::cast_from(
                    x[batch * x.stride(0)
                        + ch * x.stride(1)
                        + (yy - 1usize) * x.stride(2)
                        + (xx - 1usize) * x.stride(3)],
                );
            }
        }
        for j in 0..4usize {
            t[j] = d[j] - d[8usize + j];
            t[4usize + j] = d[4usize + j] + d[8usize + j];
            t[8usize + j] = d[8usize + j] - d[4usize + j];
            t[12usize + j] = d[4usize + j] - d[12usize + j];
        }
        for r in 0..4usize {
            let a = t[r * 4usize];
            let b = t[r * 4usize + 1usize];
            let c = t[r * 4usize + 2usize];
            let dd = t[r * 4usize + 3usize];
            out[((r * 4usize) * channels + ch) * rows + column] = F::cast_from(a - c);
            out[((r * 4usize + 1usize) * channels + ch) * rows + column] = F::cast_from(b + c);
            out[((r * 4usize + 2usize) * channels + ch) * rows + column] = F::cast_from(c - b);
            out[((r * 4usize + 3usize) * channels + ch) * rows + column] = F::cast_from(b - dd);
        }
    }
}

#[cube(launch)]
fn inverse_output<F: Float>(
    x: &Tensor<F>,
    bias: &Tensor<F>,
    out: &mut Tensor<F>,
    start: usize,
    count: usize,
    #[comptime] has_bias: bool,
    #[comptime] leaky: bool,
) {
    let i = usize::cast_from(ABSOLUTE_POS_X + ABSOLUTE_POS_Y * CUBE_COUNT_X * CUBE_DIM_X);
    let channels = out.shape(1);
    if i < channels * count {
        let ch = i / count;
        let column = i % count;
        let tile = start + column;
        let h = out.shape(2);
        let w = out.shape(3);
        let th = (h + 1usize) / 2usize;
        let tw = (w + 1usize) / 2usize;
        let batch = tile / (th * tw);
        let ty = (tile / tw) % th;
        let tx = tile % tw;
        let mut t = Array::<f32>::new(8usize);
        for j in 0..4usize {
            let a = f32::cast_from(x[j * x.stride(0) + ch * x.stride(1) + column]);
            let b = f32::cast_from(x[(4usize + j) * x.stride(0) + ch * x.stride(1) + column]);
            let c = f32::cast_from(x[(8usize + j) * x.stride(0) + ch * x.stride(1) + column]);
            let d = f32::cast_from(x[(12usize + j) * x.stride(0) + ch * x.stride(1) + column]);
            t[j] = a + b + c;
            t[4usize + j] = b - c - d;
        }
        for r in 0..2usize {
            for cc in 0..2usize {
                let yy = ty * 2usize + r;
                let xx = tx * 2usize + cc;
                if batch < out.shape(0) && yy < h && xx < w {
                    let mut value = f32::new(0.0_f32);
                    if cc == 0usize {
                        value = t[r * 4usize] + t[r * 4usize + 1usize] + t[r * 4usize + 2usize];
                    } else {
                        value = t[r * 4usize + 1usize]
                            - t[r * 4usize + 2usize]
                            - t[r * 4usize + 3usize];
                    }
                    let mut native = F::cast_from(value);
                    if has_bias {
                        native += bias[ch];
                    }
                    if leaky {
                        if native < F::new(0.0_f32) {
                            native *= F::new(0.01_f32);
                        }
                    }
                    out[((batch * channels + ch) * h + yy) * w + xx] = native;
                }
            }
        }
    }
}

fn grid(elements: usize, max_x: usize, max_y: usize) -> Result<CubeCount, ExecutionError> {
    let groups = elements.div_ceil(256);
    let gy = groups.div_ceil(max_x).max(1);
    let gx = groups.div_ceil(gy).max(1);
    if gy > max_y {
        return Err(ExecutionError::generic(
            "Winograd dispatch exceeds grid limits",
        ));
    }
    Ok(CubeCount::Static(gx as u32, gy as u32, 1))
}

#[derive(Debug)]
struct WeightOp {
    desc: CustomOpIr,
}
impl Operation<FusionCubeRuntime> for WeightOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([input], [output]) = self.desc.as_fixed();
        let x = handles.get_float_tensor::<CubeBackend>(input);
        let [o, c, _, _] = x.meta.shape().dims::<4>();
        let client = x.client.clone();
        let result = empty_device_contiguous_dtype(
            client.clone(),
            x.device.clone(),
            Shape::new([16, o, c]),
            x.dtype,
        );
        let hw = &client.properties().hardware;
        let count = grid(
            o * c,
            hw.max_cube_count.0 as usize,
            hw.max_cube_count.1 as usize,
        )?;
        let dim = CubeDim::new_1d(256);
        macro_rules! launch {
            ($ty:ty) => {
                transform_weights::launch::<$ty>(
                    &client,
                    count,
                    dim,
                    x.into_tensor_arg(),
                    result.clone().into_tensor_arg(),
                )
            };
        }
        match x.dtype {
            DType::F16 => launch!(half::f16),
            DType::BF16 => launch!(half::bf16),
            DType::F32 => launch!(f32),
            _ => return Err(ExecutionError::generic("Unsupported Winograd dtype")),
        };
        handles.register_float_tensor::<CubeBackend>(&output.id, result);
        Ok(())
    }
}

#[derive(Debug)]
struct ConvOp {
    desc: CustomOpIr,
    rows: usize,
    has_bias: bool,
    leaky: bool,
}
impl Operation<FusionCubeRuntime> for ConvOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let x = handles.get_float_tensor::<CubeBackend>(&self.desc.inputs[0]);
        let w = handles.get_float_tensor::<CubeBackend>(&self.desc.inputs[1]);
        let bias = if self.has_bias {
            handles.get_float_tensor::<CubeBackend>(&self.desc.inputs[2])
        } else {
            w.clone()
        };
        let [b, c, h, ww] = x.meta.shape().dims::<4>();
        let o = w.meta.shape().dims::<3>()[1];
        let client = x.client.clone();
        let output = empty_device_contiguous_dtype(
            client.clone(),
            x.device.clone(),
            Shape::new([b, o, h, ww]),
            x.dtype,
        );
        let packed = empty_device_contiguous_dtype(
            client.clone(),
            x.device.clone(),
            Shape::new([16, c, self.rows]),
            x.dtype,
        );
        let product = empty_device_contiguous_dtype(
            client.clone(),
            x.device.clone(),
            Shape::new([16, o, self.rows]),
            x.dtype,
        );
        let tiles = b * h.div_ceil(2) * ww.div_ceil(2);
        let dim = CubeDim::new_1d(256);
        for start in (0..tiles).step_by(self.rows) {
            let count = self.rows.min(tiles - start);
            let hw = &client.properties().hardware;
            let input_grid = grid(
                c * self.rows,
                hw.max_cube_count.0 as usize,
                hw.max_cube_count.1 as usize,
            )?;
            macro_rules! input_launch {
                ($ty:ty) => {
                    transform_input::launch::<$ty>(
                        &client,
                        input_grid,
                        dim,
                        x.clone().into_tensor_arg(),
                        packed.clone().into_tensor_arg(),
                        start,
                    )
                };
            }
            match x.dtype {
                DType::F16 => input_launch!(half::f16),
                DType::BF16 => input_launch!(half::bf16),
                DType::F32 => input_launch!(f32),
                _ => return Err(ExecutionError::generic("Unsupported Winograd dtype")),
            };
            matmul(
                w.clone(),
                packed.clone(),
                Some(product.clone()),
                MatmulStrategy::default(),
                x.dtype,
            )
            .map_err(|e| ExecutionError::generic(format!("Winograd GEMM: {e:?}")))?;
            let output_grid = grid(
                o * count,
                hw.max_cube_count.0 as usize,
                hw.max_cube_count.1 as usize,
            )?;
            macro_rules! output_launch {
                ($ty:ty) => {
                    inverse_output::launch::<$ty>(
                        &client,
                        output_grid,
                        dim,
                        product.clone().into_tensor_arg(),
                        bias.clone().into_tensor_arg(),
                        output.clone().into_tensor_arg(),
                        start,
                        count,
                        self.has_bias,
                        self.leaky,
                    )
                };
            }
            match x.dtype {
                DType::F16 => output_launch!(half::f16),
                DType::BF16 => output_launch!(half::bf16),
                DType::F32 => output_launch!(f32),
                _ => unreachable!(),
            };
            CHUNKS.fetch_add(1, Ordering::Relaxed);
        }
        LAYERS.fetch_add(1, Ordering::Relaxed);
        handles.register_float_tensor::<CubeBackend>(&self.desc.outputs[0].id, output);
        Ok(())
    }
}

pub fn weights(x: BurnTensor<4>) -> Option<BurnTensor<3>> {
    let [o, c, _, _] = x.dims();
    let x = x.inner().try_into_primitive::<Gpu>().ok()?;
    let client = x.client.clone();
    let out = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([16, o, c]),
        x.dtype,
    );
    let desc = CustomOpIr::with_scalars("lino_winograd_weights_v1", &[x.into_ir()], &[out], vec![]);
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            WeightOp { desc },
        )
        .outputs();
    Some(BurnTensor::from_primitive::<Gpu>(out))
}

pub fn forward(
    x: BurnTensor<4>,
    weights: BurnTensor<3>,
    bias: Option<BurnTensor<1>>,
    rows: usize,
    leaky: bool,
) -> Option<BurnTensor<4>> {
    let [b, _, h, w] = x.dims();
    let o = weights.dims()[1];
    let x = x.inner().try_into_primitive::<Gpu>().ok()?;
    let weights = weights.inner().try_into_primitive::<Gpu>().ok()?;
    let client = x.client.clone();
    let has_bias = bias.is_some();
    let mut inputs = vec![x.into_ir(), weights.into_ir()];
    if let Some(bias) = bias {
        inputs.push(bias.inner().try_into_primitive::<Gpu>().ok()?.into_ir());
    }
    let out = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([b, o, h, w]),
        inputs[0].dtype,
    );
    let scalars = [rows, has_bias as usize, leaky as usize]
        .map(|n| ScalarIr::UInt(n as u64))
        .to_vec();
    let desc = CustomOpIr::with_scalars("lino_winograd_conv_v1", &inputs, &[out], scalars);
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            ConvOp {
                desc,
                rows,
                has_bias,
                leaky,
            },
        )
        .outputs();
    Some(BurnTensor::from_primitive::<Gpu>(out))
}

pub fn stats() -> serde_json::Value {
    serde_json::json!({"layers":LAYERS.load(Ordering::Relaxed),"chunks":CHUNKS.load(Ordering::Relaxed)})
}
