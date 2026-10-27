//! Experimental bounded NCHW im2col. Weights keep their existing OIHW storage.
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
fn pack<F: Float>(
    x: &Tensor<F>,
    out: &mut Tensor<F>,
    start: usize,
    #[comptime] height: usize,
    #[comptime] width: usize,
    #[comptime] kernel: usize,
    #[comptime] stride: usize,
    #[comptime] pad: usize,
    #[comptime] output_h: usize,
    #[comptime] output_w: usize,
    #[comptime] upsample: bool,
) {
    let i = usize::cast_from(ABSOLUTE_POS_X + ABSOLUTE_POS_Y * CUBE_COUNT_X * CUBE_DIM_X);
    if i < out.len() {
        let rows = out.shape(2);
        let reduction = i / rows;
        let row = start + i % rows;
        let ch = reduction / (kernel * kernel);
        let ky = (reduction / kernel) % kernel;
        let kx = reduction % kernel;
        let batch = row / (output_h * output_w);
        let y = ((row / output_w) % output_h) * stride + ky;
        let xx = (row % output_w) * stride + kx;
        let mut value = F::new(0.0_f32);
        if batch < x.shape(0) && y >= pad && y < height + pad && xx >= pad && xx < width + pad {
            let base = batch * x.stride(0) + ch * x.stride(1);
            if upsample {
                let ih = x.shape(2);
                let iw = x.shape(3);
                let fy = f32::cast_from(y - pad) * f32::cast_from(ih - 1usize)
                    / f32::cast_from(height - 1usize);
                let fx = f32::cast_from(xx - pad) * f32::cast_from(iw - 1usize)
                    / f32::cast_from(width - 1usize);
                let mut y0 = usize::cast_from(f32::floor(fy));
                let mut x0 = usize::cast_from(f32::floor(fx));
                if y0 >= ih {
                    y0 = ih - 1usize;
                }
                if x0 >= iw {
                    x0 = iw - 1usize;
                }
                let mut y1 = y0 + 1usize;
                let mut x1 = x0 + 1usize;
                if y1 >= ih {
                    y1 = ih - 1usize;
                }
                if x1 >= iw {
                    x1 = iw - 1usize;
                }
                let ay = fy - f32::cast_from(y0);
                let ax = fx - f32::cast_from(x0);
                let a = f32::cast_from(x[base + y0 * x.stride(2) + x0 * x.stride(3)]);
                let b = f32::cast_from(x[base + y0 * x.stride(2) + x1 * x.stride(3)]);
                let c = f32::cast_from(x[base + y1 * x.stride(2) + x0 * x.stride(3)]);
                let d = f32::cast_from(x[base + y1 * x.stride(2) + x1 * x.stride(3)]);
                let top = a + (b - a) * ax;
                let bottom = c + (d - c) * ax;
                // Retain the original resized activation's native-dtype rounding.
                value = F::cast_from(top + (bottom - top) * ay);
            } else {
                value = x[base + (y - pad) * x.stride(2) + (xx - pad) * x.stride(3)];
            }
        }
        out[i] = value;
    }
}

#[derive(Debug)]
struct PackOp {
    desc: CustomOpIr,
    height: usize,
    width: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
    start: usize,
    rows: usize,
    upsample: bool,
}
impl Operation<FusionCubeRuntime> for PackOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([input], [output]) = self.desc.as_fixed();
        let x = handles.get_float_tensor::<CubeBackend>(input);
        let channels = x.meta.shape().dims::<4>()[1];
        let client = x.client.clone();
        let hw = &client.properties().hardware;
        let shape = Shape::new([1, channels * self.kernel * self.kernel, self.rows]);
        let groups = (channels * self.kernel * self.kernel * self.rows).div_ceil(256);
        let gy = groups.div_ceil(hw.max_cube_count.0 as usize).max(1);
        let gx = groups.div_ceil(gy).max(1);
        if gy > hw.max_cube_count.1 as usize {
            return Err(ExecutionError::generic(
                "Convolution packing exceeds device grid limits",
            ));
        }
        let result =
            empty_device_contiguous_dtype(client.clone(), x.device.clone(), shape, x.dtype);
        let oh = (self.height + 2 * self.pad - self.kernel) / self.stride + 1;
        let ow = (self.width + 2 * self.pad - self.kernel) / self.stride + 1;
        let count = CubeCount::Static(gx as u32, gy as u32, 1);
        let dim = CubeDim::new_1d(256);
        macro_rules! launch {
            ($ty:ty) => {
                pack::launch::<$ty>(
                    &client,
                    count,
                    dim,
                    x.into_tensor_arg(),
                    result.clone().into_tensor_arg(),
                    self.start,
                    self.height,
                    self.width,
                    self.kernel,
                    self.stride,
                    self.pad,
                    oh,
                    ow,
                    self.upsample,
                )
            };
        }
        match x.dtype {
            DType::F16 => launch!(half::f16),
            DType::BF16 => launch!(half::bf16),
            DType::F32 => launch!(f32),
            _ => return Err(ExecutionError::generic("Unsupported convolution dtype")),
        };
        handles.register_float_tensor::<CubeBackend>(&output.id, result);
        Ok(())
    }
}

pub fn supported(x: &BurnTensor<4>) -> bool {
    let Ok(x) = x.clone().inner().try_into_primitive::<Gpu>() else {
        return false;
    };
    let client = x.client.device().client();
    let hw = &client.properties().hardware;
    hw.max_units_per_cube >= 256 && hw.max_cube_dim.0 >= 256
}

pub fn patches(
    x: BurnTensor<4>,
    height: usize,
    width: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
    start: usize,
    rows: usize,
    upsample: bool,
) -> BurnTensor<3> {
    let channels = x.dims()[1];
    let x = x
        .inner()
        .try_into_primitive::<Gpu>()
        .expect("checked convolution backend");
    let client = x.client.clone();
    let output = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([1, channels * kernel * kernel, rows]),
        x.dtype,
    );
    // All non-tensor semantics participate in the fusion key, including the row offset.
    let scalars = [
        height,
        width,
        kernel,
        stride,
        pad,
        start,
        rows,
        upsample as usize,
    ]
    .map(|x| ScalarIr::UInt(x as u64))
    .to_vec();
    let desc = CustomOpIr::with_scalars("lino_conv_pack_v1", &[x.into_ir()], &[output], scalars);
    let op = PackOp {
        desc: desc.clone(),
        height,
        width,
        kernel,
        stride,
        pad,
        start,
        rows,
        upsample,
    };
    let [out] = client
        .register(StreamId::current(), OperationIr::Custom(desc), op)
        .outputs();
    BurnTensor::from_primitive::<Gpu>(out)
}
