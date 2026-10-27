//! One custom operation per projection, with an optional in-place native-dtype epilogue.
use burn::tensor::Tensor as BurnTensor;
use burn_backend::{DType, ExecutionError, Shape};
use burn_cubecl::{
    CubeBackend,
    fusion::{CubeFusionHandle, FusionCubeRuntime},
    kernel::matmul::{MatmulStrategy, matmul},
    ops::{numeric::empty_device_contiguous_dtype, reshape},
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
static CALLS: AtomicU64 = AtomicU64::new(0);
static EPILOGUES: AtomicU64 = AtomicU64::new(0);

#[cube(launch)]
fn epilogue<F: Float>(
    x: &Tensor<F>,
    bias: &Tensor<F>,
    out: &mut Tensor<F>,
    #[comptime] has_bias: bool,
    #[comptime] activation: usize,
) {
    let i = usize::cast_from(ABSOLUTE_POS_X + ABSOLUTE_POS_Y * CUBE_COUNT_X * CUBE_DIM_X);
    if i < out.len() {
        let mut value = x[i];
        if has_bias {
            value += bias[i % out.shape(2)];
        }
        if activation == 1usize {
            // Follow Burn's erf GELU sequence, retaining native rounding at each step.
            let scaled = value / F::new(1.4142135623730951_f32);
            let erf = F::cast_from(f32::erf(f32::cast_from(scaled)));
            let shifted = erf + F::new(1.0_f32);
            let product = value * shifted;
            value = product / F::new(2.0_f32);
        } else if activation == 2usize {
            if value < F::new(0.0_f32) {
                value = F::new(0.0_f32);
            }
        } else if activation == 3usize {
            if value < F::new(0.0_f32) {
                value *= F::new(0.01_f32);
            }
        }
        out[i] = value;
    }
}

#[derive(Debug)]
struct LinearOp {
    desc: CustomOpIr,
    has_bias: bool,
    activation: usize,
}
impl Operation<FusionCubeRuntime> for LinearOp {
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
        let [b, n, c] = x.meta.shape().dims::<3>();
        let o = w.meta.shape().dims::<2>()[1];
        let client = x.client.clone();
        let dtype = x.dtype;
        let destination = empty_device_contiguous_dtype(
            client.clone(),
            x.device.clone(),
            Shape::new([b * n, o]),
            dtype,
        );
        let product = matmul(
            reshape(x, Shape::new([b * n, c])),
            w,
            Some(destination),
            MatmulStrategy::default(),
            dtype,
        )
        .map_err(|e| ExecutionError::generic(format!("Native linear GEMM: {e:?}")))?;
        let result = reshape(product.clone(), Shape::new([b, n, o]));
        if self.has_bias || self.activation != 0 {
            let hw = &client.properties().hardware;
            let groups = (b * n * o).div_ceil(256);
            let gy = groups.div_ceil(hw.max_cube_count.0 as usize).max(1);
            let gx = groups.div_ceil(gy).max(1);
            if gy > hw.max_cube_count.1 as usize {
                return Err(ExecutionError::generic(
                    "Linear epilogue exceeds grid limits",
                ));
            }
            let count = CubeCount::Static(gx as u32, gy as u32, 1);
            let dim = CubeDim::new_1d(256);
            // Product is freshly allocated and not exposed outside this operation.
            // Each lane reads/writes its own element, so in-place storage is safe.
            macro_rules! launch {
                ($ty:ty) => {
                    epilogue::launch::<$ty>(
                        &client,
                        count,
                        dim,
                        product.into_tensor_arg(),
                        bias.into_tensor_arg(),
                        result.clone().into_tensor_arg(),
                        self.has_bias,
                        self.activation,
                    )
                };
            }
            match dtype {
                DType::F16 => launch!(half::f16),
                DType::BF16 => launch!(half::bf16),
                DType::F32 => launch!(f32),
                _ => return Err(ExecutionError::generic("Unsupported linear dtype")),
            };
            EPILOGUES.fetch_add(1, Ordering::Relaxed);
        }
        CALLS.fetch_add(1, Ordering::Relaxed);
        handles.register_float_tensor::<CubeBackend>(&self.desc.outputs[0].id, result);
        Ok(())
    }
}

pub fn forward(
    x: BurnTensor<3>,
    w: BurnTensor<2>,
    bias: Option<BurnTensor<1>>,
    activation: usize,
) -> Option<BurnTensor<3>> {
    let [b, n, _] = x.dims();
    let o = w.dims()[1];
    let x = x.inner().try_into_primitive::<Gpu>().ok()?;
    let w = w.inner().try_into_primitive::<Gpu>().ok()?;
    let client = x.client.clone();
    let compute = client.device().client();
    let hw = &compute.properties().hardware;
    if hw.max_units_per_cube < 256 || hw.max_cube_dim.0 < 256 {
        return None;
    }
    let has_bias = bias.is_some();
    let mut inputs = vec![x.into_ir(), w.into_ir()];
    if let Some(bias) = bias {
        inputs.push(bias.inner().try_into_primitive::<Gpu>().ok()?.into_ir());
    }
    let out = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([b, n, o]),
        inputs[0].dtype,
    );
    let desc = CustomOpIr::with_scalars(
        "lino_native_linear_v1",
        &inputs,
        &[out],
        vec![
            ScalarIr::UInt(has_bias as u64),
            ScalarIr::UInt(activation as u64),
        ],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            LinearOp {
                desc,
                has_bias,
                activation,
            },
        )
        .outputs();
    Some(BurnTensor::from_primitive::<Gpu>(out))
}
pub fn stats() -> serde_json::Value {
    serde_json::json!({"projections":CALLS.load(Ordering::Relaxed),"epilogues":EPILOGUES.load(Ordering::Relaxed)})
}
