//! Experimental direct FlashAttention. No quadratic score allocation or autotune fallback.
use burn::tensor::{Tensor as BurnTensor, TensorCreationOptions};
use burn_backend::{ExecutionError, ops::AttentionModuleOptions};
use burn_cubecl::{
    CubeBackend,
    fusion::{CubeFusionHandle, FusionCubeRuntime},
    kernel::attention::flash_attention,
};
use burn_fusion::{
    Fusion,
    custom::{
        CustomOpIr, HandleContainer, Operation, OperationIr, OperationOutput, StreamId, TensorIr,
    },
};
use cubek::attention::forward::{
    launch::{BlueprintStrategy, Strategy},
    routines::blackbox_accelerated::BlackboxAcceleratedStrategy,
};
use std::{
    collections::HashSet,
    sync::atomic::{AtomicU64, Ordering},
    sync::{Mutex, OnceLock},
};
type Gpu = Fusion<CubeBackend>;
static ACCELERATED: AtomicU64 = AtomicU64::new(0);
static UNIT: AtomicU64 = AtomicU64::new(0);
fn failures() -> &'static Mutex<HashSet<String>> {
    static SET: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    SET.get_or_init(|| Mutex::new(HashSet::new()))
}
#[derive(Debug)]
struct FlashOp {
    desc: CustomOpIr,
    key: String,
}
impl Operation<FusionCubeRuntime> for FlashOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<CubeFusionHandle>,
    ) -> Result<(), ExecutionError> {
        let ([q, k, v], [out]) = self.desc.as_fixed();
        let [q, k, v] = [q, k, v].map(|ir| handles.get_float_tensor::<CubeBackend>(ir));
        let options = AttentionModuleOptions {
            scale: None,
            softcap: None,
            is_causal: false,
        };
        let accelerated = if failures().lock().unwrap().contains(&self.key) {
            None
        } else {
            Some(
                flash_attention(
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    None,
                    None,
                    options,
                    Strategy::BlackboxAccelerated(BlueprintStrategy::Inferred(
                        BlackboxAcceleratedStrategy {
                            num_planes: 4,
                            seq_q: 1,
                            seq_kv: 1,
                        },
                    )),
                )
                .or_else(|_| {
                    flash_attention(
                        q.clone(),
                        k.clone(),
                        v.clone(),
                        None,
                        None,
                        options,
                        Strategy::BlackboxAccelerated(BlueprintStrategy::Inferred(
                            BlackboxAcceleratedStrategy {
                                num_planes: 2,
                                seq_q: 1,
                                seq_kv: 1,
                            },
                        )),
                    )
                }),
            )
        };
        let result = match accelerated {
            Some(Ok(result)) => {
                ACCELERATED.fetch_add(1, Ordering::Relaxed);
                result
            }
            error => {
                if let Some(Err(e)) = error {
                    failures().lock().unwrap().insert(self.key.clone());
                    crate::progress::message(format!(
                        "Experimental flash: unit fallback for {} ({e:?})",
                        self.key
                    ));
                }
                UNIT.fetch_add(1, Ordering::Relaxed);
                flash_attention(q, k, v, None, None, options, Strategy::Unit(BlueprintStrategy::Inferred(())))
                    .map_err(|e| ExecutionError::generic(format!("Experimental flash unsupported for {}: {e:?}; rerun without --flash-attention (or with --no-experimental-flash)", self.key)))?
            }
        };
        handles.register_float_tensor::<CubeBackend>(&out.id, result);
        Ok(())
    }
}
pub fn forward(
    q: BurnTensor<4>,
    k: BurnTensor<4>,
    v: BurnTensor<4>,
    scale: f64,
    valid_m: usize,
) -> Option<BurnTensor<3>> {
    let [b, h, n, d] = q.dims();
    let m = k.dims()[2];
    if valid_m == 0 || valid_m > m || d % 16 != 0 {
        return None;
    }
    // Check before constructing GPU-only work, preserving CPU/reference fallback.
    q.clone().inner().try_into_primitive::<Gpu>().ok()?;
    let factor = scale * (d as f64).sqrt();
    // CubeK 0.3 applies 1/sqrt(D), ignoring AttentionModuleOptions.scale.
    let q = if factor == 1.0 { q } else { q * factor };
    let pad = n.div_ceil(64) * 64;
    let q = if pad > n {
        let zero = BurnTensor::<4>::zeros(
            [b, h, pad - n, d],
            TensorCreationOptions::new(q.device()).with_dtype(q.dtype()),
        );
        BurnTensor::cat(vec![q, zero], 2)
    } else {
        q
    };
    // Never pad keys: dummy keys would corrupt the normalization denominator.
    let k = if valid_m < m {
        k.slice([0..b, 0..h, 0..valid_m, 0..d])
    } else {
        k
    };
    let v = if valid_m < m {
        v.slice([0..b, 0..h, 0..valid_m, 0..d])
    } else {
        v
    };
    let key = format!(
        "{:?}/{:?}/{:?}/{:?}",
        q.device().identity(),
        q.dtype(),
        q.dims(),
        k.dims()
    );
    let q = q.inner().try_into_primitive::<Gpu>().ok()?;
    let k = k.inner().try_into_primitive::<Gpu>().ok()?;
    let v = v.inner().try_into_primitive::<Gpu>().ok()?;
    let client = q.client.clone();
    let output = TensorIr::uninit(client.create_empty_handle(), q.shape.clone(), q.dtype);
    let desc = CustomOpIr::with_scalars(
        "lino_experimental_flash_v1",
        &[q.into_ir(), k.into_ir(), v.into_ir()],
        &[output],
        vec![],
    );
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            FlashOp { desc, key },
        )
        .outputs();
    let out = BurnTensor::<4>::from_primitive::<Gpu>(out);
    let out = if pad > n {
        out.slice([0..b, 0..h, 0..n, 0..d])
    } else {
        out
    };
    Some(out.swap_dims(1, 2).reshape([b, n, h * d]))
}
pub fn stats() -> serde_json::Value {
    serde_json::json!({"accelerated":ACCELERATED.load(Ordering::Relaxed),"unit":UNIT.load(Ordering::Relaxed)})
}
