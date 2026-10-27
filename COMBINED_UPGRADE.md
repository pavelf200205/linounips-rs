# Combined decoder, projection and persistence upgrade

## Defaults and scope

Winograd is now enabled by default as requested. `--no-winograd` disables it;
the old positive `--winograd-convolutions` flag is accepted for compatibility.
GEMM convolution, fused resize/convolution, reusable convolution workspaces and
FlashAttention remain enabled. All image/light counts, transformer tokens,
attention key masks, RNG draws and pixel context groups are retained.

The previous measured PBR run averaged about 30.56s with Winograd. Its decoder
submission/deferred-execution scopes cost about 7.6s, and the backbone about 14s.
The changes below target these costs. They are not newly measured speedups.

No Rust build, cargo check, Rust tests, model inference or GPU benchmark was run
for this revision. Cargo metadata with the locked dependency graph resolved
successfully (including the local cache repair). CPU layout/algebra checks and
format/script syntax checks passed. Successful compilation and GPU numerical,
performance and VRAM behavior still require the user's single manual build/test.

## Packed light attention

Short MAB attention consumes Q `[B,N,C]` and packed KV `[B,M,2*C]` directly.
It avoids constructing/slicing separate head-major K and V tensors and fuses
the `Q + attention` residual into the output store. Query scaling still occurs
before dot accumulation; softmax statistics and accumulation remain FP32, with
native-dtype attention rounding before residual addition. Expanded constant
pooling queries use their actual zero batch stride. The supported route covers
up to 64 lights and head width 128 on subgroup-capable GPUs. Unsupported layouts,
devices, masks or longer sequences fall back to the existing attention paths.

`--legacy-packed-attention` restores the previous decoder attention layout.
Backbone attention remains on its existing paths.

## Native projections and activation epilogues

Linear projections now use a single custom operation to submit native GEMM and
its optional bias/activation epilogue. Output allocation is explicitly contiguous;
the newly created product buffer is reused in place. This reduces fusion-graph
planning and intermediate allocations. Existing weight layout and GEMM shapes
are retained. Bias-free projections without activation need no extra epilogue.

GELU uses erf, not the tanh approximation. Its native-dtype division/erf/add/mul/
division sequence follows Burn's implementation; it is not intentionally replaced
with an unrounded FP32 approximation. ReLU and leaky-ReLU can also be fused.
Backend kernel selection and operation fusion can still change rounding, so
bitwise identity is not promised. `--legacy-linears` restores Burn linear and
separate activation dispatch for both backbone and decoder. CPU/reference modes
retain the portable tensor implementations.

## PBR head packing

Four `[384,192]` hidden projections are concatenated into one `[384,768]` GEMM,
with ReLU applied to each head's own channels. This retains four independent
heads; there is no sharing or averaging of learned parameters. Extra packed
weights occupy about 0.57 MiB, only in PBR mode with this path enabled. A batch's
combined hidden activation is about 3 MiB at 2048 pixels/FP16; compared with one
sequential head this adds at most about 2.25 MiB of live hidden storage, subject
to backend queueing and allocation reuse.

The final 3/3/1/1 projections are evaluated by one subgroup kernel, accumulating
in FP32, rounding projection results to native dtype before bias, normalizing the
three normal components in FP32 and directly emitting `[normal XYZ, base RGB,
roughness, metallic]` as one FP32 tensor. This avoids four small GEMMs, separate
normal-normalization passes, per-head casts and final concatenation. Material
clamping/mapping and output reconstruction remain in the existing output code.
Different dot-reduction order can change results; compare numerical quality.

`--legacy-pbr-heads` restores four independent hidden/final paths and avoids
packed-weight preparation. `--separate-pbr-projection` retains the combined hidden
GEMM but restores the individual final GEMMs, useful for isolating differences.
Unsupported GPUs retain those final GEMMs automatically.

## Decoder padding and completion

The host adds native zero rows before the existing upload, avoiding GPU zero
allocations and concatenation for every group. Original valid pixel counts are
passed separately; padded rows remain excluded as keys in regressor attention.
The original shuffles and array-split groups are unchanged. The transient host
padding is less than 64 rows per group, about 0.28 MiB for nine FP16 feature maps.
`--legacy-device-padding` restores GPU-side padding.

Decoded maps now share one output tensor and a channel table for both standard
and PBR inference. There is still one completion/readback per group, retaining
the previous bounded decoder working set. No extra full feature stack or queued
decoder batch was introduced. Output scatter has a separate timing scope.

## Persistent artifacts and tuning

The binary now includes CubeCL persistence support. Default Windows root:
`%LOCALAPPDATA%/lino-rs/kernel-cache`. `LINO_CACHE_DIR` or `--kernel-cache-dir PATH`
overrides it. Other platforms use XDG cache location when available, otherwise
an executable-adjacent `.lino-kernels` directory. CPU mode uses RAM storage.
`--no-kernel-cache` restores the previous process-local storage behavior.

The store is namespaced by executable source revision; a different build revision
starts a different environment. Shader IR/device keys and backend namespaces
remain managed by CubeCL. The first process may compile/tune and populate the
store; subsequent processes of the same binary can reuse its artifacts. Driver
pipeline creation, new shapes and other initialization can remain process-local.
There is no hidden model warmup pass, timing exclusion or driver environment
override. Cache writes and first-use setup remain inside observed process times.

The repository-local cubecl-environment 0.11.0 patch disables experimental
multiprocess WAL on Windows only. This fixes the same default-WindowsIO opening
failure identified in unimsps-rs. Upstream source and licenses are in vendor/;
no global registry or sibling repository source was edited. Normal Windows file
locking remains: simultaneous processes should use distinct cache roots. Keep
the database and any WAL sidecars together. No automatic cache deletion occurs.

Startup verifies a real database read/write probe. Metadata records probe status,
whether an earlier process's probe existed, initial/current entries and compiled
artifact/autotune entry counts. Those counts prove stored data, not individual
cache hits. Failures are surfaced rather than silently claimed as cache reuse.

## One-build comparison plan

Build manually once, keep the previous executable, and run identical inputs into
new output directories. Run the new defaults at least twice so first-process
cache creation can be distinguished from later reuse. Keep precision, model,
normalization, seed, image count and pixel batch fixed. Test standard and PBR.

* `--legacy-decoder --no-kernel-cache` approximates the preceding Winograd-enabled
  revision's inference paths without requiring another build.
* `--no-kernel-cache` isolates the new decoder/projection routes from persistence.
* `--legacy-linears`, `--legacy-packed-attention`, `--legacy-pbr-heads`,
  `--separate-pbr-projection` and `--legacy-device-padding` isolate individual
  changes without a rebuild.
* `--legacy-fast-path --no-kernel-cache` restores the older non-Winograd fast
  dispatch; it is not a snapshot of old compiler/driver/dependency versions.

The updated `tools/benchmark_experiments.ps1` compares the preceding paths, new
defaults, no cache and individual decoder switches. Each case has its own cache
root reused across rounds, so `-Runs 2` measures its own warm store instead of
another case's. `-FullMatrix` adds more independent switches. The script never
builds and saves logs, metadata, strict EXR/PNG comparison results and CSV/JSON.
In standard mode the PBR-head switches naturally have no effect.

Metadata includes effective flags and actual process-total native projection,
packed-attention and fused-PBR dispatch counts. All timing scopes remain inclusive
CPU wall time. Use `--profile-sync` separately for diagnosis; it changes execution.

The strict `lino compare` command continues to reject any changed EXR bits, even
when differences are visually negligible. CPU-only checks:

```powershell
python -B tools/check_decoder_layout.py
python -B tools/check_memory_layout.py
python -B tools/check_experimental_packing.py
python -B tools/check_winograd.py
```

These validate layouts and algebra, not GPU arithmetic, low-precision accuracy or
the compiler's ability to build the new kernels. No performance claim is made
until actual binary runs are available.
