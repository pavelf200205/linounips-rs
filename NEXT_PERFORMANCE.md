# Defaults and the next convolution pass

The user's `normal_01_383956ba3d4e_run2/metadata.json` records 38.3273 seconds
inference, 27.7404 seconds in the encoder, 12,008 convolution chunks and 6,004
chunk completion waits. The wait scopes total 17.0360 seconds and include queued
GPU computation. This does not establish how much time is pure synchronization
overhead. All 268 flash operations used the accelerated path, with zero unit
fallbacks. The user reported no noticeable quality difference and approved those
changes as defaults.

No build, cargo check, Rust tests, inference or GPU benchmarking was performed
for this revision. CPU transform/indexing tests, rustfmt and PowerShell parsing
are the validation available so far. GPU correctness, runtime support, numerical
behavior, speed and VRAM remain to be measured by the user.

## Default behavior

Ordinary inference now enables GEMM convolutions, fused resize/convolution and
FlashAttention without `--experimental-fast`. The old positive switches remain
accepted for compatibility. `--burn-convolutions`, `--no-fused-resize-conv` and
`--no-flash-attention` disable those routes independently. The old spelling
`--no-experimental-flash` remains an alias for `--no-flash-attention`.

The next pass also enables these changes:

* **One custom operation per convolution layer:** packing, GEMM and output stores
  are submitted directly through CubeCL, avoiding a fusion operation graph for
  each individual chunk. GEMM uses the existing native backend and tuner.
* **Reusable layer workspaces:** each layer allocates one packed-input buffer
  and one GEMM-output buffer. All chunks reuse them in GPU command order:
  pack → GEMM → store → next pack. No CPU wait is needed between these dependent
  commands. Inputs remain available until the operation finishes submitting.
* **Fused output store:** a single kernel combines bias addition, optional
  leaky-ReLU and tail-aware NCHW scattering. The GEMM result retains its native
  precision before the bias/activation arithmetic. There is no full-result
  concatenation or per-chunk output allocation. Residual and skip behavior stays
  unchanged; only the convolution whose output immediately feeds leaky-ReLU uses
  the fused activation.
* **Layer-level completion windows:** default `--conv-layer-window 2` limits CPU
  lead to two direct convolution layers. `--inflight-mib` independently limits
  the estimate of their explicit temporary buffers. Readback completes all prior
  work and resets this accounting. This replaces thousands of per-chunk waits
  with a much smaller number of layer-window waits.

The input packing budget is still 16 MiB by default. The second scratch buffer
costs `output_channels * chunk_rows * dtype_bytes`. Live inputs/outputs, allocator
reservation and backend autotuning scratch are additional; these are not total
VRAM caps. Two reusable buffers per submitted layer replace many allocations
per layer. Expected memory benefits still require measurement. Queue-order
dependence is explicit and intentional; the layer's scratch buffers must not be
written from a separate unordered GPU stream.

## Optional Winograd

`--winograd-convolutions` selects F(2×2,3×3) for eligible stride-1, padding-1,
3×3 convolutions. Other convolutions, including the fused stride-2 output head,
keep GEMM packing. Input tiles use BᵀdB, weights use GgGᵀ, and results use AᵀMA.
Sixteen transform-domain channel products generate each 2×2 output tile instead
of the direct method's 36. Boundary padding, odd-size output tails, all channels,
all images and all spatial dependencies are retained.

This is an algebraic change of algorithm, not a context or resolution shortcut.
It changes numerical conditioning and transform-domain rounding. FP16/BF16
accuracy and output quality need separate validation, so Winograd is off by
default. Arithmetic reduction does not guarantee a speedup once transforms,
dispatches, packing layout and GEMM utilization are included.

Winograd reuses two chunk buffers just like direct convolution. Its packed-input
budget uses `16 * channels * tile_rows * dtype_bytes`; GEMM output storage uses
`16 * output_channels * tile_rows * dtype_bytes`. Transformed weights are cached
with a 64 MiB cap per loaded model. Beyond the cap, weights are transformed on
demand without retaining another cached copy. Original weights stay available
for fallback, so optional Winograd adds up to 64 MiB of resident cached weights
plus current workspaces. Cache bytes are reported in metadata.

## Comparison switches

Append to an existing inference command, with a fresh output directory:

* No extra flags: new defaults and reusable direct dispatch.
* `--legacy-fast-path`: restore the measured 38-second preset (old convolution
  dispatch and separate activation), with Winograd disabled.
* `--legacy-conv-dispatch`: restore just the old pack/GEMM/scatter dispatch.
* `--separate-conv-activation`: retain reusable workspaces but separate activation.
* `--conv-layer-window 1` or `4`: vary the direct layer completion window.
* `--winograd-convolutions`: try the transform algorithm.

`--conv-queue-depth` only governs the old high-level convolution dispatch. A direct
layer reuses its buffers rather than queueing a new workspace for each chunk.
The existing workspace, MLP/query chunk, image batch and allocator controls remain.
`--legacy-fast-path` is a dispatch baseline, not a snapshot of the old compiler,
driver or dependencies. Compare against the saved old executable as well.

`tools/benchmark_experiments.ps1` now compares the old fast preset, the new default,
separate activation, Winograd and layer windows. `-FullMatrix` adds workspace
sizes, Winograd workspace sizes, wide fusion, MLP/query chunks, retained pool
allocations and disabling flash. It performs no build. Summary CSV/JSON and
strict EXR/PNG equality logs are retained for every case.

## Reporting and CPU checks

Metadata includes the effective direct-dispatch/activation/window/Winograd
settings, cached transform-weight bytes and process-total actual layer/chunk
counts. `conv/direct_window_wait` records high-level waits. Scratch-allocation
counts refer only to explicit custom-op workspaces, excluding backend tuning.
Decoder input upload and fusion submission now have separate timings, covering
work that was previously outside the decoder operation spans.

Custom-op launches are asynchronous. High-level scope times and direct operation
counts do not measure individual GPU kernels. `--profile-sync` perturbs execution
and should be used in a separate diagnostic run. The strict comparison command
still rejects any changed EXR bits; visually unchanged output is not a proof of
bitwise equivalence.

```powershell
python -B tools/check_winograd.py
```

This independent matrix-algebra oracle checks expanded transform formulas,
equivalence to direct convolution in double precision, multiple channels/batches,
boundary padding, odd sizes, chunk tails and one-to-one direct output stores.
It does not execute GPU kernels or establish low-precision numerical quality.
