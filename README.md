# lino-rs

Inference-only Rust implementation of the standard and PBR LINO-UniPS graphs,
based on upstream revision 4ddb4113d0196b853978d2cc72de582611b01ebd.

## Current memory/performance pass

GEMM convolution, fused resize/convolution and FlashAttention are now enabled by
default, following the user's 38.33-second run and quality assessment. The next
pass adds reusable GPU convolution workspaces and fused convolution stores.
`--legacy-fast-path` restores the measured 38-second dispatch path.
`--winograd-convolutions` is an additional opt-in experiment. See
[NEXT_PERFORMANCE.md](NEXT_PERFORMANCE.md) for the changes, memory limits and
comparison commands. This revision has not been built or run on a GPU by the agent.

See [PERFORMANCE.md](PERFORMANCE.md) for the current implementation, exact-output
comparison procedure and timing interpretation. This pass retains arithmetic,
precision, attention algorithms and pixel grouping; it changes feature storage,
Gaussian memory access and intermediate buffer ownership. No Rust build or GPU
inference was run for this pass. CPU indexing checks and formatting passed;
actual bitwise output equality, speed and peak VRAM require your binary tests.

New options: `--legacy-memory-path` restores the previous memory paths for A/B
comparison; `--profile-sync` adds diagnostic synchronization and convolution-shape
timings. Do not use synchronized profiling runs to measure production throughput.
`lino compare --reference OLD_OUTPUT_DIR --candidate NEW_OUTPUT_DIR` checks EXR
floating-point bits and decoded PNG bytes with zero tolerance, without a GPU.

The older implementation notes below describe earlier passes. In particular,
current long attention uses GEMM plus softmax, not FlashAttention; dispatch counts
now use the correct names. Optimized final features are foreground-only, and
metadata includes per-block timings even without `--verbose`.

**Initial implementation: not compiled or run against checkpoints yet.** Source
syntax, dependency resolution and graph structure were inspected; this does not
establish numerical parity, output quality, speed or a measured VRAM limit.
The user performs builds and inference. No weights are bundled or downloaded.

## Dependencies and device support

Burn 0.22.0, CubeCL 0.11.0 and the resolved CubeK 0.3.0 match unimsps-rs.
WGPU/Vulkan is the default on Windows/Linux, including AMD and Intel devices.
There are no custom CUDA-only kernels. Optional CUDA requires --features cuda.
--api metal, --api dx12 and --cpu are also exposed through Burn; actual operator
and precision availability depends on the backend. Auto precision prefers BF16,
then FP16, with FP32 on CPU or devices lacking either half type.

Cross-run shader/tuning persistence is disabled. As in unimsps-rs, pinned
CubeCL's transitively enabled throughput storage is forced to its upstream RAM
fallback by using the existing executable file as its impossible directory root.
No vendored crates, storage patches, writable cache directory or cache CLI exist.

## Build manually

From this directory:

~~~powershell
.\tools\build_release.ps1
~~~

Or cargo build --release. The helper also preserves lino_<commit>.exe beside
Cargo's lino.exe; it supports -TargetDir for shared dependency output directories.
The current unimsps-rs binaries are not touched.

## Checkpoints

Use the standard lino.pth for normals; lino_pbr.pth is a different checkpoint and
is not an equivalent-quality replacement. Select the correct mode explicitly.
All required inference tensor names/shapes are checked when loading. PBR mode
requires all three material heads; missing tensors are errors, not random weights.
Unused training/confidence tensors are ignored because they do not affect outputs.

Rust reads F32/F16/BF16 safetensors directly. Passing .pth invokes the bundled
checkpoint conversion script using Python + PyTorch + safetensors, then inference
runs entirely in Rust. No Python model code is imported and no Python inference
runs. If needed, install the converter dependencies yourself:

~~~powershell
python -m pip install torch safetensors
.\target\release\lino.exe convert --model ..\lino-weights\lino.pth --out ..\lino-weights\lino.safetensors
~~~

--python accepts a specific Python executable. The exporter uses weights_only=True,
strips module./model. prefixes and checks the source SHA-256 before reusing its
converted artifact. A .pth input produces a sibling .lino-rs.safetensors file.
A source positional embedding on another square grid is converted using the
upstream bicubic/antialias interpolation to the inference grid of 32x32.

## Inference

Examples from lino-rs; replace checkpoint/data paths with actual locations:

~~~powershell
.\target\release\lino.exe infer --model ..\lino-weights\lino.pth --mode standard --data ..\data\own_objects_normals\horse_960\mvps_png_full\view_00.data --out .\experiments\normal_01 --nb_img 9
.\target\release\lino.exe infer --model ..\lino-weights\lino_pbr.pth --mode pbr --batch --data ..\data\own_objects_normals\horse_960\mvps_png_full --out .\experiments\pbr_01 --nb_img 9
~~~

--batch processes immediate subdirectories with one loaded model. Images start
with L by default (--prefix changes this), are sorted lexically and limited by
--nb_img/--images; -1 selects all. Supported inputs: PNG/JPEG/TIFF/EXR. Optional
mask.png must match input dimensions. Inputs are RGB, not BGR. The ROI follows
upstream's margin and exclusive-maximum convention. The ROI is resized to a
square multiple of 512; default cap 5632 avoids upstream's non-divisible 6000 cap.
The 512px blocks are interleaved tile_stride grids, not ordinary contiguous crops.
Only the current block is materialized; images and masks stay on the CPU.

--normalization eval (default) uses the upstream random mean/max blend.
--normalization max uses demo-style maximum normalization. --seed defaults to 42.
NumPy-compatible MT19937 permutations and array_split group sizes are used for
the two decoder pixel shuffles. Image subset selection is deliberately lexical,
unlike the random benchmark subset, so RNG consumption and predictions will not
match a benchmark run that randomly selected its images. Floating-point resize,
backend kernels and explicit half precision can also differ from OpenCV and
PyTorch mixed precision. This implementation does not claim bitwise equivalence.

Normal output: normal.exr (signed XYZ) and normal.png (XYZ mapped to 0..1).
PBR adds diffuse, roughness and metallic in both formats. Diffuse names the
upstream baseColor head; it is not a separate physically derived diffuse term.
Material values follow upstream: clamp raw output to [-1,1], then map to [0,1].
Scalar material EXRs replicate their value across RGB. PNGs include mask alpha.
Use --formats exr or --formats png to select a subset. metadata.json records mode,
revision, precision, ROI, file selection, settings and wall times. --verbose adds
block encoder/total timings and profile.json; these are not GPU timestamps.

### Console progress

Progress is enabled by default and follows SuperNormal's dependency-free stderr
display: one carriage-return line, refreshed at most four times per second.
It shows the view index/name, block index, canonical or wavelet branch, and the
current stage. Counts cover DINO layers/batches, transformer batches, global
attention query blocks, MLP chunks, fusion lights/layers, interpolation channels,
decoded pixels and output maps. Nested stages preserve branch context.

For example (the exact line is shortened to fit the terminal):

~~~text
View 1/20 view_00 block 1/4 | wavelet/global 2/4 / attention | [====----] 50% 73/145 q | 04:12 ETA ~00:20
~~~

Percentages and estimated remaining times refer to the current stage, not the
whole view. An ETA appears after two completed work units and can fluctuate,
particularly during first-use compilation/tuning. Elapsed time keeps updating
during long calls even if the completion counter has not moved. Counters follow submitted stages and existing readback boundaries; opaque fused
kernels show a launch stage with elapsed time rather than invented tile progress.
No GPU waits or tensor readbacks are added for progress. Verbose messages clear the live line;
redirected output uses plain snapshots every 15 seconds plus view completion.
Use --no-progress to disable the bar/heartbeat and periodic snapshots. Ordinary
result messages and optional --verbose reports remain available.

## Memory changes

* All global attention retains all lights and all wavelet streams. Online softmax
  tiles both queries and keys with FP32 statistics instead of full score matrices.
  --score-budget-mib defaults to 64 and governs score temporaries, not total VRAM.
* Per-image DINO/frame blocks use --encoder-batch (default 2). Cross-light blocks
  are only batched across independent spatial positions, never split by lights.
* Four concatenated transformer-stage outputs are copied to compact CPU backing
  one stream at a time. No full four-stage GPU stack or unused light-token outputs.
* Fusion runs one stream at a time, releases intermediates by ownership and performs
  FP32 interpolation in a fused streaming kernel with no full FP32 temporary.
  The reference fallback uses 16-channel slices.
* Only one light's dense fusion/smoothing pipeline is live on the GPU. The final
  maps live in CPU half precision (FP32 when requested), in pixel-major layout.
  Gaussian smoothing has no retained chunk list or concatenated duplicate.
* Decoder gathers only requested samples on CPU, uploading [pixels,lights,256].
  It never copies an entire dense feature stack for each pixel batch.
* Completed outputs scatter directly into CPU maps. Previous-block features are
  dropped before the next encoder pass; no all-block Unfold or output patch stack.

CPU RAM still grows with light count: final dense features alone cost 128 MiB per
light in half precision, plus transformer-stage backing and input images. GPU
weights and live global tokens also grow or remain resident; this is not a promise
of constant memory or that every light count fits every GPU. Explicit transfers
and bounded scheduling trade speed for memory. Attention now tries accelerated CubeK FlashAttention, then unit flash and
portable streaming kernels; it never falls back to a full dense score matrix.
Actual speed and memory must be measured on the target GPU.

--pixel-batch defaults to 2048 and preserves upstream sample-group semantics.
Changing it changes the regressor's spatial context and can change output quality.
For memory tuning prefer encoder-batch and score-budget-mib first. No patch-size,
checkpoint or attention-receptive-field reduction is used as a memory shortcut.

See NOTICE and the included licenses for upstream attribution.

## Optimization pass (requires your rebuild and validation)

* Position embeddings use compact separable lookup tables generated once per
  shape/precision. GPU kernels apply them directly: no repeated billion-element
  CPU trig loops or whole spatial-embedding uploads. Rotary positions use small
  reusable coordinate/frequency tables and a fused kernel as well.
* Bilinear interpolation reads four input samples, computes in FP32 and writes
  the requested dtype in one portable CubeCL kernel, preserving align_corners.
* Long attention tries accelerated flash configurations and then unit flash.
  Short light attention uses a fused subgroup online-softmax kernel. A scalar
  streaming fallback covers GPUs without subgroup support. Every fallback visits
  all keys with FP32 softmax statistics; none materializes full scores.
* Only queries are padded for accelerated-kernel alignment; keys/values are not
  padded, so normalization does not include dummy keys. CubeK 0.3's flash path
  ignores the supplied scale option; query scaling explicitly preserves LINO's
  differing normal/efficient attention scales. Floating-point rounding can still
  differ from the baseline and needs output validation.
* Decoder pre-gathers the next sample batch on the CPU while submitted GPU work
  finishes. Only compact host sample buffers are double-buffered; no additional
  full-resolution feature maps become resident on the GPU.
* Half features transfer as half data directly. Decoder gathers copy contiguous
  pixel-major rows without scalar half-to-float-to-half conversion. Stage backing
  is built directly in the target dtype instead of a full FP32 staging vector.
* Linear weights are packed into native GEMM layout once, without a duplicate
  resident copy. Normalization parameters and Haar/Gaussian filter buffers are
  also prepared/reused instead of recreated for each call.
* LayerNorm uses a subgroup kernel with FP32 register statistics and FP32 affine
  parameters, preserving each layer's epsilon. Unsupported devices/register
  sizes retain the tensor fallback. This removes the full FP32 input cast and
  separate normalization/cast passes for supported shapes.
* Small submissions queue asynchronously. --inflight-mib defaults to 64 and
  bounds an estimate of queued temporaries, with a secondary 32-operation guard.
  This is not a hard total-VRAM cap; model weights, live tokens, dense per-light
  outputs and backend workspaces are additional. Readbacks remain completion
  boundaries. Unused allocator reservation above 256 MiB is reclaimed between
  blocks, never inside hot attention loops. Geometry constants are capped at
  16 MiB per model; no new cross-run persistence is introduced.
* Independent-image batching defaults to 2; --encoder-batch 1 restores the
  smallest encoder working set. Pixel sampling remains 2048 by default and all
  global attention continues to see every light and wavelet stream.

Use --reference-kernels to bypass native attention/geometry/LayerNorm for diagnosis. It
retains bounded tiled attention, compact host transfers and the new scheduling;
it is not an exact replay of the old binary. --score-budget-mib bounds tiled
GEMM score temporaries in that fallback; fused flash/streaming have no full-score
allocation. Reducing --inflight-mib is the first queue-memory adjustment.

metadata.json now includes per-view operation counts/inclusive wall times and
process-total dispatch counts for accelerated flash, unit flash, subgroup and
scalar attention. With --verbose, profile.json contains blocks and operations.
Timings measure CPU wall time including waits encountered in that scope, not
GPU timestamps. Parent/child entries overlap and must not be summed.

No compilation, inference or benchmark was run by the coding agent. This pass
has source parsing/format and dependency-resolution checks only. Keep the known
working executable for comparison and use a fresh output directory for each run.
