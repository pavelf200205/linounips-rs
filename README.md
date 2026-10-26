# lino-rs

Inference-only Rust implementation of the standard and PBR LINO-UniPS graphs,
based on upstream revision 4ddb4113d0196b853978d2cc72de582611b01ebd.

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

## Memory changes

* All global attention retains all lights and all wavelet streams. Online softmax
  tiles both queries and keys with FP32 statistics instead of full score matrices.
  --score-budget-mib defaults to 64 and governs score temporaries, not total VRAM.
* Per-image DINO/frame blocks use --encoder-batch (default 1). Cross-light blocks
  are only batched across independent spatial positions, never split by lights.
* Four concatenated transformer-stage outputs are copied to compact CPU backing
  one stream at a time. No full four-stage GPU stack or unused light-token outputs.
* Fusion runs one stream at a time, releases intermediates by ownership and performs
  FP32 interpolation in 16-channel slices. It never creates the huge whole-batch
  FP32 upsample plus a second whole-batch BF16 result.
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
and bounded scheduling trade speed for memory. The online GEMM attention is a
portable baseline, not a claimed speed match for PyTorch FlashAttention.

--pixel-batch defaults to 2048 and preserves upstream sample-group semantics.
Changing it changes the regressor's spatial context and can change output quality.
For memory tuning prefer encoder-batch and score-budget-mib first. No patch-size,
checkpoint or attention-receptive-field reduction is used as a memory shortcut.

See NOTICE and the included licenses for upstream attribution.
