# Memory and exact-output performance pass

## Contract

Target: reduce the saved ~93–97 second inference time without changing output
values or materially increasing VRAM. No Rust build, cargo check, Rust test,
inference or GPU benchmark was run for this revision, at the user's request.
Do not interpret the changes as measured speedups or established bitwise parity.

The existing weights, dtype, GEMM/convolution algorithms and shapes, attention
scales, all 21 Gaussian taps, FP32 accumulation order, native-dtype rounding
boundaries, padding, random-number consumption and pixel context groups remain.
The legacy switch keeps the previous memory implementations for comparison.

## Implemented changes

* **Horizontal Gaussian halo reuse:** 256 threads stage 276 native-dtype samples
  in shared memory instead of each thread separately loading its 21 inputs.
  Boundary taps are still skipped, not replaced with extra zero additions.
* **Vertical Gaussian and layout conversion:** a 32-channel × 32-pixel shared
  tile makes both NCHW reads and NHWC writes contiguous across neighboring lanes.
  Results are cast to the original dtype before the shared-memory transpose.
* **Foreground-only final pass:** when the mask is sparse, compute the vertical
  Gaussian only for requested foreground pixels and emit compact pixel-major
  features directly. All preceding operations, including horizontal filtering,
  remain dense. Every contributing halo sample remains available. The final
  feature buffer drops from 128 MiB/light in FP16 to `foreground * 256 * 2` bytes.
  No full dense feature copy is retained on the GPU for the decoder.
* **Portable compact fallback:** CPU/reference paths gather only foreground rows
  after dense smoothing, with at most an 8 MiB selected-feature temporary.
  Compact indices use less than 3 MiB even at almost-full occupancy. The direct
  GPU path has no selected-feature temporary. Host features are also compact.
* **Decoder mapping:** keep foreground rows in raster order and translate the
  original shuffled pixel IDs to compact rows. The two original RNG shuffles,
  array-split batch sizes, RGB order and output scatter are unchanged.
* **Ownership-checked allocation reuse:** softmax scores, LayerNorm inputs and
  positional embedding inputs reuse storage only when the backend reports
  exclusive ownership and exact contiguous strides. Residual/shared inputs keep
  separate outputs. Arithmetic kernels for these operations are unchanged.
* **Preallocated chunk outputs:** DINO batches, independent frame/light batches,
  large MLP chunks and long-attention query chunks write into final output storage
  rather than retaining every chunk and concatenating a second full copy. GEMM
  and attention chunk sizes and padding remain unchanged.

Shared-memory tiles use only 552/1104 bytes for the horizontal FP16/FP32 pass and
2112/4224 bytes for the vertical pass, per workgroup. These are not large global
VRAM allocations. Actual allocator reservation and total peak VRAM still need
measurement; fewer live buffers do not guarantee immediate driver reclamation.

## Changes intentionally excluded under exact-output requirements

Convolution-to-GEMM, FlashAttention, resize/convolution fusion, new activation
fusions, shorter Gaussian support, quantization, changed batch/context sizes,
different checkpoints and altered autotune/cache policy can change reduction
order, intermediate rounding or selected kernels. They are not part of this pass.
The earlier analysis suggested some of these before bitwise equality was required.

## Reporting

Metadata schema version 2 adds:

* Per-block encoder/total time and foreground count, even without verbose output.
* Canonical/wavelet backbone scopes, canonical fusion/readback, wavelet
  fusion/smoothing/readback, CPU wavelet transform, decoder host gathering and
  decoder output readback scopes.
* Feature readback bytes/calls and sampled allocator reserved/in-use maxima.
  These samples are **not** a device-wide VRAM high-water mark.
* Startup, model-load and process-elapsed times. Startup is a once-per-process
  observation repeated in each view's metadata, not a per-view charge.
* Correct attention route labels: `gemm_subgroup_softmax`,
  `gemm_scalar_softmax`, `subgroup_streaming`, `scalar_streaming`.
  Both process-total and per-view dispatch counts are recorded.
* `--profile-sync`: waits at span boundaries and adds convolution parameter/shape
  scopes. This helps separate queued compute from readback, but changes fusion,
  overlap and execution overhead. Times are inclusive CPU wall times, not GPU
  timestamps; nested scopes must not be added together.

Normal runs remain asynchronous. A long readback scope can include all GPU work
queued since the preceding completion boundary; it is not a transfer benchmark.

## User-run validation

Keep the existing working executable and its outputs. Build manually as usual.
Run the same model, input, precision, seed, normalization, image count and batch
settings into a new directory. Include EXR output; PNG alone can hide differences.
Use the same backend, hardware and driver for exact comparisons. Batch mode must
start from the same ordered view list because RNG state advances between views.

Compare one output-view directory at a time:

```powershell
.\target\release\lino.exe compare --reference .\experiments\baseline --candidate .\experiments\optimized
```

The comparison requires matching image file sets, shapes and types, compares all
decoded EXR channel bits (including signed zero) and PNG bytes, and exits nonzero
on any difference. It does not compare compressed file metadata or tolerate errors.

Also run the new executable with `--legacy-memory-path` into a separate directory
to isolate this pass from differences caused by rebuilding dependencies/compiler.
Test standard/PBR, full/sparse masks, mask edges, multiple blocks and supported
precisions. GPU compiler code generation can affect bitwise equality even where
source-level arithmetic order is preserved; a successful comparison is required.

Use ordinary runs for timing and VRAM. Run `--profile-sync --verbose` separately
for diagnosis, and retain its metadata/profile. Compare equivalent cold-process
and later-view timings rather than attributing warmup differences to kernel speed.

CPU-only indexing checks, which do not build Rust:

```powershell
python -B tools/check_memory_layout.py
```

These check halo addresses/tap order, dense/sparse tile coverage, shared transpose
mapping including partial tiles, exact FP32 index encoding, and native-bit compact
gather equivalence after shuffling. They do not execute the CubeCL shaders.
