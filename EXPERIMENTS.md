# Experimental inference routes

**Current defaults supersede the opt-in descriptions below:** the user tested
GEMM convolution, fused resize/convolution and FlashAttention at 38.33 seconds
with no noticeable quality change and requested them as defaults. They now run
without flags; the positive switches remain accepted for compatibility. Use
`--burn-convolutions`, `--no-fused-resize-conv`, and `--no-flash-attention` to
disable them separately. See [NEXT_PERFORMANCE.md](NEXT_PERFORMANCE.md) for the
new default dispatch, optional Winograd and the updated benchmark cases.

The cb276ee run2 metadata records 94.70 seconds inference: 24.49 seconds canonical
fusion/readback and 41.07 seconds wavelet fusion/smoothing/readback. CPU gathering
is 0.16 seconds. The new experiments target dense GPU work and dispatch overhead.
Readback scopes still include queued computation; they are not PCIe measurements.

No Rust build, cargo check, inference or GPU benchmark was run for this revision.
Formatting, CPU packing-oracle checks and PowerShell syntax checks are the only
validation. Successful compilation, runtime support, numerical behavior, speed
and peak VRAM remain to be established on the user's machine.

## First experiments

Append these options to the same inference command, using a new output directory:

* `--gemm-convolutions`: replace ordinary dense convolutions with bounded NCHW
  patch packing followed by native GEMM. Original OIHW weights are used directly,
  with no duplicate GPU weight bank. Fixed power-of-two column chunks reduce
  shape variation. Stride, padding, channels, weights and receptive fields stay
  the same. CPU and reference-kernel modes retain the existing convolution.
* `--fused-resize-conv`: replace the final 512×512 upsample followed by stride-2
  3×3 convolution with bilinear sampling directly into packed GEMM patches.
  align_corners coordinates and the intermediate native-dtype cast are retained.
  The large full-resolution resized activation is eliminated. This flag can be
  tried independently of GEMM conversion of the other convolutions.
* `--flash-attention`: try direct accelerated FlashAttention, then unit flash.
  No quadratic score allocation or attention autotuner fallback is allowed.
  Only queries are padded. Padded decoder keys are removed using valid_m.
  Query scaling compensates for CubeK 0.3 ignoring the supplied scale option.
  Accelerated failures are cached per device/dtype/shape; dispatch counts expose
  actual accelerated/unit usage. If both strategies fail, the experiment errors
  clearly; rerun without this flag. A unit-flash route can be slower than GEMM.
* `--experimental-fast`: enable all three paths above. To retain the convolution
  experiments while excluding flash, add `--no-experimental-flash`.

These experiments preserve the mathematical model, all lights, all tokens and
the pixel groups. They can change floating-point rounding/reduction order and
therefore **do not promise bitwise equality**. The existing default path remains
available. The strict compare command continues to reject any changed output
bits; an experimental result is not automatically accepted as equal quality.

## Memory and scheduling controls

`--conv-workspace-mib 16` bounds each packed input workspace (default 16 MiB).
Try 8 or 32. `--conv-queue-depth 2` completes at most two chunk submissions at a
time and drains tails at convolution boundaries. Try 1 or 4. A depth of 0 removes
these waits, permits many simultaneous temporaries and is intentionally absent
from the benchmark matrix. The bound is not a total VRAM cap: GEMM output chunks,
live tensors, allocator reservations and backend scratch also consume memory.

`--wide-wavelet-fusion` processes the four wavelet streams in a single fusion
batch instead of two pairs. Each stream remains independent. This may reduce
dispatch overhead but increases live activations, so the low-memory preset does
not enable it. It can change selected GEMM/convolution kernels and rounding.

`--mlp-chunk-tokens 8192` tries fewer, larger MLP operations. Default 0 retains
the previous budget-based logic. Overrides round down to multiples of 512, with
a minimum of 512, and may increase live activations substantially.

`--attention-query-chunk 1024` raises the maximum GEMM query chunk (default 512).
The existing score budget still applies, and the actual chunk is rounded down
to a power of two. The score budget may prevent the requested size from being used.

`--pool-cache-mib 0` retains allocator caches across blocks. Default 256 retains
the existing cleanup threshold. This has little expected effect on a single-block
view, but is useful for multi-block/multi-view comparisons and may raise reserved
memory. No driver cache or persistent autotuning policy is changed by this pass.

Metadata records every setting and route/chunk counts under `experiments`, plus
actual process-total flash dispatch counts. Existing `--profile-sync` adds packing,
GEMM and queue-wait timings but perturbs execution; time ordinary runs separately.

## Manual matrix (never builds)

From lino-rs after building manually:

```powershell
.\tools\benchmark_experiments.ps1 -Executable .\target\release\lino.exe -Model ..\lino-weights\lino.safetensors -Data ..\data\own_objects_normals\horse_960\mvps_png_full\view_00.data -Out .\experiments\matrix_01
```

The six default cases are baseline, GEMM, fused resize/convolution, both
convolution experiments, flash, and the combined preset. `-FullMatrix` adds
workspace sizes, queue depths, four-stream fusion, MLP/query chunks and allocator
retention. `-Runs 2` repeats fresh processes and reverses experimental order on
alternate rounds. `-Mode pbr -Model PATH_TO_PBR_WEIGHTS` covers PBR separately.

The script refuses to overwrite an existing output root, saves command logs,
metadata, strict comparison logs and summary CSV/JSON, and continues to later
experiments after an unsuccessful native process. It compares each experiment
against that round's baseline using decoded EXR bits and PNG bytes. Also compare
the baseline against your previous executable's output to detect rebuild effects.
There is no image-count, context, normalization, mask or pixel-group shortcut.

CPU-only address/reduction oracle (does not run CubeCL):

```powershell
python -B tools/check_experimental_packing.py
```
