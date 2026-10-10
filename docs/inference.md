# Face alignment and resident inference

For unaligned images, `embed(image, width, height, landmarks)` accepts packed
RGB and five landmarks per face. Pass the original RGB image and detections
from [scrfd-hrx](https://github.com/zacharydenton/scrfd-hrx). `alignment::crop`
exposes alignment separately. Affine fitting, inversion and bilinear sampling
use FP32, with black borders and ties-to-even rounding to RGB bytes;
`alignment::transform` returns `[[f32; 3]; 2]`. Crops are numerically qualified
against the FP64 reference, not promised to be pixel-identical.

`embed` uploads the source image once, samples crops using HRX's shared GPU
affine operation, and downloads only embeddings plus four geometry-status bytes
per face. FP32 similarity fitting and inversion also run on GPU. For a shared
device pipeline, load with `load_in(path, &context, max_batch)`
and use `submit_image(&image, &landmarks)` with a resident U8 NHWC
`[1,height,width,3]` image and F32 `[faces,5,2]` landmarks (strided detection-row
views are accepted). It returns `FaceInference`; embeddings and geometry status
stay resident, and `wait` checks geometry before returning results. `align`
exposes resident crops and status separately; `submit` accepts
already aligned resident crops. These device methods accept one nonempty batch
up to `max_batch` and report bounded-capacity backpressure. The standalone
`alignment::crop` function is an explicitly requested CPU utility, not a fallback.

`Options` selects the device and resident batch size (default 16, range 1–64).
Larger batches are chunked automatically. Activation memory is 3.8 MB per
resident image, plus weights and I/O.

## Weights

The default is `recognition/model.onnx` from a pinned revision of
[immich-app/buffalo_l](https://huggingface.co/immich-app/buffalo_l/tree/d09715916a0778919a770c343533641e250b8699),
reusing the Hugging Face cache. Use `ArcFace::load(path, options)` or
`--model w600k_r50.onnx` for local weights.

`--offline` or `HF_HUB_OFFLINE=1` requires cached weights. `HF_HOME` and
`HF_HUB_CACHE` select the cache location. `HRX_OFFLINE=1` separately disables
runtime downloads.

## Performance and tests

Models retain weights, buffers and compiled kernels, replaying HRX graphs on
warm calls. See [benchmarks](../docs/optimization-2026-09-10.md).
Timings use synchronized host clocks, not GPU timestamps.

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo test --release -- --include-ignored --test-threads=1
```

The full suite checks numerical references, face alignment, RGB conversion and
graph replay. GPU tests require `gfx1151`; model tests download the pinned
weights when needed, unless `ARCFACE_MODEL` supplies a local path. Run `cargo doc --open` for
the API, and see [CHANGELOG.md](../CHANGELOG.md) for migrations.

## Runtime

Requires HRX 0.10.1. Graph preparation evaluates each Loom kernel's compiled
launch metadata; cached inference reuses the resulting commands. Kernel tile
sizes and workgroup dimensions have one source of truth in Loom. Convolution
and projection accumulators remain FP32.
