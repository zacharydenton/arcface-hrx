# arcface-hrx

ArcFace face embeddings in Rust with [HRX](https://github.com/zacharydenton/hrx-rs)
and Loom kernels for AMD GPUs. Runs InsightFace's `w600k_r50` model with
five-point face alignment. Library and CLI in one crate.

Requires Rust 1.91+, Linux x86-64, glibc 2.43+ and a Radeon 8060S (`gfx1151`).
HRX downloads its pinned runtime/compiler bundle; model weights use Hugging Face Hub.

## CLI

```bash
cargo run --release -- --input aligned-crops.rgb --output embeddings.f32
```

Input is packed uint8 **RGB**, 112×112×3 bytes per aligned crop. Output is
little-endian float32, 512 values per crop. Add `--benchmark 100` for warm
inference timings; benchmark input must fit one resident batch.

## Rust

```rust
use arcface_hrx::{ArcFace, Options};

fn main() -> anyhow::Result<()> {
    let crops = std::fs::read("aligned-crops.rgb")?;
    let mut model = ArcFace::from_pretrained(Options::default())?;
    let embeddings = model.embeddings(&crops)?;
    println!("{} embeddings", embeddings.len());
    Ok(())
}
```

`embeddings` accepts aligned RGB crops and returns unnormalized `[f32; 512]`
vectors, matching InsightFace. Use `similarity` for cosine similarity.

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
warm calls. See [benchmarks](docs/optimization-2026-09-10.md).
Timings use synchronized host clocks, not GPU timestamps.

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo test --release -- --include-ignored --test-threads=1
```

The full suite checks numerical references, face alignment, RGB conversion and
graph replay. GPU tests require `gfx1151`; model tests download the pinned
weights when needed, unless `ARCFACE_MODEL` supplies a local path. Run `cargo doc --open` for
the API, and see [CHANGELOG.md](CHANGELOG.md) for migrations.

## License

Code: [Apache-2.0](LICENSE). InsightFace weights have separate terms and are
not bundled. See [third-party notices](THIRD_PARTY_NOTICES.md).
