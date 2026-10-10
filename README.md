# arcface-hrx

ArcFace face embeddings in Rust, using [Loom](https://github.com/ROCm/hrx-system)
kernels and [HRX](https://github.com/zacharydenton/hrx-rs) on AMD Strix Halo.
Runs InsightFace's `w600k_r50` model and returns 512-dimensional embeddings.

Warm aligned-crop inference measured **2.4 ms for one face** and **4.3 ms for six**
on Radeon 8060S. These are September 2026 measurements, including host I/O but
excluding model loading, compilation and image decoding.
[Measurements and methodology](docs/optimization-2026-09-10.md).

## Features

- Five-point face alignment from RGB images and landmarks.
- Batched embeddings and cosine similarity.
- Resident weights and reusable HRX graphs.
- Device inputs and outputs for composition with [SCRFD](https://github.com/zacharydenton/scrfd-hrx).

## Build and run

Requires Rust 1.91+, Linux x86-64, glibc 2.43+ and Radeon 8060S (`gfx1151`).
From the checkout:

```sh
cargo run --release -- --input aligned-crops.rgb --output embeddings.f32
```

Input is packed RGB bytes, 112×112×3 per aligned crop. Output is little-endian
float32, 512 values per crop. The pinned model and HRX runtime download on first
use. `--model w600k_r50.onnx` selects local weights; `--offline` requires cached
weights. Add `--benchmark 100` to measure warm inference.

## Rust API

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

Embeddings are unnormalized, matching InsightFace. Use `similarity` for cosine
similarity, or `embed(image, width, height, landmarks)` to align and embed faces
from an original image. [Alignment, resident API and model files](docs/inference.md).

## Development

```sh
cargo test
cargo clippy --all-targets -- -D warnings
cargo test --release -- --include-ignored --test-threads=1
```

Ignored model tests need the GPU and pinned weights; `ARCFACE_MODEL` selects a
local checkpoint. [Validation](docs/inference.md#performance-and-tests) ·
[Changelog](CHANGELOG.md)

## License

[Apache-2.0](LICENSE). Model weights have separate terms;
see [third-party notices](THIRD_PARTY_NOTICES.md).
