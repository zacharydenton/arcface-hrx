# Changelog

## Unreleased

- Use HRX 0.5's shared Hugging Face resolver, checked ONNX loader, scratch
  planner, compiler selection, and top-level `hrx::model` API.
- Accept packed RGB images and aligned crops in `embed`, `embeddings`, `alignment::crop`, and the CLI. BGR callers must swap red and blue before calling.
- Normalize and pad each input pixel with one vector store in the GPU preprocessing kernel.
- Return errors instead of panicking when malformed ONNX models pass non-spatial tensors to Transpose, pooling, or Resize.

- Fetch pinned pretrained weights through the shared Hugging Face cache by default; retain local-file and offline loading.
- Require Rust 1.91 for the HF Hub 1.0 dependency stack.

- Use resident coherent inputs and terminal outputs to remove GPU transfer submissions.
- Record graph dependencies from buffer hazards so independent branches can overlap.

## 0.1.0 — Rust migration

- Renamed `arcface-loom` to `arcface-hrx`.
- Replaced the Python package and C ABI with a Rust library and CLI using HRX 0.4.0.
- Load original model files directly; weight conversion and graph scheduling run in Rust.
- Cache resident GPU graphs by batch size and reuse activation and readback allocations.
- Removed standalone HIP hosts, build scripts, Python tooling and obsolete experiments.

This replaces the former APIs; there are no deprecated compatibility wrappers.
