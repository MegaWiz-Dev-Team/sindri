//! Sindri — a from-scratch ONNX inference engine (library root).
//! Exposes the engine modules so both the binary (src/main.rs) and
//! examples/benches can use them.

pub mod graph;
pub mod infer;
pub mod onnx_loader;
pub mod ops;
pub mod tensor;

// Milestone 5: GPU GEMM via Metal (Apple Silicon / macOS only).
#[cfg(target_os = "macos")]
pub mod metal_gemm;

// prost-generated ONNX structs (build.rs → $OUT_DIR/onnx.rs, package `onnx`).
pub mod onnx_proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}
