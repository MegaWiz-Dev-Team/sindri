//! Generate Rust structs from the vendored ONNX protobuf schema at build time.
//!
//! Uses the system `protoc` (found on PATH — you have it at /opt/homebrew/bin/protoc).
//! Output lands in $OUT_DIR/onnx.rs (named after the proto `package onnx;`),
//! which `src/main.rs` pulls in via `include!`.

fn main() {
    println!("cargo:rerun-if-changed=proto/onnx.proto");
    prost_build::compile_protos(&["proto/onnx.proto"], &["proto"])
        .expect("failed to compile proto/onnx.proto — is protoc installed?");
}
