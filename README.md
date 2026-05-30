# Sindri 🔨

> Named after the Norse dwarven master-smith who forged the gods' greatest
> treasures (Mjölnir, Gungnir, Skíðblaðnir) — here, forging an inference engine
> from raw parts.

A **from-scratch ONNX inference engine in Rust**, built as a learning project —
the Rust + Apple Silicon port of Michal Pitr's C++ "Build Your Own Inference
Engine" article. Loads an ONNX model, builds a graph, topo-sorts it, and runs
inference on CPU with hand-written operators.

> Scope on purpose: dense MNIST FFN, 4 ops (Gemm, Relu, Add, Flatten), CPU.
> This is for *understanding how engines work* — not for competing with MLX on
> LLM serving (that's a different, bandwidth-bound problem).

## Layout

```
proto/onnx.proto      vendored official ONNX schema (proto2)
build.rs              prost-build → generates ONNX structs at build time
src/
  tensor.rs           Tensor = Vec<f32> + shape (row-major)
  graph.rs            Node + Graph (index-based arena) + topo_sort  ← Rust idiom
  ops.rs              gemm / relu / add / flatten  (naive, with tests)
  onnx_loader.rs      decode protobuf → weights + Graph
  infer.rs            walk topo order, run ops, thread tensors
  main.rs             CLI
```

## Build & test

```bash
cargo build      # generates ONNX structs via protoc, compiles engine
cargo test       # 6 tests: ops + topological sort
```

## Run

```bash
cargo run -- model.onnx                 # load + print graph summary
cargo run -- model.onnx image.ubyte     # also run a 28x28 MNIST image
```

## Milestones

- [x] **1. Load** — parse ONNX protobuf, extract weights → `Tensor`
- [x] **2. Graph + topo sort** — index arena, DFS post-order
- [x] **3. Inference (correctness)** — 4 naive ops, predict the digit
      (output verified to match PyTorch to ~5 decimal places)
- [x] **4. Fast CPU GEMM** — `gemm` crate (SIMD/NEON + threads). **22× over
      naive** at 1024³ on M4 Pro (~700 GFLOP/s). See `examples/bench_gemm.rs`.
- [x] **5. GPU via Metal** — hand-written tiled MSL kernel (`src/metal_gemm.rs`),
      `storageModeShared` (unified memory = no host↔device copy). A *naive* GPU
      kernel: beats naive CPU but loses to the CPU `gemm` crate, and is slower
      than naive CPU at 128³ (dispatch overhead).
- [x] **6. GPU via MPS** — Apple's tuned `MPSMatrixMultiplication`
      (`src/metal_mps.rs`, via raw objc message sends). The only backend that
      decisively beats the CPU `gemm` crate (3.7× at 2048³) — but the *slowest*
      of all at 128³ (per-call setup overhead). Tuned ≠ universally fast: it's
      tuned for large matrices.
- [ ] **7. Graph optimizations** — operator fusion, parallel branches.

### Benchmark (M4 Pro, GFLOP/s)

```
size    naive   gemm crate (CPU)   Metal (hand kernel)   MPS (tuned)
────────────────────────────────────────────────────────────────────
 128³     37          71                  14               4   ← MPS slowest!
 512³     32         464                 233             299
1024³     32         665                 470            1096   ← MPS 1.6× gemm
2048³     29         602                 689            2225   ← MPS 3.7× gemm
```

The lesson in one table: a hand kernel (even on the GPU) loses to a tuned CPU
library; only a tuned GPU library (MPS) wins — and only at the size it's tuned
for. Small matrices (≈ a single LLM decode step) are overhead-bound, where the
fancy backends lose to plain CPU code.

Run: `cargo run --release --example bench_gemm`

## Getting a test model (to reach milestone 3)

Train a tiny MNIST FFN in PyTorch and export to ONNX:

```python
# pip install torch torchvision
import torch, torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = torch.flatten(x, 1)       # Flatten
        x = torch.relu(self.fc1(x))   # Gemm + Relu
        return self.fc2(x)            # Gemm

m = Net().eval()   # (train it first for real accuracy; random works to test the engine)
dummy = torch.randn(1, 1, 28, 28)
torch.onnx.export(m, dummy, "model.onnx", input_names=["input"], output_names=["logits"])
```

Then `cargo run -- model.onnx` and check the printed topo order is
`Flatten Gemm Relu Gemm`. Feed a 784-byte image to get a prediction.

## Notes

- Input normalization in `main.rs` divides by 255 — adjust to match how your
  model was trained.
- `protoc` must be on PATH (Homebrew: `brew install protobuf`).
