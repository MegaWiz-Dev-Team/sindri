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
- [ ] **4. Fast CPU GEMM** — swap naive loop for the `gemm` crate or Accelerate
      BLAS (`cblas_sgemm`). Expect tens of × speedup. (deps already stubbed in
      `Cargo.toml`)
- [ ] **5. GPU via Metal** — MPS `MPSMatrixMultiplication`, then a hand-written
      MSL kernel. Use `MTLBuffer` `storageModeShared` (unified memory = no
      host↔device copy).
- [ ] **6. Graph optimizations** — operator fusion, parallel branches.

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
