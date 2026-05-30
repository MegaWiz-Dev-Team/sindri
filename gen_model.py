"""Generate a tiny MNIST FFN, export to ONNX, and dump a reference input +
torch's own logits so we can verify tiny-infer matches torch exactly."""
import struct
import torch
import torch.nn as nn

torch.manual_seed(42)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)        # -> Flatten
        x = torch.relu(self.fc1(x))    # -> Gemm + Relu
        return self.fc2(x)             # -> Gemm


m = Net().eval()

# A fixed pseudo-image: 28x28 bytes 0..255 (raw, like the article's ubyte input).
g = torch.Generator().manual_seed(7)
img_u8 = (torch.rand(28 * 28, generator=g) * 255).to(torch.uint8)
with open("image.ubyte", "wb") as f:
    f.write(bytes(img_u8.tolist()))

# Normalize the SAME way tiny-infer's main.rs does (/255), shape [1,1,28,28].
x = (img_u8.float() / 255.0).reshape(1, 1, 28, 28)

with torch.no_grad():
    logits = m(x).squeeze(0)

# Legacy (TorchScript) exporter -> exports Linear as Gemm(transB=1), flatten as Flatten.
torch.onnx.export(
    m, x, "model.onnx",
    input_names=["input"], output_names=["logits"],
    opset_version=13, dynamo=False,
)

print("wrote model.onnx + image.ubyte (784 bytes)")
print("torch argmax :", int(logits.argmax()))
print("torch logits :", [round(v, 4) for v in logits.tolist()])
