"""Convert digits_cnn.pt (PyTorch state_dict) -> digits_cnn.onnx for ONNX Runtime Web.

Run in the folder that contains digits_cnn.pt:
    pip install torch onnx onnxruntime
    python export_to_onnx.py

IMPORTANT: the notebook saved make_cnn(bn=True), i.e. the network WITH BatchNorm2d layers
(that is why the .pt contains keys like 1.running_mean and 5.running_var). The architecture
below must match it exactly or load_state_dict will fail.
"""
import numpy as np
import torch
import torch.nn as nn

# 1) Rebuild the exact architecture that was trained and saved (bn=True)
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(128, 10),
)

# 2) Load your trained weights, then switch to eval mode (BatchNorm must use running stats)
model.load_state_dict(torch.load("digits_cnn.pt", map_location="cpu"))
model.eval()

# 3) Export. Input is 1x8x8, batch size is dynamic. Output = raw logits (softmax happens in JS).
dummy = torch.zeros(1, 1, 8, 8)
kwargs = dict(
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=13,
)
try:
    # torch >= 2.5: use the classic TorchScript exporter, it writes one self-contained file
    torch.onnx.export(model, dummy, "digits_cnn.onnx", dynamo=False, **kwargs)
except TypeError:
    torch.onnx.export(model, dummy, "digits_cnn.onnx", **kwargs)
print("saved digits_cnn.onnx")

# 4) Check that ONNX gives the same numbers as PyTorch
import onnxruntime as ort
sess = ort.InferenceSession("digits_cnn.onnx")
x = np.random.rand(5, 1, 8, 8).astype(np.float32)
with torch.no_grad():
    ref = model(torch.from_numpy(x)).numpy()
out = sess.run(None, {"input": x})[0]
print("max difference PyTorch vs ONNX:", float(np.abs(ref - out).max()))
assert np.allclose(ref, out, atol=1e-4), "ONNX output does not match PyTorch!"
print("OK: ONNX matches PyTorch")
