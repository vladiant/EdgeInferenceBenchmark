#!/usr/bin/env python3
"""Export a LeNet-style MNIST CNN to ONNX for EdgeInferenceBenchmark.

This SETUP HELPER lives outside the benchmarked C++ code path (design gap G-1).
Model training/export is out of scope for the benchmarked code; this script
produces an artifact the benchmark can consume via OpenCV's cv::dnn.

It requires PyTorch + ONNX. When they are unavailable it prints clear
instructions and exits non-zero WITHOUT failing the build: the classical path
and all unit tests build and pass without any ONNX model present, and the
benchmark app degrades gracefully (it skips the CNN method with a warning).

Usage:
    python3 scripts/export_lenet.py [--out models/mnist_lenet.onnx] [--epochs 1]
"""

import argparse
import os
import sys

INSTRUCTIONS = """\
Could not export an ONNX model because PyTorch/ONNX are not installed.

To produce a real LeNet-style MNIST model, in a Python environment run:

    pip install torch torchvision onnx
    python3 scripts/export_lenet.py --out models/mnist_lenet.onnx --epochs 1

Alternatively, drop any MNIST classifier ONNX model that accepts a
1x1x28x28 float input and emits 10 logits at models/mnist_lenet.onnx.

Without the model the benchmark still runs the classical path; the CNN method
is skipped with a warning (this is expected and non-fatal).
"""


def build_and_export(out_path, epochs):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), 2)
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    model = LeNet()

    try:
        from torchvision import datasets, transforms
        tf = transforms.Compose([transforms.ToTensor()])
        train = datasets.MNIST("data", train=True, download=True, transform=tf)
        loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for epoch in range(epochs):
            for i, (x, y) in enumerate(loader):
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(x), y)
                loss.backward()
                opt.step()
                if i % 200 == 0:
                    print(f"  epoch {epoch} step {i} loss {loss.item():.4f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  (training skipped: {exc}); exporting randomly-initialised model")

    model.eval()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    dummy = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"], output_names=["logits"],
        opset_version=11,
    )
    print(f"Wrote ONNX model to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="models/mnist_lenet.onnx")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
        print(INSTRUCTIONS)
        return 1

    build_and_export(args.out, args.epochs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
