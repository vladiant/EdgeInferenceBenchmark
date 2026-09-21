#!/usr/bin/env python3
"""Train a LeNet-style MNIST CNN and export it to ONNX for EdgeInferenceBenchmark.

This SETUP HELPER lives outside the benchmarked C++ code path (design gap G-1).
Model training/export is out of scope for the benchmarked code; this script
produces a *genuinely trained* artifact the benchmark consumes via OpenCV's
cv::dnn.

It reads the MNIST IDX files prepared by ``scripts/prepare_data.py`` (so it has
no dependency on torchvision's downloader), trains a small LeNet on the training
split for a few epochs, evaluates test accuracy, and only then exports ONNX.

There is intentionally NO silent fallback to a random-weight model. If the
required dependencies (torch, onnx) or the MNIST data are missing, or if the
trained model fails to reach ``--min-accuracy`` on the test split, the script
FAILS LOUDLY (non-zero exit) with an actionable message. A random-weight LeNet
scores ~10% (chance) and would make the CNN half of the benchmark meaningless,
so exporting one is treated as an error, never a fallback.

Usage:
    python3 scripts/export_lenet.py [--out models/mnist_lenet.onnx]
                                    [--data data] [--epochs 3]
                                    [--min-accuracy 0.97]

Requires:
    pip install torch onnx
    python3 scripts/prepare_data.py   # to obtain the real MNIST IDX files
"""

import argparse
import os
import struct
import sys


class SetupError(Exception):
    """Raised for actionable, user-facing setup failures (missing deps/data)."""


def _require_torch():
    try:
        import torch  # noqa: F401
        import onnx  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SetupError(
            "PyTorch and/or ONNX are not installed "
            f"({exc}).\n\nInstall them with:\n\n"
            "    pip install torch onnx\n"
        ) from exc


def _read_idx_images(path):
    """Return (count, rows, cols, raw_bytes) parsed from an IDX3 image file."""
    with open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 0x00000803:
            raise SetupError(f"{path}: bad IDX image magic 0x{magic:08x}")
        buf = f.read(count * rows * cols)
    if len(buf) != count * rows * cols:
        raise SetupError(f"{path}: truncated image data")
    return count, rows, cols, buf


def _read_idx_labels(path):
    with open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 0x00000801:
            raise SetupError(f"{path}: bad IDX label magic 0x{magic:08x}")
        buf = f.read(count)
    if len(buf) != count:
        raise SetupError(f"{path}: truncated label data")
    return count, buf


def _load_split(data_dir, img_name, lbl_name):
    import torch

    img_path = os.path.join(data_dir, img_name)
    lbl_path = os.path.join(data_dir, lbl_name)
    for p in (img_path, lbl_path):
        if not os.path.exists(p):
            raise SetupError(
                f"MNIST file not found: {p}\n\n"
                "Prepare the dataset first with:\n\n"
                "    python3 scripts/prepare_data.py\n"
            )

    n_img, rows, cols, img_buf = _read_idx_images(img_path)
    n_lbl, lbl_buf = _read_idx_labels(lbl_path)
    if n_img != n_lbl:
        raise SetupError(
            f"count mismatch: {img_path} has {n_img} images but "
            f"{lbl_path} has {n_lbl} labels"
        )
    if (rows, cols) != (28, 28):
        raise SetupError(f"{img_path}: expected 28x28 images, got {rows}x{cols}")

    images = torch.frombuffer(bytearray(img_buf), dtype=torch.uint8)
    images = images.view(n_img, 1, rows, cols).float().div_(255.0)
    labels = torch.frombuffer(bytearray(lbl_buf), dtype=torch.uint8).long()
    return images, labels


def _build_model():
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
            x = x.flatten(1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    return LeNet()


def _evaluate(model, images, labels, batch_size):
    import torch

    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, images.size(0), batch_size):
            xb = images[start:start + batch_size]
            yb = labels[start:start + batch_size]
            preds = model(xb).argmax(dim=1)
            correct += int((preds == yb).sum())
    return correct / images.size(0)


def train_and_export(args):
    import torch

    torch.manual_seed(args.seed)

    train_images, train_labels = _load_split(
        args.data, "train-images-idx3-ubyte", "train-labels-idx1-ubyte")
    test_images, test_labels = _load_split(
        args.data, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")

    print(f"Loaded {train_images.size(0)} train / {test_images.size(0)} test "
          f"samples from '{args.data}'")

    if train_images.size(0) < 1000 or test_images.size(0) < 500:
        raise SetupError(
            "MNIST data looks too small / synthetic to train a real model "
            f"({train_images.size(0)} train, {test_images.size(0)} test). "
            "Run 'python3 scripts/prepare_data.py' without --synthetic to fetch "
            "the full dataset before exporting a CNN."
        )

    model = _build_model()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    n = train_images.size(0)

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        running = 0.0
        step = 0
        for step, start in enumerate(range(0, n, args.batch_size)):
            idx = perm[start:start + args.batch_size]
            xb, yb = train_images[idx], train_labels[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item()
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step} loss {loss.item():.4f}")
        train_acc = _evaluate(model, train_images, train_labels, args.batch_size)
        test_acc = _evaluate(model, test_images, test_labels, args.batch_size)
        print(f"epoch {epoch}: avg_loss={running / (step + 1):.4f} "
              f"train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

    final_train = _evaluate(model, train_images, train_labels, args.batch_size)
    final_test = _evaluate(model, test_images, test_labels, args.batch_size)
    print(f"\nFinal accuracy - train: {final_train:.4f}  test: {final_test:.4f}")

    if final_test < args.min_accuracy:
        raise SetupError(
            f"trained test accuracy {final_test:.4f} is below the required "
            f"minimum {args.min_accuracy:.4f}. Refusing to export a weak model. "
            "Increase --epochs or check the training data, then re-run."
        )

    model.eval()
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    dummy = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["input"], output_names=["logits"],
        opset_version=11,
        dynamo=False,  # legacy exporter: needs only torch + onnx (no onnxscript)
    )
    print(f"Wrote TRAINED ONNX model to {args.out} "
          f"(test_acc={final_test:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="models/mnist_lenet.onnx")
    parser.add_argument("--data", default="data",
                        help="directory holding the MNIST IDX files")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-accuracy", type=float, default=0.97,
                        help="fail if trained test accuracy is below this")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        _require_torch()
        train_and_export(args)
    except SetupError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
