#!/usr/bin/env python3
"""Obtain or synthesize MNIST IDX data for EdgeInferenceBenchmark.

This script is a SETUP HELPER and lives outside the benchmarked C++ code path.
It tries to download the standard MNIST IDX files into ``data/``. If the
network is unavailable it generates a tiny SYNTHETIC IDX sample instead, so the
suite can still be built and exercised end-to-end (with meaningless accuracy).

Only the Python standard library is used (no numpy/torch/cv2 required).

Usage:
    python3 scripts/prepare_data.py [--out data] [--synthetic] [--count N]
"""

import argparse
import gzip
import os
import struct
import sys
import urllib.request

# Mirror hosting the classic MNIST IDX files (redistributable, see data/README).
BASE_URLS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]

FILES = {
    "train-images-idx3-ubyte": "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte": "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte": "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte": "t10k-labels-idx1-ubyte.gz",
}


def download_one(out_dir, name, gz_name):
    dest = os.path.join(out_dir, name)
    if os.path.exists(dest):
        print(f"  exists: {dest}")
        return True
    for base in BASE_URLS:
        url = base + gz_name
        try:
            print(f"  downloading {url}")
            with urllib.request.urlopen(url, timeout=20) as resp:
                blob = resp.read()
            data = gzip.decompress(blob)
            with open(dest, "wb") as f:
                f.write(data)
            print(f"  wrote {dest} ({len(data)} bytes)")
            return True
        except Exception as exc:  # noqa: BLE001 - best-effort with fallback
            print(f"  failed ({exc})")
    return False


def write_idx_images(path, images):
    rows, cols = 28, 28
    with open(path, "wb") as f:
        f.write(struct.pack(">IIII", 0x00000803, len(images), rows, cols))
        for img in images:
            f.write(bytes(img))


def write_idx_labels(path, labels):
    with open(path, "wb") as f:
        f.write(struct.pack(">II", 0x00000801, len(labels)))
        f.write(bytes(labels))


def synth_digit(label):
    """Produce a crude but deterministic 28x28 pattern for a digit 0-9."""
    px = [0] * (28 * 28)
    # Draw a filled rectangle whose size/position depends on the label so that
    # different labels are at least visually distinct (accuracy is irrelevant).
    top = 4 + (label % 5) * 2
    left = 4 + (label % 3) * 3
    for y in range(top, min(28, top + 12 + label)):
        for x in range(left, min(28, left + 8 + (label % 4) * 3)):
            px[y * 28 + x] = 255
    return px


def generate_synthetic(out_dir, count):
    images = [synth_digit(i % 10) for i in range(count)]
    labels = [i % 10 for i in range(count)]
    img_path = os.path.join(out_dir, "t10k-images-idx3-ubyte")
    lbl_path = os.path.join(out_dir, "t10k-labels-idx1-ubyte")
    write_idx_images(img_path, images)
    write_idx_labels(lbl_path, labels)
    # Provide a small "train" split too so the SVM trainer has data.
    train_img = [synth_digit(i % 10) for i in range(count * 4)]
    train_lbl = [i % 10 for i in range(count * 4)]
    write_idx_images(os.path.join(out_dir, "train-images-idx3-ubyte"), train_img)
    write_idx_labels(os.path.join(out_dir, "train-labels-idx1-ubyte"), train_lbl)
    print(f"  wrote synthetic IDX data ({count} test / {count * 4} train) to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data", help="output directory")
    parser.add_argument("--synthetic", action="store_true",
                        help="force synthetic data (skip download)")
    parser.add_argument("--count", type=int, default=200,
                        help="synthetic test-sample count")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if not args.synthetic:
        print("Attempting MNIST download...")
        ok = all(download_one(args.out, name, gz) for name, gz in FILES.items())
        if ok:
            print("MNIST data ready in", args.out)
            return 0
        print("Download unavailable; falling back to synthetic data.")

    print("Generating synthetic IDX data...")
    generate_synthetic(args.out, args.count)
    print("NOTE: synthetic data is for pipeline testing only; accuracy is not "
          "meaningful. Re-run without --synthetic when network is available.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
