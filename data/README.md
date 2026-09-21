# Data directory

This directory holds the MNIST **IDX** files consumed by the benchmark. The
files are **not committed** (see `.gitignore`); provision them locally with the
setup helper:

```bash
python3 scripts/prepare_data.py
```

Expected files:

| File | Purpose |
| --- | --- |
| `t10k-images-idx3-ubyte` | test images (evaluated set) |
| `t10k-labels-idx1-ubyte` | test labels |
| `train-images-idx3-ubyte` | training images (for `eib_train_svm`) |
| `train-labels-idx1-ubyte` | training labels |

## Provenance & licensing (OQ-4, NFR-6)

MNIST is the classic handwritten-digit dataset by LeCun, Cortes, and Burges,
widely redistributed for research/education. The download helper fetches the
standard IDX files from a public mirror. If the network is unavailable, the
helper generates a tiny **synthetic** IDX sample so the pipeline still runs
(accuracy is then not meaningful).

The IDX binary format:
- Images: magic `0x00000803`, big-endian header `[count, rows, cols]`, then
  `count * rows * cols` unsigned bytes.
- Labels: magic `0x00000801`, big-endian header `[count]`, then `count` bytes.
