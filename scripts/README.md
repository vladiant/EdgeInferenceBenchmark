# Setup scripts (outside the benchmarked code path)

These helpers prepare **data** and **models** for the benchmark. They are
intentionally kept **outside** the benchmarked library (`eib_core`) and app
(`edge_inference_benchmark`) so that model training / data provisioning is not
part of the measured code (SRS OOS-1; design gap G-1 resolution).

Nothing here is required to **build** the project or run the **unit tests**.
They are required only to obtain artifacts for an end-to-end benchmark run.

## 1. Data — `prepare_data.py`

Downloads the standard MNIST IDX files into `data/`, or generates a tiny
**synthetic** IDX sample when the network is unavailable (accuracy is then not
meaningful — it only exercises the pipeline). Pure Python standard library.

```bash
python3 scripts/prepare_data.py            # download, fallback to synthetic
python3 scripts/prepare_data.py --synthetic --count 200   # force synthetic
```

Produces (in `data/`):
`t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`,
`train-images-idx3-ubyte`, `train-labels-idx1-ubyte`.

## 2. Classical model — `eib_train_svm` (C++ tool)

The classical SVM artifact is produced by the `eib_train_svm` executable
(built when `-DEIB_BUILD_TOOLS=ON`, the default). It links `eib_core` only to
reuse the **exact** HOG configuration (`eib::compute_hog`) so the trained model
stays compatible with `ClassicalClassifier` (design gap G-2).

```bash
./build/eib_train_svm \
    --images data/train-images-idx3-ubyte \
    --labels data/train-labels-idx1-ubyte \
    --out    models/mnist_svm.xml \
    --max    5000
```

## 3. CNN model — `export_lenet.py`

**Trains** a LeNet-style CNN on the MNIST training split (read directly from the
IDX files in `data/`) and only then exports `models/mnist_lenet.onnx`. It prints
the achieved train/test accuracy for provenance and reaches ~98% test accuracy
in a few CPU epochs.

There is **no** silent random-weights fallback: if `torch`/`onnx` are missing,
the MNIST data is absent/synthetic, or the trained test accuracy falls below
`--min-accuracy` (default 0.97), the script **fails loudly** (non-zero exit)
with an actionable message instead of emitting a useless model. The benchmark
app still degrades gracefully when the ONNX model is simply absent (the CNN
method is skipped with a warning).

```bash
pip install torch onnx
python3 scripts/prepare_data.py            # obtain the real MNIST IDX files first
python3 scripts/export_lenet.py --out models/mnist_lenet.onnx --epochs 3
```

## End-to-end

```bash
python3 scripts/prepare_data.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/eib_train_svm --images data/train-images-idx3-ubyte \
                      --labels data/train-labels-idx1-ubyte \
                      --out models/mnist_svm.xml --max 5000
python3 scripts/export_lenet.py   # optional; enables the CNN method
./build/edge_inference_benchmark \
    --images data/t10k-images-idx3-ubyte \
    --labels data/t10k-labels-idx1-ubyte \
    --svm    models/mnist_svm.xml \
    --onnx   models/mnist_lenet.onnx \
    --csv out/results.csv --json out/results.json
```
