# Models directory

This directory holds the model artifacts consumed by the benchmark. They are
**not committed** (see `.gitignore`); produce them locally with the setup
helpers in `scripts/` (design gap G-1: training is outside the benchmarked
code path).

| File | Produced by | Consumed by |
| --- | --- | --- |
| `mnist_svm.xml` | `eib_train_svm` (C++ tool) | `ClassicalClassifier` |
| `mnist_lenet.onnx` | `scripts/export_lenet.py` | `CnnClassifier` (`cv::dnn`) |

## Classical SVM (`mnist_svm.xml`)

An OpenCV `cv::ml::SVM` (RBF kernel, `C_SVC`) trained on **HOG features** of
28x28 MNIST images. The HOG configuration is fixed by `eib::make_mnist_hog()`
and **must match** between training and inference (design gap G-2):

- window `28x28`, block `14x14`, block stride `7x7`, cell `7x7`, 9 bins
- moment-based deskew applied before HOG

Produce it with:

```bash
./build/eib_train_svm --images data/train-images-idx3-ubyte \
                      --labels data/train-labels-idx1-ubyte \
                      --out models/mnist_svm.xml --max 5000
```

## CNN ONNX (`mnist_lenet.onnx`)

A LeNet-style CNN exported to ONNX (opset 11), input `1x1x28x28` float, output
10 logits. Produced by `scripts/export_lenet.py` (requires `torch`/`onnx`).
Keep the graph to standard conv/pool/gemm/relu ops for OpenCV importer
coverage (design gap G-3). If the model is absent, the benchmark skips the CNN
method gracefully.

## Licensing (OQ-3, NFR-6)

Both artifacts are generated locally by the setup helpers from MNIST data;
no third-party pre-trained weights are vendored. Any externally sourced model
placed here must carry a license compatible with the repository LICENSE.
