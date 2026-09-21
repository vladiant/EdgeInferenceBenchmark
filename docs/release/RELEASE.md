# Release Process

This document describes how EdgeInferenceBenchmark is versioned, how CI gates a
change, how to cut a release, and how to reproduce the benchmark results table
locally. It covers **build/CI/packaging** only — no application logic.

## Versioning (SemVer)

The single source of truth for the version is the top-level [`VERSION`](../../VERSION)
file. It contains one line, `MAJOR.MINOR.PATCH` (currently **`0.1.0`**), and is
read by `CMakeLists.txt` into the CMake `project(... VERSION ...)` and reused as
the CPack package version.

Bump rules (Semantic Versioning 2.0.0):

- **MAJOR** — incompatible CLI/format/behavior changes.
- **MINOR** — backward-compatible functionality (new methods, flags, outputs).
- **PATCH** — backward-compatible fixes, docs, CI/packaging changes.

> The final version bump and the annotated git tag are performed by the PM via
> the `semver-version-publish` process. Do not hand-edit tags; edit `VERSION`
> and let the release process publish the matching tag.

## CI gates

Two mirrored pipelines run the same required gate on Ubuntu:

- GitHub Actions — [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)
- GitLab CI — [`.gitlab-ci.yml`](../../.gitlab-ci.yml)

Both perform, on push / merge request to `main`:

1. Install `build-essential cmake ninja-build libopencv-dev`.
2. Configure in `Release`.
3. Build all targets (`eib_core`, `edge_inference_benchmark`, `eib_train_svm`, `eib_tests`).
4. Run the **18 unit tests** via `ctest` — the required gate.
5. Build the TGZ package with CPack.

The unit-test gate requires **no** MNIST data and **no** ONNX/SVM model
artifacts, so it is fully self-contained. Data and models are gitignored and
provisioned locally (see below); CI never depends on committed artifacts.

## Packaging

Packaging uses CPack (TGZ generator). It bundles the app binary, `LICENSE`, and
`docs/` — it does **not** bundle the gitignored `data/` or `models/` artifacts.

Build a package locally from a configured build tree:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cmake --build build --target package     # or: (cd build && cpack)
```

The tarball is written to `build/EdgeInferenceBenchmark-<version>-<System>.tar.gz`.

## Cutting a release

1. Ensure QA has signed off and `main` is green (both CI pipelines pass).
2. Update [`VERSION`](../../VERSION) per the SemVer rules above (PM / release process).
3. Publish the annotated tag `vMAJOR.MINOR.PATCH` (PM via `semver-version-publish`).
4. CI builds on the tagged commit and produces the TGZ package artifact, which
   is attached to the GitHub Release / GitLab release.

## Reproducing the benchmark results table (manual, optional)

These steps are **not** part of the required CI gate — they need network access
and Python (`torch` CPU + `onnx`). They reproduce the end-to-end results
reported in [`docs/qa/QA_REPORT.md`](../qa/QA_REPORT.md).

```bash
# 1. Data: download the real MNIST IDX files into data/
python3 scripts/prepare_data.py

# 2. Build (Release)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# 3. Classical model (SVM over the exact HOG config used by the classifier)
./build/eib_train_svm \
    --images data/train-images-idx3-ubyte \
    --labels data/train-labels-idx1-ubyte \
    --out    models/mnist_svm.xml \
    --max    5000

# 4. CNN model (trains LeNet, fails loudly if test accuracy < threshold)
pip install torch onnx
python3 scripts/export_lenet.py --out models/mnist_lenet.onnx --epochs 3

# 5. Run the benchmark and emit CSV/JSON
./build/edge_inference_benchmark \
    --images data/t10k-images-idx3-ubyte \
    --labels data/t10k-labels-idx1-ubyte \
    --svm    models/mnist_svm.xml \
    --onnx   models/mnist_lenet.onnx \
    --csv out/results.csv --json out/results.json
```

The benchmark app degrades gracefully when a model is absent (the affected
method is skipped with a warning), so partial provisioning still runs.
