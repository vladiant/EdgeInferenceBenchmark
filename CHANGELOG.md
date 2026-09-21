# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-22

Initial release: a self-contained CPU-only benchmark suite comparing a classical
OpenCV pipeline against a lightweight CNN on MNIST digit classification.

### Added

- **Classical classifier** — moment-based deskew → HOG features → `cv::ml::SVM`,
  behind the polymorphic `IClassifier` interface.
- **CNN classifier** — LeNet-style ONNX model run through OpenCV's `cv::dnn`
  module on the CPU backend (`DNN_TARGET_CPU`), behind the same `IClassifier`
  interface.
- **Benchmark harness** — sweeps each classifier across two execution targets
  (`single-thread` and `all-cores`) via an RAII `ThreadScope`, with discarded
  warmup iterations and per-image timing of only the `predict()` call.
- **Metrics** — p50/p95/mean/min/max latency, throughput (items/s), and
  top-1 accuracy per method × target, using nearest-rank percentiles.
- **Reporters** — console table plus optional CSV and JSON output, each with a
  full run-metadata header (OS, CPU, compiler, OpenCV version,
  warmup/iterations, timestamp).
- **MNIST IDX loader** — big-endian IDX3/IDX1 parsing with magic-number and
  dimension validation, failing fast on malformed input.
- **Setup scripts** — `prepare_data.py` (MNIST download / synthetic fallback),
  `eib_train_svm` (C++ SVM trainer sharing the exact inference HOG config), and
  `export_lenet.py` (trains and exports the CNN ONNX, failing loudly rather than
  emitting an untrained model).
- **Unit tests** — 18 GoogleTest cases covering IDX parsing, metrics math, the
  classifier seam (via a stub), and reporter formatting; runnable without data
  or model artifacts.
- **CI/CD** — mirrored GitHub Actions and GitLab CI pipelines that build all
  targets and run the unit-test gate on Ubuntu.
- **Packaging** — CPack (TGZ) bundling the app binary, `LICENSE`, and `docs/`;
  gitignored data/model artifacts are intentionally excluded.
- **Documentation** — requirements (SRS), design (DDS), QA report, release
  process, and this changelog.

[0.1.0]: https://github.com/vladiant/EdgeInferenceBenchmark/releases/tag/v0.1.0
