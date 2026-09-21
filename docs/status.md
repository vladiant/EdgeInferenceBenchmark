# Project Status

**Project:** EdgeInferenceBenchmark
**Current version:** 0.1.0
**Last updated:** 2026-09-22
**Overall state:** Released — docs complete, ready for version tag.

## Stages completed

| Stage | Status | Artifact |
| --- | --- | --- |
| Requirement Analysis | Complete | [docs/requirements/SRS.md](requirements/SRS.md) |
| Design | Complete | [docs/design/DESIGN.md](design/DESIGN.md) |
| Implementation | Complete | `src/`, `apps/`, `include/eib/` |
| QA / Verification | **PASSED** (18/18 tests, all 11 acceptance criteria) | [docs/qa/QA_REPORT.md](qa/QA_REPORT.md) |
| Release Engineering | Complete (CI green, CPack packaging, VERSION 0.1.0) | [docs/release/RELEASE.md](release/RELEASE.md) |
| Documentation | Complete (README, CHANGELOG, this status) | [README.md](../README.md), [CHANGELOG.md](../CHANGELOG.md) |

## Summary

The suite compares a classical OpenCV pipeline (deskew + HOG + SVM) against a
lightweight LeNet-style CNN (ONNX via `cv::dnn`) on MNIST digit classification,
CPU-only. QA verified a meaningful classical-vs-CNN comparison: classical 98.65%
accuracy at ~3.0k items/s, CNN 97.50% accuracy at ~7.9k–12.4k items/s across the
single-thread and all-cores targets.

## Known limitations / future work

- **Single task/dataset.** Only MNIST digit classification is covered. Adding
  further tasks/datasets would broaden the trade-off study.
- **Single inference runtime for the CNN.** The CNN uses OpenCV `cv::dnn`; an
  ONNX Runtime backend option was noted as a viable alternative but intentionally
  not adopted, to keep the dependency surface minimal.
- **Two execution targets.** Only `single-thread` and `all-cores` are defined;
  additional hardware/thread targets could be added.
- **DEF-1 (untrained CNN ONNX → random-level accuracy) — RESOLVED.** Fixed at
  its root cause in `scripts/export_lenet.py` (train-first, fail-loud, no silent
  random-weights fallback) and independently re-verified by QA. No open defects
  remain.

## Post-release notes

- **CI fix (released in 0.1.1).** GitLab pipeline image now installs `git` and
  `ca-certificates` so GoogleTest fetches via CMake FetchContent; configure-time
  build failure resolved. No code/behavior change. See [CHANGELOG.md](../CHANGELOG.md).

## Next step

Released: `v0.1.1` (patch) — GitLab CI reliability fix. Tags `v0.1.0` and
`v0.1.1` are published locally via the `semver-version-publish` workflow;
VERSION reads 0.1.1.
