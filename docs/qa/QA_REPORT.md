# EdgeInferenceBenchmark — QA Report

Stage: Testing / Verification
Source of truth: [docs/requirements/SRS.md](../requirements/SRS.md),
[docs/design/DESIGN.md](../design/DESIGN.md)
Date: 2026-09-22
Verified by: QA Engineer agent
Status: **PASSED** (re-verified after DEF-1 fix)

> **Re-verification note (2026-09-22):** The single blocking defect DEF-1
> (untrained CNN ONNX → random-level accuracy) has been fixed by the developer
> and independently re-verified by QA. `scripts/export_lenet.py` now trains a
> LeNet on the real MNIST IDX split before export and fails loudly instead of
> ever emitting a random model. The CNN top-1 accuracy is now **0.9750** on
> 2000 images (was 0.0950), comparable to the classical 0.9865. See
> [§8 DEF-1 — RESOLVED](#def-1--vendored-cnn-onnx-model-is-untrained-random-level-accuracy--severity-major--resolved)
> and [§9 Overall Verdict](#9-overall-verdict). Overall verdict is now
> **QA PASSED**.

---

## 1. Environment Summary

| Item | Value |
| --- | --- |
| OS | Linux |
| CPU | Intel(R) Core(TM) i7-6700 @ 3.40GHz (4C/8T) |
| Compiler | GCC 13.3.0 |
| CMake | 3.28.3 |
| OpenCV | 4.6.0 (system, components: core imgproc ml dnn objdetect) |
| Build type | Release |
| Dataset | Real full MNIST (t10k = 10000 imgs, train = 60000 imgs), IDX format |
| Models | `models/mnist_svm.xml` (HOG+SVM, trained), `models/mnist_lenet.onnx` (**trained**, test_acc 0.9822 — DEF-1 resolved) |
| Python (setup) | torch 2.14.0+cpu, onnx 1.23.0 (used only by `export_lenet.py`) |

> Note: The project ships a plain CMake + `find_package(OpenCV)` build (no
> `conanfile.py` / Conan profiles present), so the SRS/design-documented
> `cmake -S . -B build` flow was used, exactly as the developer handoff and the
> task specified.

---

## 2. Build Result — PASS

Command (per DESIGN §9 and task):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

- Clean configure; OpenCV 4.6.0 found with all required components.
- Full build of `eib_core`, `edge_inference_benchmark`, `eib_train_svm`,
  `eib_tests` completed.
- **Compiler warnings/errors: 0** (`-Wall -Wextra -Wpedantic` via
  `cmake/EibHelpers.cmake`).
- CPU-only: no CUDA/OpenCL required; CNN pinned to `DNN_TARGET_CPU` (NFR-1).

## 3. Unit Test Result — PASS (18/18)

`ctest --test-dir build --output-on-failure` → **100% passed, 0 failed, 0
skipped, 0 flaky** (total 0.18 s).

Coverage vs. DESIGN §7 testing strategy — all four mandated seams present and
genuine (not hollow):

| Design seam | Test file | Tests | Status |
| --- | --- | --- | --- |
| IDX parsing / validation (FR-1, FR-14) | `test_mnist_dataset.cpp` | 6 (valid load, max-items cap, missing file, bad magic, count mismatch, wrong dims) | Covered |
| percentile / metrics math (MET-4..6) | `test_metrics.cpp` | 6 (nearest-rank p50/p95, empty, single, stats+throughput, accuracy override, empty-finalize safety) | Covered |
| classifier seam via stub (FR-8/9, MET-1/2) | `test_runner_with_stub.cpp` | 3 (per-target results, warmup exclusion + call count, predict_all cycling) | Covered |
| reporter formatting (FR-10/11/13) | `test_reporter.cpp` | 3 (CSV header+metadata+rows, JSON metadata+results, console table) | Covered |

Minor coverage gap (non-blocking): the CLI arg parser in `apps/main.cpp`
(exit-code / bad-arg handling) has no unit test because it lives in the app
target, not `eib_core`. It was instead verified by the end-to-end robustness
checks in §6. No test was added to avoid refactoring app code for testability.

## 4. End-to-End Benchmark — Deliverable Captured

Command:

```bash
./build/edge_inference_benchmark \
  --images data/t10k-images-idx3-ubyte --labels data/t10k-labels-idx1-ubyte \
  --svm models/mnist_svm.xml --onnx models/mnist_lenet.onnx \
  --max-images 200 --warmup 20 --iterations 200 \
  --csv out/qa_results.csv --json out/qa_results.json
```

Provisioning status (re-verified 2026-09-22):
- MNIST data: **available** (real full IDX files present in `data/`).
- SVM model: **available and trained** (validated — see accuracy below).
- ONNX model: **available and trained** — `models/mnist_lenet.onnx` produced by
  the fixed `export_lenet.py` (final test_acc 0.9822). DEF-1 RESOLVED.

### Console results table (both methods × both targets) — re-verification run

Command actually run (2000 images, warmup 20, iterations 200):

```
method      target        threads iters  p50(ms)   p95(ms)   mean(ms)  thrpt(ips)  accuracy
classical   single-thread 1       200    0.2932    0.5021    0.3325    3007.37     0.9865
classical   all-cores     8       200    0.3031    0.5139    0.3358    2978.15     0.9865
cnn         single-thread 1       200    0.1164    0.1790    0.1268    7883.85     0.9750
cnn         all-cores     8       200    0.0802    0.0947    0.0807    12384.33    0.9750
```

The CNN top-1 is now **0.9750** (realistic, ~97–99% band) versus the prior
**0.0950** random-level figure — the classical-vs-CNN comparison is now
meaningful. Independent cross-check: a freshly retrained model exported to a
temp path (`/tmp/qa_retrain.onnx`, test_acc 0.9822) benchmarked at the same
**0.9750** CNN top-1, confirming the provisioned artifact is genuinely trained
and reproducible, not a one-off file.

Both CSV (`out/qa_reverify.csv`) and JSON (`out/qa_reverify.json`) were written
and contain identical metrics plus the full metadata header (OS, CPU, compiler,
OpenCV version, timestamp, max_images/warmup/iterations) — see §7 AC-7/AC-8.

### Accuracy realism cross-check (independent QA check)

The classical HOG+SVM path scores a realistic **0.9865** over 2000 images,
confirming the classical path and data are genuine (a 100% figure only appears
on tiny 200-image samples).

The CNN now scores **0.9750** over 2000 images — in the expected ~97–99% band
for a few-epoch LeNet, and no longer the ~0.0950 random-level figure that
signalled the untrained model in the original pass (DEF-1, now RESOLVED).

### Fail-loud export behavior (DEF-1 fix) — PASS

`scripts/export_lenet.py` was run pointed at an empty data directory to force
the training data to be unavailable:

```
$ python3 scripts/export_lenet.py --data /tmp/empty_data --out /tmp/qa_should_not_exist.onnx

ERROR: MNIST file not found: /tmp/empty_data/train-images-idx3-ubyte

Prepare the dataset first with:

    python3 scripts/prepare_data.py

EXIT_CODE=1
--- model emitted? ---
ls: cannot access '/tmp/qa_should_not_exist.onnx': No such file or directory
```

The script exits **non-zero (1)** with an actionable message and emits **no**
ONNX file — the silent random-weights fallback is gone. A successful run instead
trains and reports provenance:

```
Loaded 60000 train / 10000 test samples from 'data'
epoch 0: ... test_acc=0.9713
epoch 1: ... test_acc=0.9778
epoch 2: ... test_acc=0.9822
Final accuracy - train: 0.9820  test: 0.9822
Wrote TRAINED ONNX model to /tmp/qa_retrain.onnx (test_acc=0.9822)
```

The source additionally guards against weak models (`final_test <
--min-accuracy`, default 0.97) and synthetic/too-small data, each raising the
same fail-loud `SetupError` — so an untrained or under-trained artifact can no
longer be produced.

### Classical-only path across both targets (CNN artifact absent)

With a missing ONNX path the app still delivers the mandated classical table
across both targets and exits 0 (graceful degradation, SRS-compliant):

```
warning: skipping method 'cnn': CnnClassifier: failed to read ONNX model ...
classical   single-thread 1  ...  accuracy 1.0000
classical   all-cores     8  ...  accuracy 1.0000
exit=0
```

## 5. Determinism (AC-10, NFR-4) — PASS

Two identical classical runs (max-images 500) produced **identical accuracy
0.9880** both times. Predicted labels/accuracy are thread-invariant; only
timings vary, as designed.

## 6. Robustness Spot-Checks — PASS

| Scenario | Observed | Exit |
| --- | --- | --- |
| Missing data file | `error: IDX: cannot open file 'data/nope.idx'` | 1 |
| Missing SVM (classical only) | clear warning + `error: no classifier could be constructed...` | 1 |
| Bad CLI arg `--frobnicate` | `error: unknown argument '--frobnicate'` | 2 |
| Missing required `--images/--labels` | usage printed + `error: --images and --labels are required` | 2 |
| CNN ONNX missing, classical present | `warning: skipping method 'cnn': ...` then classical table runs | 0 |

No crashes or segfaults; all failures are clear, non-crashing, and non-zero
exit where required (FR-14, AC-9). OpenCV emits internal `[ERROR:...]` lines on
a missing model file, but the app still catches the exception and reports it
cleanly.

---

## 7. Acceptance Criteria Trace

| AC | Verdict | Justification |
| --- | --- | --- |
| AC-1 Build on Linux, no GPU dep | PASS | Release build, 0 warnings; CPU-only, `DNN_TARGET_CPU`. |
| AC-2 Both methods emit digit 0–9 + score per image | PASS | Both ran and produced valid `Prediction{label∈0..9, score}` per image; CNN accuracy is now realistic (0.9750), not random (DEF-1 resolved). |
| AC-3 Accuracy reported per method | PASS | Accuracy column present for classical and CNN in console/CSV/JSON. |
| AC-4 Table has p50/p95/mean, throughput, accuracy, iters per method×target | PASS | All fields present in console + CSV + JSON. |
| AC-5 ≥2 named targets, each row labeled | PASS | `single-thread` (threads=1) and `all-cores` (threads=8) labeled per row. |
| AC-6 Warmup excluded; warmup+iters reported | PASS | `iterations`=measured only (unit-test verified exclusion); warmup+iterations echoed in metadata/rows. |
| AC-7 Machine-readable CSV/JSON with same metrics | PASS | Both files written; metrics match the console table. |
| AC-8 Metadata: OS, CPU, compiler/build, OpenCV ver, per-target threads, timestamp | PASS | All present in console + CSV header + JSON `metadata`. |
| AC-9 Missing/malformed artifacts → clear error, non-zero exit | PASS | Verified for data + model + bad args (§6). |
| AC-10 Re-run → identical labels/accuracy | PASS | Identical accuracy across repeated runs (§5). |
| AC-11 Licenses compatible | PASS | Repo MIT; artifacts generated locally from redistributable MNIST; no third-party weights vendored (`models/README.md`). |

## 8. Defects

### DEF-1 — Vendored CNN ONNX model is untrained (random-level accuracy) — Severity: MAJOR — **RESOLVED**

- **Original symptom:** CNN accuracy was **0.0950 (~1/10, i.e. random
  guessing)** across both targets and multiple sample sizes, while classical
  was a realistic ~98.6%.
- **Original root cause (in source, not the benchmark code):**
  [scripts/export_lenet.py](../../scripts/export_lenet.py) silently fell back to
  exporting a randomly-initialised model when Torch training failed or MNIST
  could not be downloaded — it caught the exception, printed a low-visibility
  note, and still wrote an untrained `.onnx`.
- **Fix applied by developer (verified):**
  1. `export_lenet.py` now **trains** a LeNet on the real MNIST IDX training
     split (read via `prepare_data.py`, no torchvision downloader) and prints
     per-epoch train/test accuracy for provenance before exporting.
  2. The silent random-weights fallback is **removed**. Missing torch/onnx,
     missing/synthetic data, or trained test accuracy below `--min-accuracy`
     (default 0.97) each raise a fail-loud `SetupError` → non-zero exit, and
     **no** ONNX file is written.
  3. A genuinely trained `models/mnist_lenet.onnx` (test_acc 0.9822) is
     provisioned.
- **QA re-verification evidence (2026-09-22):**
  - Benchmark CNN top-1 = **0.9750** on 2000 images across both targets (was
     0.0950), comparable to classical 0.9865 — the classical-vs-CNN comparison
     is now meaningful (§4).
  - Fail-loud confirmed: export against an empty data dir exits **1** with an
     actionable message and emits **no** model (§4 fail-loud subsection).
  - Reproducibility/independence: a freshly retrained temp model (test_acc
     0.9822) benchmarked at the same 0.9750 CNN top-1, confirming the artifact
     is genuinely trained, not a one-off file.
- **Residual note (non-defect):** the export prints a harmless
  `DeprecationWarning` about the legacy TorchScript ONNX exporter; `dynamo=False`
  is used intentionally to avoid an `onnxscript` dependency and keep the graph
  to OpenCV-importable ops (design gap G-3). No action required.

No other defects found. Build, tests, classical path, CNN path, metrics,
reporting, determinism, robustness, and licensing are all clean.

---

## 9. Overall Verdict

- **Build:** PASS (0 warnings)
- **Unit tests:** PASS (18/18)
- **Classical-path deliverable (latency/throughput/accuracy table across two
  targets, CSV+JSON):** PASS
- **CNN comparison deliverable:** PASS (CNN top-1 0.9750, DEF-1 resolved)
- **All 11 acceptance criteria:** PASS

The previously blocking DEF-1 (untrained CNN model → random-level accuracy) has
been fixed at its root cause in `scripts/export_lenet.py` (train-first,
fail-loud, no silent fallback) and a trained `mnist_lenet.onnx` is provisioned.
Re-running the end-to-end benchmark now yields a meaningful classical-vs-CNN
comparison (classical 0.9865 vs CNN 0.9750), satisfying the project's core value
proposition. Every line of the benchmarked C++ code, all 18 unit tests, the
classical and CNN paths, robustness, determinism, and licensing are green.

**QA PASSED — ready for the Release Engineer agent to deploy.**

- DEF-1 (MAJOR) → **RESOLVED and verified**; no open defects remain.
- No architecture or requirements changes were needed; the fix was confined to
  the setup helper outside the benchmarked code path.
