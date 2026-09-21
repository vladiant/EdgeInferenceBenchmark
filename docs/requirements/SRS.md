# EdgeInferenceBenchmark — Software Requirements Specification (SRS)

Status: Draft for design hand-off
Stage: Requirement Analysis
Scope: Small, self-contained, reproducible C++ benchmark suite

## 1. Overview

EdgeInferenceBenchmark is a small C++ portfolio project that benchmarks and
compares two implementations of the **same computer-vision task** on **CPU only**:

1. A **classical OpenCV pipeline** (hand-engineered features + a classical
   classifier), and
2. A **lightweight CNN** run through a CPU inference runtime.

The suite runs both methods over the same input dataset, verifies that their
outputs are comparable (same task, same label space), measures latency and
throughput across at least two hardware/execution targets, and emits a
results table (console + machine-readable file).

The goal is to demonstrate — for a resume/portfolio — clean C++, reproducible
benchmarking methodology, and a defensible "classical CV vs. small CNN"
trade-off study on commodity CPUs.

### 1.1 Chosen concrete task (decision)

**Handwritten digit classification on MNIST** (28×28 grayscale images,
10 classes: digits 0–9).

Justification:
- Both a classical OpenCV pipeline **and** a small CNN are natural, canonical,
  and directly comparable on this task — they share an identical input format
  (28×28 grayscale) and an identical output space (a single integer label
  0–9 plus a confidence/score).
- MNIST is tiny, freely redistributable, and CPU-friendly, keeping the suite
  "small," self-contained, and fast to run in CI.
- Accuracy is a meaningful secondary axis: it lets the benchmark report the
  classic latency-vs-accuracy trade-off, not just raw speed.
- It avoids GPU pressure, large models, and heavyweight datasets, matching the
  CPU-only, reproducible-build intent.

### 1.2 Classical CV approach (sketch — no code)

Deterministic, hand-engineered OpenCV pipeline per image:
- Preprocessing: convert to single-channel grayscale (already grayscale for
  MNIST), optional binarization/normalization, and **deskew** using image
  moments to reduce slant variance.
- Feature extraction: **HOG (Histogram of Oriented Gradients)** descriptor over
  the 28×28 image (via OpenCV's `cv::HOGDescriptor`), producing a fixed-length
  feature vector.
- Classification: a classical classifier from the OpenCV `ml` module —
  **SVM (`cv::ml::SVM`)** as the primary choice (kNN acceptable as a fallback),
  loaded from a pre-trained model file.
- Model training is **out of scope**; a pre-fitted classifier artifact is
  assumed to be provided/vendored.

### 1.3 CNN approach (sketch — no code)

- A **LeNet-style lightweight CNN** (small conv/pool/fully-connected network)
  appropriate for 28×28 MNIST input, exported to **ONNX**.
- Inference runtime (decision): **OpenCV DNN module (`cv::dnn`) loading the
  ONNX model on the default CPU backend.**
  - Rationale: reuses the single OpenCV dependency the classical path already
    needs (no second heavyweight runtime to vendor/build), is CPU-only,
    portable, and trivial to build on Linux. ONNX Runtime is noted as a viable
    alternative but is intentionally **not** selected to keep the dependency
    surface minimal for a "small" suite. See OQ-2.
- Preprocessing must match the model's expected input (scale/normalize to the
  training convention, correct tensor layout `NCHW`, 1×1×28×28).
- **Model training and export are out of scope**; a pre-trained/exported
  `.onnx` model artifact is assumed to be provided/vendored.

## 2. Functional Requirements

- **FR-1** The harness shall load a fixed set of MNIST test images (and their
  ground-truth labels) from a local, vendored/downloadable dataset path
  configured at runtime.
- **FR-2** The harness shall provide a **classical CV** classifier that takes a
  28×28 grayscale image and returns a predicted digit label (0–9) plus a score.
- **FR-3** The harness shall provide a **CNN** classifier (ONNX via `cv::dnn`)
  that takes the same 28×28 grayscale image and returns a predicted digit label
  (0–9) plus a score.
- **FR-4** Both methods shall consume identical inputs and produce outputs in an
  **identical, comparable format** (same label space 0–9, same output schema),
  so results can be compared item-by-item.
- **FR-5** The harness shall compute **classification accuracy** for each method
  over the evaluated set (predicted label vs. ground truth), reported alongside
  performance.
- **FR-6** The harness shall measure **per-item latency** for each method under
  a controlled protocol (see FR-8) and record the raw per-iteration timings.
- **FR-7** The harness shall compute and report **throughput** (items/second)
  for each method under each target configuration.
- **FR-8** The harness shall support a benchmark protocol with configurable
  **warmup iterations** (discarded) and a configurable number of **measured
  iterations**, defaulting to values documented in Section 4.
- **FR-9** The harness shall run each method across **at least two target
  configurations** (see Section 3) within a single invocation or via a
  documented configuration sweep, and label results by target.
- **FR-10** The harness shall emit a **results table** to stdout in a
  human-readable form containing, per (method × target): p50 latency, p95
  latency, mean latency, throughput (items/s), accuracy, iteration count.
- **FR-11** The harness shall additionally emit the same results in a
  **machine-readable file** (CSV and/or JSON) at a configurable output path.
- **FR-12** The harness shall accept configuration via command-line arguments
  and/or a config file for: dataset path, model artifact paths, number of images
  to evaluate, warmup count, iteration count, and thread/target settings.
- **FR-13** The harness shall record and report **run metadata** in the output:
  OS, CPU model string (where available), compiler and build type, OpenCV
  version, thread count per target, and a timestamp.
- **FR-14** The harness shall fail fast with a clear error message if a required
  dataset or model artifact is missing or malformed.

## 3. Hardware-Target Requirement ("at least two hardware targets")

Concrete, portable, reproducible interpretation (decision):

- **TR-1 (primary interpretation):** The "two hardware targets" requirement is
  satisfied by running the benchmark under **at least two distinct CPU execution
  configurations on the same machine**, each treated as a named "target." The
  minimum mandated set is a **thread-count / core-utilization sweep**:
  - Target A: **single-threaded** (1 inference thread / core-pinned where
    supported), representing a constrained/edge-like CPU profile.
  - Target B: **multi-threaded** using all available hardware cores,
    representing a desktop/server CPU profile.
  This makes the "two targets" fully reproducible on any single developer
  machine without special hardware.
- **TR-2 (secondary, opportunistic):** When the suite is also executed on a
  **CI runner** (different physical CPU than the developer machine), those runs
  count as additional real hardware targets and their results tables should be
  retained. This is encouraged but not required for a passing run, because CI
  hardware is not guaranteed stable.
- **TR-3:** Thread/target configuration shall be explicit and logged (FR-13) so
  each row in the results table is unambiguously attributable to a target.

## 4. Metrics Definition

- **MET-1 Warmup:** A configurable number of warmup iterations (default **N=20**
  per method per target) shall be executed and **excluded** from all statistics
  to absorb cold-cache/JIT/allocation effects.
- **MET-2 Measured iterations:** A configurable number of measured iterations
  (default **N=200** timed samples per method per target) shall be collected.
  The evaluated image set may be smaller; iterations may cycle deterministically
  over the set.
- **MET-3 Latency:** Latency is the wall-clock time to classify **one image**
  (preprocessing + inference for that method), measured with a monotonic
  high-resolution clock, reported in **milliseconds**.
- **MET-4 Latency percentiles:** Report **p50 (median)**, **p95**, and **mean**
  latency per method per target, plus min/max for context.
- **MET-5 Throughput:** Report **items per second**, computed over the measured
  (non-warmup) workload, per method per target.
- **MET-6 Accuracy:** Report top-1 classification accuracy over the evaluated
  set per method (a correctness/quality axis independent of timing).
- **MET-7 Reproducibility of metrics:** All defaults (warmup, iterations, image
  count) shall be recorded in the output so a run is self-describing.

## 5. Non-Functional Requirements

- **NFR-1 (CPU-only):** The suite shall run entirely on CPU with **no GPU
  dependency** and no CUDA/OpenCL requirement at build or run time.
- **NFR-2 (Portability):** **Linux is the primary supported platform.** The code
  shall avoid platform-specific APIs where reasonable; Windows/macOS support is
  desirable but not guaranteed for the first iteration.
- **NFR-3 (Reproducible builds):** The project shall build via CMake with pinned
  dependency versions and a documented build procedure; results shall include
  enough metadata (FR-13) to reproduce a run.
- **NFR-4 (Determinism where possible):** Classical-CV outputs shall be
  deterministic for a fixed input and model. CNN outputs shall be deterministic
  for a fixed model/threading configuration; any nondeterminism from thread
  scheduling shall affect only timing, not predicted labels.
- **NFR-5 (Small footprint):** Total vendored artifacts (models + a test subset
  of MNIST) shall remain small enough to commit or fetch quickly, keeping the
  suite "small" and CI-friendly.
- **NFR-6 (License compatibility):** All third-party dependencies, datasets, and
  model artifacts shall use licenses compatible with the repository's LICENSE
  and permit redistribution for a public portfolio.
- **NFR-7 (Minimal dependency surface):** Prefer a single primary CV/inference
  dependency (OpenCV, providing both classical `ml`/`HOG` and `dnn` ONNX
  inference) to simplify building and vendoring.
- **NFR-8 (Measurement hygiene):** Timing shall use a monotonic clock, exclude
  I/O and dataset loading from per-item latency, and separate preprocessing from
  pure inference internally where feasible for fair comparison.
- **NFR-9 (Testability):** Core logic (metric computation, output formatting,
  classifier interfaces) shall be unit-testable via GoogleTest or Catch2.

## 6. Constraints and Assumptions

- **A-1** A pre-trained/exported **LeNet-style ONNX model** for MNIST is
  available/vendored (no training performed by this project).
- **A-2** A pre-fitted **classical classifier artifact** (SVM/kNN) plus its
  feature configuration is available/vendored.
- **A-3** The MNIST test subset used for evaluation is redistributable under a
  license compatible with NFR-6.
- **A-4** OpenCV (with `ml`, `dnn`, and `imgproc`/HOG support) is available as
  the primary dependency on the target platforms.
- **A-5** Both methods target the **same label space (digits 0–9)** and the same
  28×28 grayscale input, enabling item-by-item comparison.
- **A-6** The benchmark runs on general-purpose CPUs; no special accelerators,
  NPUs, or SIMD extensions beyond what the compiler/OpenCV use by default are
  required.

## 7. Out of Scope

- **OOS-1** Model **training**, fine-tuning, or ONNX export tooling.
- **OOS-2** **GPU / CUDA / OpenCL / NPU** acceleration of any kind.
- **OOS-3** Real-time or streaming **video** inference.
- **OOS-4** **Mobile / embedded-device** deployment and cross-compilation to ARM
  edge boards (may be a future extension; not this iteration).
- **OOS-5** A GUI, web dashboard, or interactive visualization (a static table /
  CSV / JSON is sufficient).
- **OOS-6** Tasks beyond MNIST digit classification (no multi-task or
  multi-dataset comparison in this iteration).
- **OOS-7** Distributed or multi-process benchmarking; the suite is single-host.

## 8. Acceptance Criteria

- **AC-1** Building the project per its documented CMake instructions on Linux
  succeeds with no GPU dependency present.
- **AC-2** Running the harness with valid dataset and model artifacts produces,
  for **both** the classical and CNN methods, a predicted digit (0–9) and score
  for each evaluated image (verifies FR-2, FR-3, FR-4).
- **AC-3** The harness reports **accuracy** for each method over the evaluated
  set (verifies FR-5, MET-6).
- **AC-4** The results table includes, per (method × target): p50, p95, mean
  latency (ms), throughput (items/s), accuracy, and iteration count
  (verifies FR-6, FR-7, FR-10, MET-3..MET-6).
- **AC-5** The harness runs across **at least two named target configurations**
  (single-threaded and all-cores minimum) in a documented way, and each results
  row is labeled by its target (verifies FR-9, TR-1, TR-3).
- **AC-6** Warmup iterations are excluded from statistics, and the warmup and
  measured iteration counts are reported in the output (verifies FR-8, MET-1,
  MET-2, MET-7).
- **AC-7** A machine-readable results file (CSV and/or JSON) is written to the
  configured path and contains the same metrics as the console table
  (verifies FR-11).
- **AC-8** Output includes run metadata: OS, CPU string, compiler/build type,
  OpenCV version, per-target thread count, and timestamp (verifies FR-13).
- **AC-9** Missing or malformed dataset/model artifacts cause a clear,
  non-crashing error message and non-zero exit (verifies FR-14).
- **AC-10** Re-running with identical inputs/config yields identical predicted
  labels and accuracy for both methods (timings may vary) (verifies NFR-4).
- **AC-11** All bundled dependencies/datasets/models carry licenses compatible
  with the repository LICENSE (verifies NFR-6).

## 9. Open Questions (for downstream confirmation)

- **OQ-1** Exact classical classifier: confirm **SVM vs. kNN** (and HOG
  parameters) — this SRS assumes HOG + SVM as primary.
- **OQ-2** Confirm the CNN inference runtime: this SRS selects **OpenCV `dnn`
  (ONNX, CPU)**; confirm whether a second target of **ONNX Runtime** is desired
  as a future comparison axis.
- **OQ-3** Source and licensing of the pre-trained ONNX model and the classical
  classifier artifact — where are they obtained/vendored from?
- **OQ-4** How is the MNIST test subset provisioned: committed to the repo,
  downloaded at build/run time, or provided by the user? Confirm size/licensing.
- **OQ-5** Output format priority: is **CSV**, **JSON**, or **both** required for
  the machine-readable results?
- **OQ-6** Preferred test framework: **GoogleTest vs. Catch2** (both acceptable
  per project conventions).
- **OQ-7** Whether real CI-hardware runs (TR-2) should be captured as committed
  reference results or treated as informational only.
- **OQ-8** Default evaluation set size (number of images) and default
  warmup/iteration counts — confirm the Section 4 defaults are acceptable.

---

**Requirements are ready for the System Architect agent to design against.**
