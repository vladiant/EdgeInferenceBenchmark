#pragma once

#include <string>
#include <vector>

namespace eib {

/// The two benchmarked pipelines behind IClassifier.
enum class Method { Classical, Cnn };

/// Parses a method name ("classical" / "cnn"); throws std::runtime_error on an
/// unknown value.
Method method_from_string(const std::string& name);

/// Canonical lowercase name for a Method ("classical" / "cnn").
std::string to_string(Method method);

/// One named execution target = one OpenCV thread configuration (TR-1, TR-3).
struct TargetConfig {
    std::string name;        ///< "single-thread", "all-cores".
    int num_threads = 1;     ///< 1 = single; 0 = OpenCV default (all cores).
};

/// Immutable, value-semantic run configuration assembled from the CLI (FR-12).
struct BenchmarkConfig {
    std::string images_path;     ///< MNIST IDX images (FR-1).
    std::string labels_path;     ///< MNIST IDX labels.
    std::string svm_model_path;  ///< classical artifact (A-2).
    std::string onnx_model_path; ///< CNN artifact (A-1).

    int max_images = 200; ///< evaluated set size (OQ-8 default).
    int warmup = 20;      ///< discarded iterations (MET-1 default).
    int iterations = 200; ///< measured iterations (MET-2 default).

    std::vector<Method> methods{Method::Classical, Method::Cnn};
    std::vector<TargetConfig> targets{{"single-thread", 1}, {"all-cores", 0}};

    std::string csv_out;  ///< optional CSV path (FR-11).
    std::string json_out; ///< optional JSON path (FR-11).
};

} // namespace eib
