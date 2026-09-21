#pragma once

#include <string>

#include "eib/benchmark_config.hpp"

namespace eib {

/// Environment and configuration metadata captured for reproducibility
/// (FR-13, AC-8). Value semantics; cheap to copy and easy to test.
struct RunMetadata {
    std::string os;             ///< OS name/version (best effort).
    std::string cpu_model;      ///< CPU model string (best effort).
    std::string compiler;       ///< compiler id/version + build type.
    std::string opencv_version; ///< CV_VERSION.
    std::string timestamp_utc;  ///< ISO-8601 UTC timestamp.
    int max_images = 0;         ///< evaluated set size (MET-7).
    int warmup = 0;             ///< warmup iterations (MET-7).
    int iterations = 0;         ///< measured iterations (MET-7).
};

/// Captures run metadata from the host and the given configuration.
RunMetadata capture_run_metadata(const BenchmarkConfig& cfg);

} // namespace eib
