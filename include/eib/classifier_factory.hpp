#pragma once

#include <memory>

#include "eib/benchmark_config.hpp"
#include "eib/classifier.hpp"

namespace eib {

/// Constructs the requested classifier, loading its artifacts eagerly.
/// Throws std::runtime_error with a clear message on failure (FR-14).
std::unique_ptr<IClassifier> make_classifier(Method method,
                                             const BenchmarkConfig& cfg);

} // namespace eib
