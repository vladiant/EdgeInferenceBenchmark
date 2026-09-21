#pragma once

#include <vector>

#include "eib/benchmark_config.hpp"
#include "eib/classifier.hpp"
#include "eib/metrics.hpp"
#include "eib/types.hpp"

namespace eib {

/// Runs the warmup + measured loop for one classifier across every target and
/// returns one BenchmarkResult per target (MET-1..MET-5). Model-free and pure
/// w.r.t. the classifier, so it is unit-testable with a stub classifier.
/// The measured loop cycles deterministically over `samples`.
std::vector<BenchmarkResult> sweep_classifier(const IClassifier& clf,
                                              const std::vector<Sample>& samples,
                                              const std::vector<TargetConfig>& targets,
                                              int warmup,
                                              int iterations,
                                              double accuracy);

/// Drives the benchmark: loads the dataset once, constructs the requested
/// classifiers, and sweeps every (method x target) pair, returning one
/// BenchmarkResult per pair (FR-6..FR-9). I/O and model loads are excluded
/// from the timed region (NFR-8).
class BenchmarkRunner {
public:
    explicit BenchmarkRunner(BenchmarkConfig cfg);

    std::vector<BenchmarkResult> run();

private:
    BenchmarkConfig cfg_;
};

} // namespace eib
