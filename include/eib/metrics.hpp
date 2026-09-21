#pragma once

#include <string>
#include <vector>

namespace eib {

/// Latency summary statistics, all values in milliseconds (MET-3, MET-4).
struct Statistics {
    double p50 = 0.0;
    double p95 = 0.0;
    double mean = 0.0;
    double min = 0.0;
    double max = 0.0;
};

/// One fully computed result row per (method x target) (MET-3..MET-6, FR-10).
struct BenchmarkResult {
    std::string method;         ///< classifier name.
    std::string target;         ///< TargetConfig name.
    int num_threads = 0;        ///< effective threads (FR-13).
    int iterations = 0;         ///< measured count (MET-2).
    int warmup = 0;             ///< discarded count (MET-1).
    Statistics latency_ms;      ///< latency percentiles (MET-4).
    double throughput_ips = 0.0;///< items/second (MET-5).
    double accuracy = 0.0;      ///< top-1 over evaluated set (MET-6).
};

/// Nearest-rank percentile of a latency sample (MET-4, MET-7): sort ascending,
/// index = ceil(p/100 * N) - 1 clamped to [0, N-1]. Returns 0 for an empty
/// input. The argument is taken by value and sorted internally.
double percentile(std::vector<double> values, double p);

/// Accumulates per-item latency and correctness, then computes a result row.
class MetricsCollector {
public:
    /// Record one measured item (MET-2).
    void record(double latency_ms, bool correct);

    /// Number of recorded measured items.
    std::size_t count() const { return latencies_ms_.size(); }

    /// Compute statistics, throughput, and accuracy into a BenchmarkResult.
    /// accuracy_override, when >= 0, supplies accuracy measured on the
    /// evaluated set (MET-6); otherwise correctness of recorded items is used.
    BenchmarkResult finalize(std::string method,
                             std::string target,
                             int num_threads,
                             int warmup,
                             double accuracy_override = -1.0) const;

private:
    std::vector<double> latencies_ms_;
    int correct_ = 0;
};

} // namespace eib
