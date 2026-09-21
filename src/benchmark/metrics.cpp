#include "eib/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace eib {

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());

    const double n = static_cast<double>(values.size());
    long index = static_cast<long>(std::ceil((p / 100.0) * n)) - 1;
    if (index < 0) {
        index = 0;
    }
    if (index >= static_cast<long>(values.size())) {
        index = static_cast<long>(values.size()) - 1;
    }
    return values[static_cast<std::size_t>(index)];
}

void MetricsCollector::record(double latency_ms, bool correct) {
    latencies_ms_.push_back(latency_ms);
    if (correct) {
        ++correct_;
    }
}

BenchmarkResult MetricsCollector::finalize(std::string method,
                                           std::string target,
                                           int num_threads,
                                           int warmup,
                                           double accuracy_override) const {
    BenchmarkResult r;
    r.method = std::move(method);
    r.target = std::move(target);
    r.num_threads = num_threads;
    r.warmup = warmup;
    r.iterations = static_cast<int>(latencies_ms_.size());

    if (!latencies_ms_.empty()) {
        r.latency_ms.p50 = percentile(latencies_ms_, 50.0);
        r.latency_ms.p95 = percentile(latencies_ms_, 95.0);
        const double sum =
            std::accumulate(latencies_ms_.begin(), latencies_ms_.end(), 0.0);
        r.latency_ms.mean = sum / static_cast<double>(latencies_ms_.size());
        r.latency_ms.min =
            *std::min_element(latencies_ms_.begin(), latencies_ms_.end());
        r.latency_ms.max =
            *std::max_element(latencies_ms_.begin(), latencies_ms_.end());

        const double sum_seconds = sum / 1000.0;
        r.throughput_ips =
            sum_seconds > 0.0 ? static_cast<double>(latencies_ms_.size()) / sum_seconds
                              : 0.0;
    }

    if (accuracy_override >= 0.0) {
        r.accuracy = accuracy_override;
    } else if (!latencies_ms_.empty()) {
        r.accuracy = static_cast<double>(correct_) /
                     static_cast<double>(latencies_ms_.size());
    }
    return r;
}

} // namespace eib
