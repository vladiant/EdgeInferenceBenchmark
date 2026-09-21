#include "eib/reporter.hpp"

#include <array>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace eib {
namespace {

std::string fixed(double value, int precision) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(precision) << value;
    return os.str();
}

} // namespace

std::string format_console_table(const std::vector<BenchmarkResult>& results,
                                 const RunMetadata& meta) {
    std::ostringstream os;
    os << "EdgeInferenceBenchmark results\n";
    os << "  os           : " << meta.os << "\n";
    os << "  cpu          : " << meta.cpu_model << "\n";
    os << "  compiler     : " << meta.compiler << "\n";
    os << "  opencv       : " << meta.opencv_version << "\n";
    os << "  timestamp    : " << meta.timestamp_utc << "\n";
    os << "  max_images   : " << meta.max_images << "\n";
    os << "  warmup       : " << meta.warmup << "\n";
    os << "  iterations   : " << meta.iterations << "\n\n";

    const std::array<std::string, 9> headers = {
        "method", "target", "threads", "iters",
        "p50(ms)", "p95(ms)", "mean(ms)", "thrpt(ips)", "accuracy"};
    const std::array<int, 9> widths = {12, 14, 8, 7, 10, 10, 10, 12, 9};

    for (std::size_t i = 0; i < headers.size(); ++i) {
        os << std::left << std::setw(widths[i]) << headers[i];
    }
    os << "\n";

    for (const BenchmarkResult& r : results) {
        os << std::left << std::setw(widths[0]) << r.method
           << std::setw(widths[1]) << r.target
           << std::setw(widths[2]) << r.num_threads
           << std::setw(widths[3]) << r.iterations
           << std::setw(widths[4]) << fixed(r.latency_ms.p50, 4)
           << std::setw(widths[5]) << fixed(r.latency_ms.p95, 4)
           << std::setw(widths[6]) << fixed(r.latency_ms.mean, 4)
           << std::setw(widths[7]) << fixed(r.throughput_ips, 2)
           << std::setw(widths[8]) << fixed(r.accuracy, 4) << "\n";
    }
    return os.str();
}

void ConsoleReporter::report(const std::vector<BenchmarkResult>& results,
                             const RunMetadata& meta) {
    std::cout << format_console_table(results, meta);
}

} // namespace eib
