#include "eib/reporter.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace eib {
namespace {

std::string num(double value, int precision) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(precision) << value;
    return os.str();
}

} // namespace

std::string format_csv(const std::vector<BenchmarkResult>& results,
                       const RunMetadata& meta) {
    std::ostringstream os;
    // Metadata echoed as leading comment rows (MET-7, FR-13).
    os << "# os," << meta.os << "\n";
    os << "# cpu_model," << meta.cpu_model << "\n";
    os << "# compiler," << meta.compiler << "\n";
    os << "# opencv_version," << meta.opencv_version << "\n";
    os << "# timestamp_utc," << meta.timestamp_utc << "\n";
    os << "# max_images," << meta.max_images << "\n";
    os << "# warmup," << meta.warmup << "\n";
    os << "# iterations," << meta.iterations << "\n";

    os << "method,target,num_threads,iterations,warmup,"
          "p50_ms,p95_ms,mean_ms,min_ms,max_ms,throughput_ips,accuracy\n";

    for (const BenchmarkResult& r : results) {
        os << r.method << ',' << r.target << ',' << r.num_threads << ','
           << r.iterations << ',' << r.warmup << ',' << num(r.latency_ms.p50, 6)
           << ',' << num(r.latency_ms.p95, 6) << ',' << num(r.latency_ms.mean, 6)
           << ',' << num(r.latency_ms.min, 6) << ',' << num(r.latency_ms.max, 6)
           << ',' << num(r.throughput_ips, 4) << ',' << num(r.accuracy, 6)
           << '\n';
    }
    return os.str();
}

CsvReporter::CsvReporter(std::string path) : path_(std::move(path)) {}

void CsvReporter::report(const std::vector<BenchmarkResult>& results,
                         const RunMetadata& meta) {
    std::ofstream out(path_);
    if (!out) {
        throw std::runtime_error("CsvReporter: cannot open '" + path_ + "'");
    }
    out << format_csv(results, meta);
}

} // namespace eib
