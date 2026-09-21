#include "eib/reporter.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace eib {
namespace {

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (const char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\t': out += "\\t"; break;
        case '\r': out += "\\r"; break;
        default: out += c; break;
        }
    }
    return out;
}

std::string num(double value, int precision) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(precision) << value;
    return os.str();
}

} // namespace

std::string format_json(const std::vector<BenchmarkResult>& results,
                        const RunMetadata& meta) {
    std::ostringstream os;
    os << "{\n";
    os << "  \"metadata\": {\n";
    os << "    \"os\": \"" << json_escape(meta.os) << "\",\n";
    os << "    \"cpu_model\": \"" << json_escape(meta.cpu_model) << "\",\n";
    os << "    \"compiler\": \"" << json_escape(meta.compiler) << "\",\n";
    os << "    \"opencv_version\": \"" << json_escape(meta.opencv_version) << "\",\n";
    os << "    \"timestamp_utc\": \"" << json_escape(meta.timestamp_utc) << "\",\n";
    os << "    \"max_images\": " << meta.max_images << ",\n";
    os << "    \"warmup\": " << meta.warmup << ",\n";
    os << "    \"iterations\": " << meta.iterations << "\n";
    os << "  },\n";
    os << "  \"results\": [\n";

    for (std::size_t i = 0; i < results.size(); ++i) {
        const BenchmarkResult& r = results[i];
        os << "    {\n";
        os << "      \"method\": \"" << json_escape(r.method) << "\",\n";
        os << "      \"target\": \"" << json_escape(r.target) << "\",\n";
        os << "      \"num_threads\": " << r.num_threads << ",\n";
        os << "      \"iterations\": " << r.iterations << ",\n";
        os << "      \"warmup\": " << r.warmup << ",\n";
        os << "      \"p50_ms\": " << num(r.latency_ms.p50, 6) << ",\n";
        os << "      \"p95_ms\": " << num(r.latency_ms.p95, 6) << ",\n";
        os << "      \"mean_ms\": " << num(r.latency_ms.mean, 6) << ",\n";
        os << "      \"min_ms\": " << num(r.latency_ms.min, 6) << ",\n";
        os << "      \"max_ms\": " << num(r.latency_ms.max, 6) << ",\n";
        os << "      \"throughput_ips\": " << num(r.throughput_ips, 4) << ",\n";
        os << "      \"accuracy\": " << num(r.accuracy, 6) << "\n";
        os << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }

    os << "  ]\n";
    os << "}\n";
    return os.str();
}

JsonReporter::JsonReporter(std::string path) : path_(std::move(path)) {}

void JsonReporter::report(const std::vector<BenchmarkResult>& results,
                          const RunMetadata& meta) {
    std::ofstream out(path_);
    if (!out) {
        throw std::runtime_error("JsonReporter: cannot open '" + path_ + "'");
    }
    out << format_json(results, meta);
}

} // namespace eib
