#include "eib/run_metadata.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <sstream>

#include <opencv2/core/version.hpp>

namespace eib {
namespace {

std::string read_cpu_model() {
    std::ifstream in("/proc/cpuinfo");
    if (!in) {
        return "unknown";
    }
    std::string line;
    while (std::getline(in, line)) {
        const auto colon = line.find(':');
        if (colon != std::string::npos && line.rfind("model name", 0) == 0) {
            std::string value = line.substr(colon + 1);
            const auto first = value.find_first_not_of(" \t");
            if (first != std::string::npos) {
                return value.substr(first);
            }
        }
    }
    return "unknown";
}

std::string detect_os() {
#if defined(__linux__)
    return "Linux";
#elif defined(_WIN32)
    return "Windows";
#elif defined(__APPLE__)
    return "macOS";
#else
    return "unknown";
#endif
}

std::string detect_compiler() {
    std::ostringstream os;
#if defined(__clang__)
    os << "Clang " << __clang_major__ << "." << __clang_minor__ << "."
       << __clang_patchlevel__;
#elif defined(__GNUC__)
    os << "GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "."
       << __GNUC_PATCHLEVEL__;
#elif defined(_MSC_VER)
    os << "MSVC " << _MSC_VER;
#else
    os << "unknown-compiler";
#endif
#if defined(NDEBUG)
    os << " (Release)";
#else
    os << " (Debug)";
#endif
    return os.str();
}

std::string iso8601_utc_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_utc{};
#if defined(_WIN32)
    gmtime_s(&tm_utc, &t);
#else
    gmtime_r(&t, &tm_utc);
#endif
    std::array<char, 32> buf{};
    std::strftime(buf.data(), buf.size(), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    return std::string(buf.data());
}

} // namespace

RunMetadata capture_run_metadata(const BenchmarkConfig& cfg) {
    RunMetadata meta;
    meta.os = detect_os();
    meta.cpu_model = read_cpu_model();
    meta.compiler = detect_compiler();
    meta.opencv_version = CV_VERSION;
    meta.timestamp_utc = iso8601_utc_now();
    meta.max_images = cfg.max_images;
    meta.warmup = cfg.warmup;
    meta.iterations = cfg.iterations;
    return meta;
}

} // namespace eib
