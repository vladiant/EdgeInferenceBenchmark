#pragma once

#include <memory>
#include <string>
#include <vector>

#include "eib/metrics.hpp"
#include "eib/run_metadata.hpp"

namespace eib {

/// Emits benchmark results plus run metadata to a destination (FR-10, FR-11).
class IResultsReporter {
public:
    virtual ~IResultsReporter() = default;
    virtual void report(const std::vector<BenchmarkResult>& results,
                        const RunMetadata& meta) = 0;
};

/// Writes an aligned, human-readable table to stdout (FR-10, AC-4).
class ConsoleReporter final : public IResultsReporter {
public:
    void report(const std::vector<BenchmarkResult>& results,
                const RunMetadata& meta) override;
};

/// Writes results as CSV to a file path (FR-11).
class CsvReporter final : public IResultsReporter {
public:
    explicit CsvReporter(std::string path);
    void report(const std::vector<BenchmarkResult>& results,
                const RunMetadata& meta) override;

private:
    std::string path_;
};

/// Writes results as JSON to a file path via a tiny hand-written serializer
/// (FR-11, NFR-7).
class JsonReporter final : public IResultsReporter {
public:
    explicit JsonReporter(std::string path);
    void report(const std::vector<BenchmarkResult>& results,
                const RunMetadata& meta) override;

private:
    std::string path_;
};

/// Formats the console table into a string (exposed for unit tests, FR-10).
std::string format_console_table(const std::vector<BenchmarkResult>& results,
                                 const RunMetadata& meta);

/// Formats the CSV document into a string (exposed for unit tests, FR-11).
std::string format_csv(const std::vector<BenchmarkResult>& results,
                       const RunMetadata& meta);

/// Formats the JSON document into a string (exposed for unit tests, FR-11).
std::string format_json(const std::vector<BenchmarkResult>& results,
                        const RunMetadata& meta);

} // namespace eib
