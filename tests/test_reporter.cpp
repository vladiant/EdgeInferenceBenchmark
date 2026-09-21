#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "eib/metrics.hpp"
#include "eib/reporter.hpp"
#include "eib/run_metadata.hpp"

namespace {

eib::RunMetadata make_meta() {
    eib::RunMetadata m;
    m.os = "Linux";
    m.cpu_model = "TestCPU";
    m.compiler = "TestCC";
    m.opencv_version = "4.6.0";
    m.timestamp_utc = "2026-01-01T00:00:00Z";
    m.max_images = 200;
    m.warmup = 20;
    m.iterations = 200;
    return m;
}

std::vector<eib::BenchmarkResult> make_results() {
    eib::BenchmarkResult a;
    a.method = "classical";
    a.target = "single-thread";
    a.num_threads = 1;
    a.iterations = 200;
    a.warmup = 20;
    a.latency_ms = {0.5, 0.9, 0.6, 0.4, 1.2};
    a.throughput_ips = 1600.0;
    a.accuracy = 0.95;

    eib::BenchmarkResult b;
    b.method = "cnn";
    b.target = "all-cores";
    b.num_threads = 8;
    b.iterations = 200;
    b.warmup = 20;
    b.latency_ms = {0.3, 0.5, 0.35, 0.25, 0.8};
    b.throughput_ips = 2800.0;
    b.accuracy = 0.98;

    return {a, b};
}

} // namespace

TEST(Reporter, CsvHasHeaderMetadataAndRows) {
    const std::string csv = eib::format_csv(make_results(), make_meta());

    EXPECT_NE(csv.find("# os,Linux"), std::string::npos);
    EXPECT_NE(csv.find("# opencv_version,4.6.0"), std::string::npos);
    EXPECT_NE(csv.find("method,target,num_threads,iterations,warmup,"),
              std::string::npos);
    EXPECT_NE(csv.find("classical,single-thread,1,200,20,"), std::string::npos);
    EXPECT_NE(csv.find("cnn,all-cores,8,200,20,"), std::string::npos);
}

TEST(Reporter, JsonHasMetadataObjectAndResultsArray) {
    const std::string json = eib::format_json(make_results(), make_meta());

    EXPECT_NE(json.find("\"metadata\""), std::string::npos);
    EXPECT_NE(json.find("\"results\""), std::string::npos);
    EXPECT_NE(json.find("\"opencv_version\": \"4.6.0\""), std::string::npos);
    EXPECT_NE(json.find("\"method\": \"classical\""), std::string::npos);
    EXPECT_NE(json.find("\"method\": \"cnn\""), std::string::npos);
    EXPECT_NE(json.find("\"accuracy\": 0.980000"), std::string::npos);
    // Balanced braces sanity check.
    EXPECT_EQ(std::count(json.begin(), json.end(), '{'),
              std::count(json.begin(), json.end(), '}'));
}

TEST(Reporter, ConsoleTableHasHeadersAndRows) {
    const std::string table =
        eib::format_console_table(make_results(), make_meta());

    EXPECT_NE(table.find("method"), std::string::npos);
    EXPECT_NE(table.find("p50(ms)"), std::string::npos);
    EXPECT_NE(table.find("accuracy"), std::string::npos);
    EXPECT_NE(table.find("classical"), std::string::npos);
    EXPECT_NE(table.find("cnn"), std::string::npos);
    EXPECT_NE(table.find("opencv"), std::string::npos);
}
