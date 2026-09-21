#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "eib/metrics.hpp"

using eib::MetricsCollector;
using eib::percentile;

TEST(Percentile, NearestRankOnTenValues) {
    std::vector<double> v{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}; // unsorted on purpose
    // index = ceil(p/100 * 10) - 1
    EXPECT_DOUBLE_EQ(percentile(v, 50.0), 5.0);  // ceil(5)-1=4 -> 5
    EXPECT_DOUBLE_EQ(percentile(v, 95.0), 10.0); // ceil(9.5)-1=9 -> 10
    EXPECT_DOUBLE_EQ(percentile(v, 100.0), 10.0);
    EXPECT_DOUBLE_EQ(percentile(v, 10.0), 1.0); // ceil(1)-1=0 -> 1
}

TEST(Percentile, EmptyIsZero) {
    EXPECT_DOUBLE_EQ(percentile({}, 50.0), 0.0);
}

TEST(Percentile, SingleElement) {
    EXPECT_DOUBLE_EQ(percentile({42.0}, 50.0), 42.0);
    EXPECT_DOUBLE_EQ(percentile({42.0}, 95.0), 42.0);
}

TEST(MetricsCollector, StatisticsAndThroughput) {
    MetricsCollector c;
    // Ten items, each 10 ms -> sum 100 ms = 0.1 s -> throughput 100 ips.
    for (int i = 0; i < 10; ++i) {
        c.record(10.0, /*correct=*/i < 7); // 7 correct
    }
    const eib::BenchmarkResult r =
        c.finalize("m", "t", /*num_threads=*/4, /*warmup=*/20);

    EXPECT_EQ(r.method, "m");
    EXPECT_EQ(r.target, "t");
    EXPECT_EQ(r.num_threads, 4);
    EXPECT_EQ(r.warmup, 20);
    EXPECT_EQ(r.iterations, 10);
    EXPECT_DOUBLE_EQ(r.latency_ms.p50, 10.0);
    EXPECT_DOUBLE_EQ(r.latency_ms.mean, 10.0);
    EXPECT_DOUBLE_EQ(r.latency_ms.min, 10.0);
    EXPECT_DOUBLE_EQ(r.latency_ms.max, 10.0);
    EXPECT_NEAR(r.throughput_ips, 100.0, 1e-9);
    EXPECT_NEAR(r.accuracy, 0.7, 1e-9);
}

TEST(MetricsCollector, AccuracyOverrideWins) {
    MetricsCollector c;
    c.record(5.0, false);
    c.record(5.0, false);
    const eib::BenchmarkResult r =
        c.finalize("m", "t", 1, 0, /*accuracy_override=*/0.42);
    EXPECT_NEAR(r.accuracy, 0.42, 1e-12);
}

TEST(MetricsCollector, EmptyFinalizeIsSafe) {
    MetricsCollector c;
    const eib::BenchmarkResult r = c.finalize("m", "t", 1, 0);
    EXPECT_EQ(r.iterations, 0);
    EXPECT_DOUBLE_EQ(r.throughput_ips, 0.0);
    EXPECT_DOUBLE_EQ(r.accuracy, 0.0);
}
