#include <gtest/gtest.h>

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "eib/classifier.hpp"
#include "eib/runner.hpp"
#include "eib/types.hpp"

namespace {

// Deterministic stub: predicts the label stored in the top-left pixel, so the
// test controls correctness precisely without any OpenCV model (design 7).
class StubClassifier final : public eib::IClassifier {
public:
    mutable int calls = 0;

    eib::Prediction predict(const cv::Mat& image) const override {
        ++calls;
        eib::Prediction p;
        p.label = static_cast<int>(image.at<unsigned char>(0, 0));
        p.score = 1.0f;
        return p;
    }
    std::string name() const override { return "stub"; }
};

std::vector<eib::Sample> make_samples(const std::vector<int>& labels) {
    std::vector<eib::Sample> samples;
    for (int lbl : labels) {
        cv::Mat img(28, 28, CV_8UC1, cv::Scalar(lbl));
        samples.push_back(eib::Sample{img, lbl});
    }
    return samples;
}

} // namespace

TEST(SweepClassifier, OneResultPerTargetWithCounts) {
    const auto samples = make_samples({0, 1, 2, 3});
    StubClassifier clf;
    const std::vector<eib::TargetConfig> targets{{"single", 1}, {"all", 0}};

    const std::vector<eib::BenchmarkResult> results = eib::sweep_classifier(
        clf, samples, targets, /*warmup=*/5, /*iterations=*/10, /*accuracy=*/1.0);

    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0].target, "single");
    EXPECT_EQ(results[1].target, "all");
    for (const auto& r : results) {
        EXPECT_EQ(r.method, "stub");
        EXPECT_EQ(r.iterations, 10); // measured count only (warmup excluded)
        EXPECT_EQ(r.warmup, 5);
        EXPECT_DOUBLE_EQ(r.accuracy, 1.0); // override passed through
    }
}

TEST(SweepClassifier, WarmupExcludedFromMeasuredCount) {
    const auto samples = make_samples({7});
    StubClassifier clf;
    const std::vector<eib::TargetConfig> targets{{"t", 1}};

    const std::vector<eib::BenchmarkResult> results = eib::sweep_classifier(
        clf, samples, targets, /*warmup=*/3, /*iterations=*/4, /*accuracy=*/0.5);

    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].iterations, 4);
    // predict() invoked warmup + iterations = 7 times for the single target.
    EXPECT_EQ(clf.calls, 7);
}

TEST(PredictAll, CyclesAndScoresCorrectness) {
    const auto samples = make_samples({0, 1, 2});
    StubClassifier clf;

    const std::vector<eib::Prediction> preds = eib::predict_all(clf, samples);
    ASSERT_EQ(preds.size(), 3u);
    EXPECT_EQ(preds[0].label, 0);
    EXPECT_EQ(preds[1].label, 1);
    EXPECT_EQ(preds[2].label, 2);
}
