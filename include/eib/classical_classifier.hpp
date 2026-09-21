#pragma once

#include <memory>
#include <string>

#include "eib/classifier.hpp"

namespace eib {

/// Configuration for the classical HOG + SVM/kNN pipeline (FR-2).
struct ClassicalConfig {
    std::string model_path; ///< pre-fitted SVM/kNN artifact (A-2).
    bool deskew = true;     ///< moment-based deskew (SRS 1.2).
};

/// Classical OpenCV pipeline: moment deskew -> HOG descriptor -> cv::ml model.
/// The model is loaded once at construction, outside any timed region (NFR-8).
class ClassicalClassifier final : public IClassifier {
public:
    explicit ClassicalClassifier(const ClassicalConfig& cfg);
    ~ClassicalClassifier() override;

    ClassicalClassifier(const ClassicalClassifier&) = delete;
    ClassicalClassifier& operator=(const ClassicalClassifier&) = delete;

    Prediction predict(const cv::Mat& image28x28) const override;
    std::string name() const override { return "classical"; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace eib
