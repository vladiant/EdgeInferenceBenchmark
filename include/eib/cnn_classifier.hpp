#pragma once

#include <memory>
#include <string>

#include "eib/classifier.hpp"

namespace eib {

/// Configuration for the ONNX CNN pipeline running on the OpenCV CPU backend.
struct CnnConfig {
    std::string onnx_path;       ///< LeNet-style model (A-1).
    float scale = 1.f / 255.f;   ///< input scaling; must match training (SRS 1.3).
    float mean = 0.f;            ///< per-pixel mean subtracted before scaling.
};

/// CNN pipeline: cv::dnn ONNX model on DNN_TARGET_CPU. The model is loaded
/// once at construction (NFR-8); a missing/malformed file throws
/// std::runtime_error with a clear message (FR-14).
class CnnClassifier final : public IClassifier {
public:
    explicit CnnClassifier(const CnnConfig& cfg);
    ~CnnClassifier() override;

    CnnClassifier(const CnnClassifier&) = delete;
    CnnClassifier& operator=(const CnnClassifier&) = delete;

    Prediction predict(const cv::Mat& image28x28) const override;
    std::string name() const override { return "cnn"; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace eib
