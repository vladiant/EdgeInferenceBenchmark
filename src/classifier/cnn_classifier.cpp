#include "eib/cnn_classifier.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace eib {

struct CnnClassifier::Impl {
    mutable cv::dnn::Net net; // forward() is non-const in OpenCV.
    float scale = 1.f / 255.f;
    float mean = 0.f;
};

CnnClassifier::CnnClassifier(const CnnConfig& cfg)
    : impl_(std::make_unique<Impl>()) {
    if (cfg.onnx_path.empty()) {
        throw std::runtime_error(
            "CnnClassifier: no ONNX model path configured (see models/README.md)");
    }
    try {
        impl_->net = cv::dnn::readNetFromONNX(cfg.onnx_path);
    } catch (const cv::Exception& e) {
        throw std::runtime_error(
            "CnnClassifier: failed to read ONNX model '" + cfg.onnx_path +
            "': " + e.what());
    }
    if (impl_->net.empty()) {
        throw std::runtime_error(
            "CnnClassifier: ONNX model '" + cfg.onnx_path + "' loaded empty");
    }
    impl_->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    impl_->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    impl_->scale = cfg.scale;
    impl_->mean = cfg.mean;
}

CnnClassifier::~CnnClassifier() = default;

Prediction CnnClassifier::predict(const cv::Mat& image28x28) const {
    cv::Mat blob = cv::dnn::blobFromImage(
        image28x28, impl_->scale, cv::Size(28, 28),
        cv::Scalar(impl_->mean), /*swapRB=*/false, /*crop=*/false, CV_32F);

    impl_->net.setInput(blob);
    cv::Mat out = impl_->net.forward();
    out = out.reshape(1, 1); // flatten to 1 x num_classes

    // Softmax for a normalised score; argmax for the label.
    cv::Mat prob;
    cv::exp(out - static_cast<float>(*std::max_element(
                      out.begin<float>(), out.end<float>())),
            prob);
    const float sum = static_cast<float>(cv::sum(prob)[0]);
    if (sum > 0.f) {
        prob /= sum;
    }

    cv::Point max_loc;
    double max_val = 0.0;
    cv::minMaxLoc(prob, nullptr, &max_val, nullptr, &max_loc);

    Prediction p;
    p.label = max_loc.x;
    p.score = static_cast<float>(max_val);
    return p;
}

} // namespace eib
