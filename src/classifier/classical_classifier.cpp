#include "eib/classical_classifier.hpp"

#include <memory>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

#include "eib/hog_features.hpp"

namespace eib {

struct ClassicalClassifier::Impl {
    cv::Ptr<cv::ml::StatModel> model;
    bool deskew = true;
};

namespace {

cv::Ptr<cv::ml::StatModel> load_model(const std::string& path) {
    if (path.empty()) {
        throw std::runtime_error(
            "ClassicalClassifier: no model path configured (see models/README.md)");
    }

    // Primary: SVM (OQ-1). Fallback: kNN. Both are cv::ml::StatModel.
    try {
        cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(path);
        if (svm && svm->isTrained()) {
            return svm;
        }
    } catch (const cv::Exception&) {
        // fall through to kNN
    }

    try {
        cv::Ptr<cv::ml::KNearest> knn =
            cv::Algorithm::load<cv::ml::KNearest>(path);
        if (knn && knn->isTrained()) {
            return knn;
        }
    } catch (const cv::Exception&) {
        // fall through to error
    }

    throw std::runtime_error(
        "ClassicalClassifier: failed to load a trained SVM or kNN model from '" +
        path + "'. Produce one with scripts/train_svm (see scripts/README.md).");
}

} // namespace

ClassicalClassifier::ClassicalClassifier(const ClassicalConfig& cfg)
    : impl_(std::make_unique<Impl>()) {
    impl_->model = load_model(cfg.model_path);
    impl_->deskew = cfg.deskew;
}

ClassicalClassifier::~ClassicalClassifier() = default;

Prediction ClassicalClassifier::predict(const cv::Mat& image28x28) const {
    const std::vector<float> features = compute_hog(image28x28, impl_->deskew);
    const cv::Mat feature_row(1, static_cast<int>(features.size()), CV_32F,
                              const_cast<float*>(features.data()));

    const float response = impl_->model->predict(feature_row);
    Prediction p;
    p.label = static_cast<int>(response + (response < 0 ? -0.5f : 0.5f));
    p.score = 1.0f; // multiclass StatModel exposes no calibrated score (OQ-1).
    return p;
}

} // namespace eib
