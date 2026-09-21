#pragma once

#include <string>
#include <vector>

#include "eib/types.hpp"

namespace eib {

/// Polymorphic classifier abstraction shared by the classical and CNN
/// pipelines (FR-2, FR-3, FR-4). Both consume an identical 28x28 grayscale
/// input and emit an identical Prediction schema.
class IClassifier {
public:
    virtual ~IClassifier() = default;

    /// Single-image predict: preprocessing + inference for ONE image.
    /// This is the exact scope timed per iteration (MET-3, NFR-8).
    virtual Prediction predict(const cv::Mat& image28x28) const = 0;

    /// Stable identifier used to label results, e.g. "classical" / "cnn".
    virtual std::string name() const = 0;
};

/// Untimed helper that classifies every sample; used for accuracy passes.
std::vector<Prediction> predict_all(const IClassifier& clf,
                                    const std::vector<Sample>& samples);

} // namespace eib
