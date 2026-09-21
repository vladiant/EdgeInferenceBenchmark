#pragma once

#include <opencv2/core.hpp>

namespace eib {

/// One evaluation item: a 28x28, CV_8UC1 image plus its ground-truth label.
struct Sample {
    cv::Mat image;  ///< 28x28, single channel, owns its pixels.
    int label = -1; ///< ground-truth digit 0..9.
};

/// Uniform classifier output schema shared by every IClassifier (FR-4).
struct Prediction {
    int label = -1;    ///< predicted digit 0..9.
    float score = 0.f; ///< confidence / decision score.
};

} // namespace eib
