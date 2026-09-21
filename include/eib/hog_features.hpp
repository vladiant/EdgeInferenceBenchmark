#pragma once

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

namespace eib {

/// Moment-based deskew of a 28x28 grayscale image to reduce slant variance
/// (SRS 1.2). Returns a deskewed copy; the input is not modified.
cv::Mat deskew(const cv::Mat& image);

/// The single HOG configuration used for 28x28 MNIST images. Both
/// ClassicalClassifier and the offline SVM trainer must use this identical
/// descriptor so features and the trained model stay compatible (G-2).
cv::HOGDescriptor make_mnist_hog();

/// Computes the HOG feature vector for a 28x28 grayscale image, optionally
/// applying moment-based deskew first.
std::vector<float> compute_hog(const cv::Mat& image28x28, bool apply_deskew);

} // namespace eib
