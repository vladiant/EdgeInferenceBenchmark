#include "eib/hog_features.hpp"

#include <opencv2/imgproc.hpp>

namespace eib {

cv::Mat deskew(const cv::Mat& image) {
    cv::Mat gray;
    if (image.type() != CV_8UC1) {
        image.convertTo(gray, CV_8UC1);
    } else {
        gray = image;
    }

    const cv::Moments m = cv::moments(gray, /*binaryImage=*/false);
    if (std::abs(m.mu02) < 1e-2) {
        return gray.clone();
    }

    const double skew = m.mu11 / m.mu02;
    const double side = static_cast<double>(gray.cols);
    cv::Mat warp = (cv::Mat_<float>(2, 3) << 1.0, skew, -0.5 * side * skew,
                    0.0, 1.0, 0.0);

    cv::Mat out;
    cv::warpAffine(gray, out, warp, gray.size(),
                   cv::WARP_INVERSE_MAP | cv::INTER_LINEAR);
    return out;
}

cv::HOGDescriptor make_mnist_hog() {
    // 28x28 window, 14x14 blocks, 7px stride, 7x7 cells, 9 orientation bins.
    return cv::HOGDescriptor(cv::Size(28, 28), cv::Size(14, 14),
                             cv::Size(7, 7), cv::Size(7, 7), 9);
}

std::vector<float> compute_hog(const cv::Mat& image28x28, bool apply_deskew) {
    cv::Mat prepared = apply_deskew ? deskew(image28x28) : image28x28;
    if (prepared.type() != CV_8UC1) {
        cv::Mat tmp;
        prepared.convertTo(tmp, CV_8UC1);
        prepared = tmp;
    }
    if (prepared.rows != 28 || prepared.cols != 28) {
        cv::resize(prepared, prepared, cv::Size(28, 28));
    }

    const cv::HOGDescriptor hog = make_mnist_hog();
    std::vector<float> descriptor;
    hog.compute(prepared, descriptor);
    return descriptor;
}

} // namespace eib
