// Offline SVM trainer for the classical MNIST pipeline.
//
// This tool is intentionally OUTSIDE the benchmarked library/app code path
// (resolving design gap G-1): model training is not part of eib_core or the
// benchmark app. It links eib_core only to reuse the EXACT HOG configuration
// (eib::compute_hog) so the produced model stays compatible with
// ClassicalClassifier (G-2). It reads MNIST IDX data and writes an OpenCV SVM
// artifact that the benchmark can then load.
//
// Usage:
//   eib_train_svm --images data/train-images.idx3-ubyte
//                 --labels data/train-labels.idx1-ubyte
//                 --out models/mnist_svm.xml [--max 5000]

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

#include "eib/dataset.hpp"
#include "eib/hog_features.hpp"

int main(int argc, char** argv) {
    std::string images_path;
    std::string labels_path;
    std::string out_path = "models/mnist_svm.xml";
    int max_items = 5000;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + flag);
            }
            return argv[++i];
        };
        try {
            if (arg == "--images") {
                images_path = value(arg);
            } else if (arg == "--labels") {
                labels_path = value(arg);
            } else if (arg == "--out") {
                out_path = value(arg);
            } else if (arg == "--max") {
                max_items = std::stoi(value(arg));
            } else {
                std::cerr << "unknown argument '" << arg << "'\n";
                return 2;
            }
        } catch (const std::exception& e) {
            std::cerr << "error: " << e.what() << "\n";
            return 2;
        }
    }

    if (images_path.empty() || labels_path.empty()) {
        std::cerr << "usage: eib_train_svm --images IDX3 --labels IDX1 "
                     "[--out models/mnist_svm.xml] [--max N]\n";
        return 2;
    }

    try {
        eib::MnistDataset dataset(images_path, labels_path, max_items);
        const std::vector<eib::Sample> samples = dataset.load();
        if (samples.empty()) {
            std::cerr << "error: no training samples loaded\n";
            return 1;
        }
        std::cout << "Loaded " << samples.size() << " training samples\n";

        const int feature_len =
            static_cast<int>(eib::compute_hog(samples.front().image, true).size());
        cv::Mat features(static_cast<int>(samples.size()), feature_len, CV_32F);
        cv::Mat labels(static_cast<int>(samples.size()), 1, CV_32S);

        for (std::size_t i = 0; i < samples.size(); ++i) {
            const std::vector<float> hog = eib::compute_hog(samples[i].image, true);
            for (int j = 0; j < feature_len; ++j) {
                features.at<float>(static_cast<int>(i), j) = hog[static_cast<std::size_t>(j)];
            }
            labels.at<int>(static_cast<int>(i), 0) = samples[i].label;
        }

        cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
        svm->setType(cv::ml::SVM::C_SVC);
        svm->setKernel(cv::ml::SVM::RBF);
        svm->setC(12.5);
        svm->setGamma(0.05);
        svm->setTermCriteria(cv::TermCriteria(
            cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 1e-6));

        std::cout << "Training SVM (RBF) on HOG features...\n";
        svm->train(features, cv::ml::ROW_SAMPLE, labels);
        svm->save(out_path);
        std::cout << "Saved model to " << out_path << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
