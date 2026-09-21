#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "eib/benchmark_config.hpp"
#include "eib/classifier.hpp"
#include "eib/classifier_factory.hpp"
#include "eib/dataset.hpp"
#include "eib/metrics.hpp"
#include "eib/reporter.hpp"
#include "eib/run_metadata.hpp"
#include "eib/runner.hpp"

namespace {

void print_usage(const char* argv0) {
    std::cout <<
        "Usage: " << argv0 << " [options]\n"
        "  --images PATH        MNIST IDX3 image file (required)\n"
        "  --labels PATH        MNIST IDX1 label file (required)\n"
        "  --svm PATH           classical SVM/kNN model (.xml/.yml)\n"
        "  --onnx PATH          LeNet-style ONNX model (.onnx)\n"
        "  --methods LIST       comma list of classical,cnn (default: both)\n"
        "  --max-images N       evaluated set size (default 200)\n"
        "  --warmup N           discarded warmup iterations (default 20)\n"
        "  --iterations N       measured iterations (default 200)\n"
        "  --csv PATH           write CSV results to PATH\n"
        "  --json PATH          write JSON results to PATH\n"
        "  --help               show this help\n";
}

std::string require_value(int argc, char** argv, int& i, const std::string& flag) {
    if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + flag);
    }
    return argv[++i];
}

std::vector<eib::Method> parse_methods(const std::string& csv) {
    std::vector<eib::Method> methods;
    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            methods.push_back(eib::method_from_string(token));
        }
    }
    if (methods.empty()) {
        throw std::runtime_error("--methods produced an empty list");
    }
    return methods;
}

} // namespace

int main(int argc, char** argv) {
    eib::BenchmarkConfig cfg;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            } else if (arg == "--images") {
                cfg.images_path = require_value(argc, argv, i, arg);
            } else if (arg == "--labels") {
                cfg.labels_path = require_value(argc, argv, i, arg);
            } else if (arg == "--svm") {
                cfg.svm_model_path = require_value(argc, argv, i, arg);
            } else if (arg == "--onnx") {
                cfg.onnx_model_path = require_value(argc, argv, i, arg);
            } else if (arg == "--methods") {
                cfg.methods = parse_methods(require_value(argc, argv, i, arg));
            } else if (arg == "--max-images") {
                cfg.max_images = std::stoi(require_value(argc, argv, i, arg));
            } else if (arg == "--warmup") {
                cfg.warmup = std::stoi(require_value(argc, argv, i, arg));
            } else if (arg == "--iterations") {
                cfg.iterations = std::stoi(require_value(argc, argv, i, arg));
            } else if (arg == "--csv") {
                cfg.csv_out = require_value(argc, argv, i, arg);
            } else if (arg == "--json") {
                cfg.json_out = require_value(argc, argv, i, arg);
            } else {
                throw std::runtime_error("unknown argument '" + arg + "'");
            }
        }

        if (cfg.images_path.empty() || cfg.labels_path.empty()) {
            print_usage(argv[0]);
            throw std::runtime_error("--images and --labels are required");
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }

    try {
        // Load the dataset once, outside any timed region (NFR-8).
        eib::MnistDataset dataset(cfg.images_path, cfg.labels_path, cfg.max_images);
        const std::vector<eib::Sample> samples = dataset.load();
        if (samples.empty()) {
            std::cerr << "error: dataset is empty\n";
            return 1;
        }

        std::vector<eib::BenchmarkResult> results;
        int constructed = 0;

        for (const eib::Method method : cfg.methods) {
            std::unique_ptr<eib::IClassifier> clf;
            try {
                clf = eib::make_classifier(method, cfg);
            } catch (const std::exception& e) {
                // Degrade gracefully: skip a method whose artifact is missing
                // (e.g. no ONNX model) but keep running the others (FR-14).
                std::cerr << "warning: skipping method '"
                          << eib::to_string(method) << "': " << e.what() << "\n";
                continue;
            }
            ++constructed;

            const std::vector<eib::Prediction> preds =
                eib::predict_all(*clf, samples);
            int correct = 0;
            for (std::size_t i = 0; i < samples.size(); ++i) {
                if (preds[i].label == samples[i].label) {
                    ++correct;
                }
            }
            const double accuracy =
                static_cast<double>(correct) / static_cast<double>(samples.size());

            std::vector<eib::BenchmarkResult> per_method = eib::sweep_classifier(
                *clf, samples, cfg.targets, cfg.warmup, cfg.iterations, accuracy);
            results.insert(results.end(), per_method.begin(), per_method.end());
        }

        if (constructed == 0) {
            std::cerr << "error: no classifier could be constructed; check "
                         "model artifact paths\n";
            return 1;
        }

        const eib::RunMetadata meta = eib::capture_run_metadata(cfg);

        eib::ConsoleReporter console;
        console.report(results, meta);

        if (!cfg.csv_out.empty()) {
            eib::CsvReporter(cfg.csv_out).report(results, meta);
            std::cout << "\nWrote CSV: " << cfg.csv_out << "\n";
        }
        if (!cfg.json_out.empty()) {
            eib::JsonReporter(cfg.json_out).report(results, meta);
            std::cout << "Wrote JSON: " << cfg.json_out << "\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
