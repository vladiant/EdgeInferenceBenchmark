#include "eib/runner.hpp"

#include <chrono>
#include <memory>
#include <stdexcept>
#include <utility>

#include "eib/classifier.hpp"
#include "eib/classifier_factory.hpp"
#include "eib/dataset.hpp"
#include "eib/metrics.hpp"
#include "eib/thread_scope.hpp"

namespace eib {

BenchmarkRunner::BenchmarkRunner(BenchmarkConfig cfg) : cfg_(std::move(cfg)) {}

std::vector<BenchmarkResult> sweep_classifier(const IClassifier& clf,
                                              const std::vector<Sample>& samples,
                                              const std::vector<TargetConfig>& targets,
                                              int warmup,
                                              int iterations,
                                              double accuracy) {
    std::vector<BenchmarkResult> results;
    results.reserve(targets.size());

    for (const TargetConfig& target : targets) {
        ThreadScope scope(target.num_threads);

        // Warmup: discarded (MET-1).
        for (int i = 0; i < warmup; ++i) {
            clf.predict(samples[static_cast<std::size_t>(i) % samples.size()].image);
        }

        // Measured loop (MET-2): time exactly one predict() each (MET-3).
        MetricsCollector collector;
        for (int i = 0; i < iterations; ++i) {
            const Sample& s = samples[static_cast<std::size_t>(i) % samples.size()];
            const auto t0 = std::chrono::steady_clock::now();
            const Prediction p = clf.predict(s.image);
            const auto t1 = std::chrono::steady_clock::now();
            const double ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            collector.record(ms, p.label == s.label);
        }

        results.push_back(collector.finalize(clf.name(), target.name,
                                             scope.effective_threads(), warmup,
                                             accuracy));
    }

    return results;
}

std::vector<BenchmarkResult> BenchmarkRunner::run() {
    // I/O and model loads happen here, outside every timed region (NFR-8).
    MnistDataset dataset(cfg_.images_path, cfg_.labels_path, cfg_.max_images);
    const std::vector<Sample> samples = dataset.load();
    if (samples.empty()) {
        throw std::runtime_error("BenchmarkRunner: dataset is empty");
    }

    std::vector<BenchmarkResult> results;

    for (const Method method : cfg_.methods) {
        std::unique_ptr<IClassifier> clf = make_classifier(method, cfg_);

        // Accuracy over the evaluated set, computed once and untimed (MET-6).
        const std::vector<Prediction> preds = predict_all(*clf, samples);
        int correct = 0;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            if (preds[i].label == samples[i].label) {
                ++correct;
            }
        }
        const double accuracy =
            static_cast<double>(correct) / static_cast<double>(samples.size());

        std::vector<BenchmarkResult> per_method = sweep_classifier(
            *clf, samples, cfg_.targets, cfg_.warmup, cfg_.iterations, accuracy);
        results.insert(results.end(), per_method.begin(), per_method.end());
    }

    return results;
}

} // namespace eib
