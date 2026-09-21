#include "eib/classifier_factory.hpp"

#include <stdexcept>

#include "eib/classical_classifier.hpp"
#include "eib/cnn_classifier.hpp"

namespace eib {

std::unique_ptr<IClassifier> make_classifier(Method method,
                                             const BenchmarkConfig& cfg) {
    switch (method) {
    case Method::Classical: {
        ClassicalConfig cc;
        cc.model_path = cfg.svm_model_path;
        return std::make_unique<ClassicalClassifier>(cc);
    }
    case Method::Cnn: {
        CnnConfig nc;
        nc.onnx_path = cfg.onnx_model_path;
        return std::make_unique<CnnClassifier>(nc);
    }
    }
    throw std::runtime_error("make_classifier: unknown method");
}

} // namespace eib
