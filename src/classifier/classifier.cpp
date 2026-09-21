#include "eib/classifier.hpp"

namespace eib {

std::vector<Prediction> predict_all(const IClassifier& clf,
                                    const std::vector<Sample>& samples) {
    std::vector<Prediction> preds;
    preds.reserve(samples.size());
    for (const Sample& s : samples) {
        preds.push_back(clf.predict(s.image));
    }
    return preds;
}

} // namespace eib
