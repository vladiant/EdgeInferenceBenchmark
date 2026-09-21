#include "eib/thread_scope.hpp"

namespace eib {

ThreadScope::ThreadScope(int num_threads)
    : prev_(cv::getNumThreads()), effective_(0) {
    // 0 => OpenCV default (all cores); OpenCV uses -1 to mean "reset".
    cv::setNumThreads(num_threads == 0 ? -1 : num_threads);
    effective_ = cv::getNumThreads();
}

ThreadScope::~ThreadScope() {
    cv::setNumThreads(prev_);
}

} // namespace eib
