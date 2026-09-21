#pragma once

#include <opencv2/core/utility.hpp>

namespace eib {

/// RAII guard that sets the OpenCV thread count for the lifetime of a target
/// sweep and restores the previous value on scope exit (TR-1, TR-3, NFR-8).
class ThreadScope {
public:
    /// num_threads: 1 => single-threaded; 0 => OpenCV default (all cores).
    explicit ThreadScope(int num_threads);
    ~ThreadScope();

    ThreadScope(const ThreadScope&) = delete;
    ThreadScope& operator=(const ThreadScope&) = delete;

    /// Effective thread count OpenCV reports after the scope was applied.
    int effective_threads() const { return effective_; }

private:
    int prev_;
    int effective_;
};

} // namespace eib
