#pragma once

#include <string>
#include <vector>

#include "eib/types.hpp"

namespace eib {

/// Abstract dataset source producing a vector of labelled samples (FR-1).
class IDatasetLoader {
public:
    virtual ~IDatasetLoader() = default;
    virtual std::vector<Sample> load() const = 0;
};

/// Loads the standard MNIST IDX3 (images) and IDX1 (labels) files.
///
/// Validates magic numbers (0x00000803 images, 0x00000801 labels), reads the
/// big-endian headers, and verifies image/label counts and 28x28 dimensions.
/// Throws std::runtime_error on any missing or malformed file (FR-14).
class MnistDataset final : public IDatasetLoader {
public:
    MnistDataset(std::string images_idx_path,
                 std::string labels_idx_path,
                 int max_items = -1); // -1 = all available

    std::vector<Sample> load() const override;

private:
    std::string images_path_;
    std::string labels_path_;
    int max_items_;
};

} // namespace eib
