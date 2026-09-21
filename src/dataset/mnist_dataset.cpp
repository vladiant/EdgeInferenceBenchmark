#include "eib/dataset.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace eib {
namespace {

std::uint32_t read_be_u32(std::istream& in) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (in.gcount() != 4) {
        throw std::runtime_error("IDX: unexpected end of file while reading header");
    }
    return (static_cast<std::uint32_t>(b[0]) << 24) |
           (static_cast<std::uint32_t>(b[1]) << 16) |
           (static_cast<std::uint32_t>(b[2]) << 8) |
           static_cast<std::uint32_t>(b[3]);
}

std::ifstream open_or_throw(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("IDX: cannot open file '" + path + "'");
    }
    return in;
}

} // namespace

MnistDataset::MnistDataset(std::string images_idx_path,
                           std::string labels_idx_path,
                           int max_items)
    : images_path_(std::move(images_idx_path)),
      labels_path_(std::move(labels_idx_path)),
      max_items_(max_items) {}

std::vector<Sample> MnistDataset::load() const {
    std::ifstream images = open_or_throw(images_path_);
    std::ifstream labels = open_or_throw(labels_path_);

    const std::uint32_t image_magic = read_be_u32(images);
    if (image_magic != 0x00000803u) {
        throw std::runtime_error("IDX: bad image magic number in '" + images_path_ + "'");
    }
    const std::uint32_t label_magic = read_be_u32(labels);
    if (label_magic != 0x00000801u) {
        throw std::runtime_error("IDX: bad label magic number in '" + labels_path_ + "'");
    }

    const std::uint32_t image_count = read_be_u32(images);
    const std::uint32_t rows = read_be_u32(images);
    const std::uint32_t cols = read_be_u32(images);
    const std::uint32_t label_count = read_be_u32(labels);

    if (image_count != label_count) {
        throw std::runtime_error("IDX: image/label count mismatch");
    }
    if (rows != 28 || cols != 28) {
        throw std::runtime_error("IDX: expected 28x28 images, got " +
                                 std::to_string(rows) + "x" + std::to_string(cols));
    }

    std::uint32_t wanted = image_count;
    if (max_items_ >= 0 && static_cast<std::uint32_t>(max_items_) < wanted) {
        wanted = static_cast<std::uint32_t>(max_items_);
    }

    const int pixels = static_cast<int>(rows * cols);
    std::vector<Sample> samples;
    samples.reserve(wanted);

    for (std::uint32_t i = 0; i < wanted; ++i) {
        cv::Mat image(static_cast<int>(rows), static_cast<int>(cols), CV_8UC1);
        images.read(reinterpret_cast<char*>(image.data), pixels);
        if (images.gcount() != pixels) {
            throw std::runtime_error("IDX: truncated image data in '" + images_path_ + "'");
        }

        unsigned char label_byte = 0;
        labels.read(reinterpret_cast<char*>(&label_byte), 1);
        if (labels.gcount() != 1) {
            throw std::runtime_error("IDX: truncated label data in '" + labels_path_ + "'");
        }
        if (label_byte > 9) {
            throw std::runtime_error("IDX: label out of range 0..9");
        }

        samples.push_back(Sample{std::move(image), static_cast<int>(label_byte)});
    }

    return samples;
}

} // namespace eib
