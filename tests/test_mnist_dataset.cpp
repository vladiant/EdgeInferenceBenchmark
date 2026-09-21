#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "eib/dataset.hpp"

namespace {

std::string temp_path(const std::string& name) {
    return (::testing::TempDir() + name);
}

void write_be_u32(std::ofstream& out, std::uint32_t v) {
    const unsigned char b[4] = {
        static_cast<unsigned char>((v >> 24) & 0xFF),
        static_cast<unsigned char>((v >> 16) & 0xFF),
        static_cast<unsigned char>((v >> 8) & 0xFF),
        static_cast<unsigned char>(v & 0xFF)};
    out.write(reinterpret_cast<const char*>(b), 4);
}

// Writes a valid IDX3 image file with `count` 28x28 images (pixel value = i%256)
// and a matching IDX1 label file (label = i%10).
void write_valid_fixture(const std::string& img_path, const std::string& lbl_path,
                         int count) {
    std::ofstream img(img_path, std::ios::binary);
    write_be_u32(img, 0x00000803);
    write_be_u32(img, static_cast<std::uint32_t>(count));
    write_be_u32(img, 28);
    write_be_u32(img, 28);
    for (int i = 0; i < count; ++i) {
        std::vector<unsigned char> pixels(28 * 28,
                                          static_cast<unsigned char>(i % 256));
        img.write(reinterpret_cast<const char*>(pixels.data()),
                  static_cast<std::streamsize>(pixels.size()));
    }

    std::ofstream lbl(lbl_path, std::ios::binary);
    write_be_u32(lbl, 0x00000801);
    write_be_u32(lbl, static_cast<std::uint32_t>(count));
    for (int i = 0; i < count; ++i) {
        const unsigned char label = static_cast<unsigned char>(i % 10);
        lbl.write(reinterpret_cast<const char*>(&label), 1);
    }
}

} // namespace

TEST(MnistDataset, LoadsValidFixture) {
    const std::string img = temp_path("eib_valid_img.idx3");
    const std::string lbl = temp_path("eib_valid_lbl.idx1");
    write_valid_fixture(img, lbl, 5);

    eib::MnistDataset ds(img, lbl);
    const std::vector<eib::Sample> samples = ds.load();

    ASSERT_EQ(samples.size(), 5u);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(samples[i].label, i % 10);
        EXPECT_EQ(samples[i].image.rows, 28);
        EXPECT_EQ(samples[i].image.cols, 28);
        EXPECT_EQ(samples[i].image.type(), CV_8UC1);
        EXPECT_EQ(samples[i].image.at<unsigned char>(0, 0),
                  static_cast<unsigned char>(i % 256));
    }
}

TEST(MnistDataset, MaxItemsCapsCount) {
    const std::string img = temp_path("eib_cap_img.idx3");
    const std::string lbl = temp_path("eib_cap_lbl.idx1");
    write_valid_fixture(img, lbl, 10);

    eib::MnistDataset ds(img, lbl, /*max_items=*/3);
    EXPECT_EQ(ds.load().size(), 3u);
}

TEST(MnistDataset, MissingFileThrows) {
    eib::MnistDataset ds("/nonexistent/img.idx3", "/nonexistent/lbl.idx1");
    EXPECT_THROW(ds.load(), std::runtime_error);
}

TEST(MnistDataset, BadMagicThrows) {
    const std::string img = temp_path("eib_badmagic_img.idx3");
    const std::string lbl = temp_path("eib_badmagic_lbl.idx1");
    {
        std::ofstream i(img, std::ios::binary);
        write_be_u32(i, 0x00000000); // wrong magic
        write_be_u32(i, 1);
        write_be_u32(i, 28);
        write_be_u32(i, 28);
        std::vector<unsigned char> pixels(28 * 28, 0);
        i.write(reinterpret_cast<const char*>(pixels.data()), 28 * 28);

        std::ofstream l(lbl, std::ios::binary);
        write_be_u32(l, 0x00000801);
        write_be_u32(l, 1);
        const unsigned char label = 0;
        l.write(reinterpret_cast<const char*>(&label), 1);
    }
    eib::MnistDataset ds(img, lbl);
    EXPECT_THROW(ds.load(), std::runtime_error);
}

TEST(MnistDataset, CountMismatchThrows) {
    const std::string img = temp_path("eib_mismatch_img.idx3");
    const std::string lbl = temp_path("eib_mismatch_lbl.idx1");
    {
        std::ofstream i(img, std::ios::binary);
        write_be_u32(i, 0x00000803);
        write_be_u32(i, 2); // two images
        write_be_u32(i, 28);
        write_be_u32(i, 28);
        std::vector<unsigned char> pixels(2 * 28 * 28, 0);
        i.write(reinterpret_cast<const char*>(pixels.data()), 2 * 28 * 28);

        std::ofstream l(lbl, std::ios::binary);
        write_be_u32(l, 0x00000801);
        write_be_u32(l, 1); // one label
        const unsigned char label = 0;
        l.write(reinterpret_cast<const char*>(&label), 1);
    }
    eib::MnistDataset ds(img, lbl);
    EXPECT_THROW(ds.load(), std::runtime_error);
}

TEST(MnistDataset, WrongDimensionsThrow) {
    const std::string img = temp_path("eib_dim_img.idx3");
    const std::string lbl = temp_path("eib_dim_lbl.idx1");
    {
        std::ofstream i(img, std::ios::binary);
        write_be_u32(i, 0x00000803);
        write_be_u32(i, 1);
        write_be_u32(i, 20); // not 28
        write_be_u32(i, 20);
        std::vector<unsigned char> pixels(20 * 20, 0);
        i.write(reinterpret_cast<const char*>(pixels.data()), 20 * 20);

        std::ofstream l(lbl, std::ios::binary);
        write_be_u32(l, 0x00000801);
        write_be_u32(l, 1);
        const unsigned char label = 0;
        l.write(reinterpret_cast<const char*>(&label), 1);
    }
    eib::MnistDataset ds(img, lbl);
    EXPECT_THROW(ds.load(), std::runtime_error);
}
