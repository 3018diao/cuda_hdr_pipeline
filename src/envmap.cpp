#include <cassert>
#include <utility>
#include <optional>
#include <variant>
#include <stdexcept>

#include "envmap.h"
#include "utils/io/radiance.h"
#include "utils/io/pfm.h"


namespace {
    template <bool flip>
    class image_sink : public ImageIO::Sink, ImageIO::Sink::ImageSink {
        std::optional<image2D<std::array<float, 4>>> image;

        void accept_row(const float* row, std::size_t j) override {
            for (std::size_t i = 0; i < width(*image); ++i)
                (*image)(i, flip ? height(*image) - j - 1 : j) = { row[3 * i + 0], row[3 * i + 1], row[3 * i + 2], 0.0f };
        }

        ImageIO::Sink::ImageSink& accept_R32F(std::size_t width, std::size_t height) override {
            throw std::runtime_error("environment map must be an RGB image");
        }

        ImageIO::Sink::ImageSink& accept_RGB32F(std::size_t width, std::size_t height) override {
            image.emplace(width, height);
            return *this;
        }

    public:
        image2D<std::array<float, 4>> finish() {
            assert(image);
            return std::move(*image);
        }
    };
}

image2D<std::array<float, 4>> load_envmap(const std::filesystem::path& filename, bool flip) {
    std::variant<image_sink<false>, image_sink<true>> sink;

    if (flip)
        sink.emplace<image_sink<true>>();
    else
        sink.emplace<image_sink<false>>();

    if (filename.extension() == ".pfm")
        PFM::load(std::visit([](auto&& a) -> ImageIO::Sink& { return a; }, sink), filename);
    else if (filename.extension() == ".hdr")
        Radiance::load(std::visit([](auto&& a) -> ImageIO::Sink& { return a; }, sink), filename);
    else
        throw std::runtime_error("unknown file extension '" + filename.extension().string() + '\'');

    return std::visit([](auto&& a) { return a.finish(); }, sink);
}
