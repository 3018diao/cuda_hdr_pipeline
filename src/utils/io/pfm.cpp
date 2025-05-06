#include <memory>
#include <fstream>
#include <string>
#include <string_view>

#include "utils/io.h"
#include "pfm.h"

using namespace std::literals;

namespace {
    constexpr bool is_ws(char c) {
        return c == ' ' || c == '\t' || c == '\r' || c == '\n';
    }

    std::istream& skip_ws(std::istream& in) {
        while (is_ws(in.peek())) in.get();
        return in;
    }

    std::istream& skip_line(std::istream& in) {
        while (in.peek() != '\n') in.get();
        if (in.peek() == '\n') in.get();
        return in;
    }

    std::istream& skip_comments(std::istream& in) {
        while ((in >> skip_ws).peek() == '#')
            in >> skip_line;
        return in;
    }

    ImageIO::Sink& read_image(ImageIO::Sink& sink, std::istream& file) {
        std::string magic;
        if (!(file >> skip_comments >> magic))
            throw PFM::error("error reading PFM header");

        if (magic.length() != 2 || magic[0] != 'P' || (magic[1] != 'f' && magic[1] != 'F'))
            throw PFM::error("unsupported file format");

        int channels = magic[1] == 'f' ? 1 : 3;

        std::size_t w;
        std::size_t h;
        float a;
        if (!(file >> skip_comments >> w >> skip_comments >> h >> skip_comments >> a) || file.get() != '\n')
            throw PFM::error("error parsing PFM header");

        if (w == 0 || h == 0)
            throw PFM::error("PFM image cannot be empty");

        if (a >= 0.0f)
            throw PFM::error("only little-endian PFM supported");

        auto& image_sink = channels == 1 ? sink.accept_R32F(w, h) : sink.accept_RGB32F(w, h);

        auto row = std::unique_ptr<float[]> { new float[w * channels] };

        for (std::size_t j = 0; j < h; ++j)
        {
            ::read(&row[0], file, w * channels);
            image_sink.accept_row(&row[0], h - j - 1);
        }

        return sink;
    }

    template <typename T>
    std::ostream& save(std::ostream& file, const image2D<T>& img, std::string_view type) {
        auto w = width(img);
        auto h = height(img);

        file << type << '\n'
             << w << ' ' << h << '\n'
             << -1.0f << '\n';

        for (std::size_t j = 0; j < h; ++j)
            write(file, data(img) + w * (h - j - 1), w);

        return file;
    }
}

namespace PFM {
    ImageIO::Sink& load(ImageIO::Sink& sink, const std::filesystem::path& filename) {
        std::ifstream file(filename, std::ios::binary);

        if (!file)
            throw error("failed to open file '" + filename.string() + '\'');

        return read_image(sink, file);
    }

    void saveR32F(const std::filesystem::path& filename, const image2D<float>& img) {
        std::ofstream file(filename, std::ios::binary);

        if (!file)
            throw error("failed to open file '" + filename.string() + '\'');

        ::save(file, img, "Pf"sv);
    }

    void saveRGB32F(const std::filesystem::path& filename, const image2D<std::array<float, 3>>& img) {
        std::ofstream file(filename, std::ios::binary);

        if (!file)
            throw error("failed to open file '" + filename.string() + '\'');

        ::save(file, img, "PF"sv);
    }
}
