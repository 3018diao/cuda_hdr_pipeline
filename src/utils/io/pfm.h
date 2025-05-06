#ifndef INCLUDED_UTILS_PFM_FILE_FORMAT
#define INCLUDED_UTILS_PFM_FILE_FORMAT

#include <array>
#include <cstddef>
#include <filesystem>
#include <stdexcept>

#include "image_io.h"
#include "utils/image.h"

namespace PFM {
    struct error : std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    ImageIO::Sink& load(ImageIO::Sink& sink, const std::filesystem::path& filename);

    void saveR32F(const std::filesystem::path& filename, const image2D<float>& image);
    void saveRGB32F(const std::filesystem::path& filename, const image2D<std::array<float, 3>>& image);
}

#endif // INCLUDED_UTILS_PFM_FILE_FORMAT
