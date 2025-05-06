#ifndef INCLUDED_UTILS_RADIANCE_FILE_FORMAT
#define INCLUDED_UTILS_RADIANCE_FILE_FORMAT

#include <filesystem>
#include <stdexcept>

#include "image_io.h"

namespace Radiance {
    struct error : std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    ImageIO::Sink& load(ImageIO::Sink& sink, const std::filesystem::path& filename);
}

#endif // INCLUDED_UTILS_RADIANCE_FILE_FORMAT
