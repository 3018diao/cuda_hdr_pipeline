#ifndef INCLUDED_ENVMAP
#define INCLUDED_ENVMAP

#include <filesystem>
#include <array>

#include "utils/image.h"

image2D<std::array<float, 4>> load_envmap(const std::filesystem::path& filename, bool flip = false);

#endif // INCLUDED_ENVMAP
