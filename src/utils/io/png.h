#ifndef INCLUDED_UTILS_PNG_FILE_FORMAT
#define INCLUDED_UTILS_PNG_FILE_FORMAT

#include <cstdint>
#include <tuple>
#include <iosfwd>
#include <filesystem>

#include <png.h>

#include "utils/image.h"

namespace PNG {
    class IStream {
        png_struct* png_ptr;
        png_info* info_ptr;

    public:
        IStream(const IStream&) = delete;
        IStream& operator =(const IStream&) = delete;

        IStream(std::istream& file);
        ~IStream();

        operator const png_struct*() const { return png_ptr; }
        operator png_struct*() { return png_ptr; }

        operator const png_info*() const { return info_ptr; }
        operator png_info*() { return info_ptr; }
    };

    class OStream {
        png_struct* png_ptr;
        png_info* info_ptr;

    public:
        OStream(const OStream&) = delete;
        OStream& operator =(const OStream&) = delete;

        OStream(std::ostream& file);
        ~OStream();

        operator const png_struct*() const { return png_ptr; }
        operator png_struct*() { return png_ptr; }

        operator const png_info*() const { return info_ptr; }
        operator png_info*() { return info_ptr; }
    };

    class R8G8B8OStream : public OStream {
    public:
        R8G8B8OStream(std::ostream& file, png_uint_32 width, png_uint_32 height);
        ~R8G8B8OStream();

        void writeRow(const uint32_t* row);
    };

    class R8G8B8A8OStream : public OStream {
    public:
        R8G8B8A8OStream(std::ostream& file, png_uint_32 width, png_uint_32 height);
        ~R8G8B8A8OStream();

        void writeRow(const uint32_t* row);
    };

    std::tuple<int, int> readImageSize(std::istream& file);
    std::tuple<int, int> readImageSize(const std::filesystem::path& filename);

    image2D<uint32_t> loadImage2DR8G8B8A8(std::istream& file);
    image2D<uint32_t> loadImage2DR8G8B8A8(const std::filesystem::path& filename);

    std::ostream& saveImageR8G8B8(std::ostream& file, const image2D<uint32_t>& surface);
    inline std::ostream&& saveImageR8G8B8(std::ostream&& file, const image2D<uint32_t>& surface) { return std::move(saveImageR8G8B8(file, surface)); }
    void saveImageR8G8B8(const std::filesystem::path& filename, const image2D<uint32_t>& surface);
    std::ostream& saveImageR8G8B8A8(std::ostream& file, const image2D<uint32_t>& surface);
    inline std::ostream&& saveImageR8G8B8A8(std::ostream&& file, const image2D<uint32_t>& surface) { return std::move(saveImageR8G8B8A8(file, surface)); }
    void saveImageR8G8B8A8(const std::filesystem::path& filename, const image2D<uint32_t>& surface);
}

#endif // INCLUDED_UTILS_PNG_FILE_FORMAT
