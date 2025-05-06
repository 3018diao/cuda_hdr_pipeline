#include <memory>
#include <string>
#include <stdexcept>
#include <fstream>
#include <iostream>

#include "png.h"

using namespace std::literals;

namespace {
    void error_fn(png_structp png_ptr, png_const_charp error_msg) {
        throw std::runtime_error(error_msg);
    }

    void warning_fn(png_structp png_ptr, png_const_charp warning_msg) {
    }

    void read_fn(png_structp png_ptr, png_bytep data, png_size_t length) {
        auto& file = *static_cast<std::istream*>(png_get_io_ptr(png_ptr));
        file.read(reinterpret_cast<char*>(data), length);
    }

    static void write_fn(png_structp png_ptr, png_bytep data, png_size_t length) {
        auto& file = *static_cast<std::ostream*>(png_get_io_ptr(png_ptr));
        file.write(reinterpret_cast<char*>(data), length);
    }

    static void flush_fn(png_structp png_ptr) {
        auto& file = *static_cast<std::ostream*>(png_get_io_ptr(png_ptr));
        file.flush();
    }
}

namespace PNG {
    IStream::IStream(std::istream& file) {
        png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, &error_fn, &warning_fn);
        png_set_read_fn(png_ptr, &file, &read_fn);

        if (!png_ptr)
            throw std::runtime_error("png_create_read_struct() failed");

        png_set_read_fn(png_ptr, &file, read_fn);

        info_ptr = png_create_info_struct(png_ptr);

        if (!info_ptr)
            throw std::runtime_error("png_create_info_struct() failed");

        png_read_info(png_ptr, info_ptr);
    }

    IStream::~IStream() {
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    }


    OStream::OStream(std::ostream& file) {
        png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, &error_fn, &warning_fn);

        if (!png_ptr)
            throw std::runtime_error("png_create_write_struct() failed");

        png_set_write_fn(png_ptr, &file, &write_fn, &flush_fn);

        info_ptr = png_create_info_struct(png_ptr);

        if (!info_ptr)
            throw std::runtime_error("png_create_info_struct() failed");
    }

    OStream::~OStream() {
        png_destroy_write_struct(&png_ptr, &info_ptr);
    }


    R8G8B8OStream::R8G8B8OStream(std::ostream& file, png_uint_32 width, png_uint_32 height)
        : OStream(file) {
        png_set_IHDR(*this, *this, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

        //png_set_sRGB_gAMA_and_cHRM(*this, *this, 0);

        png_write_info(*this, *this);

        png_set_filler(*this, 0U, PNG_FILLER_AFTER);
    }

    R8G8B8OStream::~R8G8B8OStream() {
        png_write_end(*this, *this);
    }

    void R8G8B8OStream::writeRow(const uint32_t* row) {
        png_write_row(*this, reinterpret_cast<const png_byte*>(row));
    }


    R8G8B8A8OStream::R8G8B8A8OStream(std::ostream& file, png_uint_32 width, png_uint_32 height)
        : OStream(file) {
        png_set_IHDR(*this, *this, width, height, 8, PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

        //png_set_sRGB_gAMA_and_cHRM(*this, *this, 0);

        png_write_info(*this, *this);
    }

    R8G8B8A8OStream::~R8G8B8A8OStream() {
        png_write_end(*this, *this);
    }

    void R8G8B8A8OStream::writeRow(const uint32_t* row) {
        png_write_row(*this, reinterpret_cast<const png_byte*>(row));
    }


    std::tuple<int, int> readImageSize(std::istream& file) {
        IStream png(file);
        png_uint_32 w, h;
        int bit_depth, color_type, interlace_method, compression_method, filter_method;
        png_get_IHDR(png, png, &w, &h, &bit_depth, &color_type, &interlace_method, &compression_method, &filter_method);

        return std::tuple<int, int>(w, h);
    }

    std::tuple<int, int> readImageSize(const std::filesystem::path& filename) {
        std::ifstream file(filename, std::ios::binary);

        if (!file)
            throw std::runtime_error("unable to open '"s + filename.string() + "'");

        return readImageSize(file);
    }


    image2D<uint32_t> loadImage2DR8G8B8A8(std::istream& file) {
        IStream png(file);

        png_uint_32 w, h;
        int bit_depth, color_type, interlace_method, compression_method, filter_method;
        png_get_IHDR(png, png, &w, &h, &bit_depth, &color_type, &interlace_method, &compression_method, &filter_method);

        if (bit_depth > 8)
            png_set_scale_16(png);
        else if (bit_depth < 8)
            png_set_packing(png);

        if (color_type == PNG_COLOR_TYPE_RGB) {
            png_set_add_alpha(png, 0xFF, PNG_FILLER_AFTER);
        } else if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
            png_set_gray_to_rgb(png);
            png_set_add_alpha(png, 0xFF, PNG_FILLER_AFTER);
        } else if (color_type == PNG_COLOR_TYPE_PALETTE) {
            png_set_palette_to_rgb(png);
            png_set_add_alpha(png, 0xFF, PNG_FILLER_AFTER);
        } else if (color_type == PNG_COLOR_TYPE_RGB_ALPHA)
            ;
        else
            throw std::runtime_error("unsupported color type");

        png_read_update_info(png, png);

        image2D<uint32_t> img(w, h);

        auto rows = std::unique_ptr<png_byte*[]>(new png_byte*[h]);
        for (png_uint_32 y = 0; y < h; ++y)
            rows[y] = reinterpret_cast<png_byte*>(data(img) + y * w);

        png_read_image(png, &rows[0]);

        png_read_end(png, png);

        return img;
    }

    image2D<uint32_t> loadImage2DR8G8B8A8(const std::filesystem::path& filename) {
        std::ifstream file(filename, std::ios::binary);

        if (!file)
            throw std::runtime_error("unable to open '"s + filename.string() + "'");

        return loadImage2DR8G8B8A8(file);
    }


    std::ostream& saveImageR8G8B8(std::ostream& file, const image2D<uint32_t>& img) {
        auto w = static_cast<png_uint_32>(width(img));
        auto h = static_cast<png_uint_32>(height(img));

        R8G8B8OStream png(file, w, h);

        for (unsigned int y = 0; y < h; ++y)
            png_write_row(png, reinterpret_cast<const png_byte*>(data(img) + y * w));

        return file;
    }

    void saveImageR8G8B8(const std::filesystem::path& filename, const image2D<uint32_t>& img) {
        std::ofstream file(filename, std::ios::binary);

        if (!file)
            throw std::runtime_error("unable to open '"s + filename.string() + "'");

        saveImageR8G8B8(file, img);
    }

    std::ostream& saveImageR8G8B8A8(std::ostream& file, const image2D<uint32_t>& img) {
        auto w = static_cast<png_uint_32>(width(img));
        auto h = static_cast<png_uint_32>(height(img));

        R8G8B8A8OStream png(file, w, h);

        for (unsigned int y = 0; y < h; ++y)
            png_write_row(png, reinterpret_cast<const png_byte*>(data(img) + y * w));

        return file;
    }

    void saveImageR8G8B8A8(const std::filesystem::path& filename, const image2D<uint32_t>& img) {
        std::ofstream file(filename, std::ios::binary);

        if (!file)
            throw std::runtime_error("unable to open '"s + filename.string() + "'");

        saveImageR8G8B8A8(file, img);
    }
}
