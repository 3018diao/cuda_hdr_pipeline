#ifndef INCLUDED_UTILS_IMAGE
#define INCLUDED_UTILS_IMAGE

#include <limits>
#include <type_traits>
#include <cassert>
#include <cstddef>
#include <memory>
#include <algorithm>

template <typename T>
class image2D {
    std::unique_ptr<T[]> img;

    std::size_t width;
    std::size_t height;

    static constexpr std::size_t size(std::size_t width, std::size_t height) noexcept {
        return width * height;
    }

    static auto alloc(std::size_t width, std::size_t height) {
        return std::unique_ptr<T[]> { new T[size(width, height)] };
    }

    static constexpr T* address(T* img, std::size_t width, std::size_t x, std::size_t y) noexcept {
        return img + y * width + x;
    }

public:
    friend std::size_t width(const image2D& img) noexcept {
        return img.width;
    }

    friend std::size_t height(const image2D& img) noexcept {
        return img.height;
    }

    friend const T* data(const image2D& img) noexcept {
        return &img.img[0];
    }

    friend T* data(image2D& img) noexcept {
        return &img.img[0];
    }

    friend constexpr std::size_t pitch(const image2D& img) noexcept {
        return img.width;
    }

    friend auto cbegin(const image2D& img) noexcept {
        return data(img);
    }

    friend auto begin(const image2D& img) noexcept {
        return data(img);
    }

    friend auto begin(image2D& img) noexcept {
        return data(img);
    }

    friend auto cend(const image2D& img) noexcept {
        return begin(img) + size(img.width, img.height);
    }

    friend auto end(const image2D& img) noexcept {
        return cend(img);
    }

    friend auto end(image2D& img) noexcept {
        return data(img) + size(img.width, img.height);
    }

    image2D(std::size_t width, std::size_t height)
        : img(alloc(width, height)),
          width(width),
          height(height) {
    }

    image2D(const image2D& s)
        : img(alloc(s.width, s.height)),
          width(s.width),
          height(s.height) {
        std::copy(begin(s), end(s), &img[0]);
    }

    image2D(image2D&& s) = default;

    image2D& operator =(const image2D& s) {
        width = s.width;
        height = s.height;
        auto buffer = alloc(width, height);
        std::copy(begin(s), end(s), &buffer[0]);
        img = move(buffer);
        return *this;
    }

    image2D& operator =(image2D&& s) = default;

    const T& operator ()(std::size_t x, std::size_t y) const noexcept { return *address(&img[0], width, x, y); }
    T& operator ()(std::size_t x, std::size_t y) noexcept { return *address(&img[0], width, x, y); }
};

#endif // INCLUDED_UTILS_IMAGE
