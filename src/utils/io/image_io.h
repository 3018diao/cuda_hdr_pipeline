#ifndef INCLUDED_UTILS_IMAGE_IO
#define INCLUDED_UTILS_IMAGE_IO

#include <cstddef>

namespace ImageIO {
    struct Sink {
        struct ImageSink {
            virtual void accept_row(const float* row, std::size_t j) = 0;

        protected:
            ImageSink() = default;
            ImageSink(const ImageSink&) = default;
            ImageSink(ImageSink&&) = default;
            ImageSink& operator =(const ImageSink&) = default;
            ImageSink& operator =(ImageSink&&) = default;
            ~ImageSink() = default;
        };

        virtual ImageSink& accept_R32F(std::size_t width, std::size_t height) = 0;
        virtual ImageSink& accept_RGB32F(std::size_t width, std::size_t height) = 0;

    protected:
        Sink() = default;
        Sink(const Sink&) = default;
        Sink(Sink&&) = default;
        Sink& operator =(const Sink&) = default;
        Sink& operator =(Sink&&) = default;
        ~Sink() = default;
    };
}

#endif // INCLUDED_UTILS_IMAGE_IO
