#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>

#include "utils/io.h"

#include "radiance.h"

using namespace std::literals;

namespace {
    constexpr bool is_horz_ws(char c) {
        return c == ' ' || c == '\t' || c == '\r';
    }

    constexpr bool is_ws(char c) {
        return is_horz_ws(c) || c == '\n';
    }

    std::istream& skip_horz_ws(std::istream& in) {
        while (is_horz_ws(in.peek())) in.get();
        return in;
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

    bool expect_magic(std::istream& file) {
        constexpr auto magic = "#?RADIANCE\n"sv;
        char buffer[magic.length()];
        return file.read(buffer, magic.length()) && std::string_view(buffer, std::size(buffer)) == magic;
    }

    template <typename Callback>
    std::istream& parse_header(Callback&& callback, std::istream& file) {
        if (!expect_magic(file))
            throw Radiance::error("not a valid Radiance file");

        while (file.peek() != '\n') {
            std::string line;
            if (!std::getline(file >> skip_comments, line))
                throw std::runtime_error("expected Radiance header entry");

            auto key_end = line.find('=');

            if (key_end == std::string::npos)
                throw std::runtime_error("expected '='");

            auto key = std::string_view(line).substr(0, key_end);
            auto value = std::string_view(line).substr(key_end + 1);

            callback(key, value);
        }

        file.get();

        return file;
    }

    auto parse_resolution_string_component(std::istream& file) {
        char sign = (file >> skip_horz_ws).get();
        if (!file)
            throw std::runtime_error("expected pixel order");

        if (sign != '+' && sign != '-')
            throw std::runtime_error("pixel order must be either '+' or '-'");

        char axis = (file >> skip_horz_ws).get();
        if (!file)
            throw std::runtime_error("expected pixel order");

        if (axis != 'X' && axis != 'Y')
            throw std::runtime_error("axis must be either 'X' or 'Y'");

        int res;
        if (!(file >> res))
            throw std::runtime_error("expected resolution");

        struct component {
            int sign;
            int axis;
            int N;
        };

        return component { sign == '-' ? -1 : 1, axis == 'X' ? 0 : 1, res };
    }

    auto parse_resolution(std::istream& file) {
        auto [sign_1, axis_1, res_1] = parse_resolution_string_component(file);
        auto [sign_2, axis_2, res_2] = parse_resolution_string_component(file);

        if (file.get() != '\n')

        if (sign_1 != -1 || axis_1 != 1 || sign_2 != 1 || axis_2 != 0)
            throw std::runtime_error("unsupported image orientation");

        struct resolution {
            int x, y;
        };

        return resolution { res_2, res_1 };
    }

    class scanline_decoder {
        std::size_t width;
        std::unique_ptr<float[]> buffer;

        template <typename Sink>
        void decode_new_rle(Sink&& sink, std::istream& file, int N) {
            while (N > 0) {
                unsigned int run = file.get();

                if (run <= 128) {
                    while (run-- > 0) {
                        const unsigned char v = file.get();
                        sink(v);
                        --N;
                    }
                } else {
                    const unsigned char v = file.get();

                    while (run-- > 128) {
                        sink(v);
                        --N;
                    }
                }
            }

            if (!file)
                throw std::runtime_error("failed to read scanline data");
        }

        void decode_row_new_rle(std::istream& file, int N) {
            if (N != width)
                throw std::runtime_error("scanline length must match image width");

            for (int c = 0; c < 3; ++c) {
                decode_new_rle([dest = &buffer[0] + c](auto v) mutable {
                    *dest = v; dest += 3;
                }, file, N);
            }

            decode_new_rle([data = &buffer[0]](unsigned char e) mutable {
                const float f = std::exp2(e - (128.0f + 8.0f));
                *data++ *= f;
                *data++ *= f;
                *data++ *= f;
            }, file, N);

            return;
        }

    public:
        scanline_decoder(std::size_t width)
            : width(width), buffer(new float[width * 3]) {
        }

        const float* operator ()(std::istream& file) {
            if (unsigned char r = file.get(), g = file.get(), b = file.get(), e = file.get(); file) {
                if (r == 2 && g == 2 && b < 128) {
                    decode_row_new_rle(file, b * 256 + e);
                    return &buffer[0];
                }

                throw std::runtime_error("old RLE not supported");
            } else
                throw std::runtime_error("failed to read scanline data");
        }
    };

    ImageIO::Sink& read_image(ImageIO::Sink& sink, std::istream& file) {
        struct {
            bool rle = false;
            float exposure = 1.0f;

            void operator ()(std::string_view key, std::string_view value) {
                if (key == "FORMAT"sv) {
                    if (value == "32-bit_rle_rgbe"sv)
                        rle = true;
                    else
                        throw std::runtime_error("Radiance FORMAT must be 32-bit_rle_rgbe");
                } else if (key == "EXPOSURE"sv) {
                    // TODO: replace this with std::from_chars once GCC/clang support is available
                    class string_view_streambuffer : protected std::streambuf {
                    public:
                        string_view_streambuffer(std::string_view str) {
                            auto begin = const_cast<char*>(&str[0]);
                            std::streambuf::setg(begin, begin, begin + str.length());
                        }
                    };

                    class string_view_stream : string_view_streambuffer, public std::istream {
                    public:
                        string_view_stream(std::string_view str)
                            : string_view_streambuffer(str), std::istream(this) {
                        }
                    };

                    string_view_stream stream(value);

                    if (!(stream >> exposure))
                        throw std::runtime_error("expected exposure value");
                } else if (key == "SOFTWARE"sv) {
                    // ¯\_(ツ)_/¯
                } else {
                    std::cerr << "WARNING: Radiance header entry '" << key << '=' << value << "' ignored\n";
                }
            }
        } params;

        parse_header(params, file);

        [[maybe_unused]] auto [width, height] = parse_resolution(file);

        auto& image_sink = sink.accept_RGB32F(width, height);

        scanline_decoder decoder(width);

        for (std::size_t j = 0; j < height; ++j)
            image_sink.accept_row(decoder(file), j);

        return sink;
    }
}

namespace Radiance {
    ImageIO::Sink& load(ImageIO::Sink& sink, const std::filesystem::path& filename) {
        std::ifstream file(filename, std::ios::binary);

        if (!file)
            throw error("failed to open file '" + filename.string() + '\'');

        return read_image(sink, file);
    }
}
