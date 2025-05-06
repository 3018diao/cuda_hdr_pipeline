#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string_view>

#include <utils/argparse.h>
#include <utils/io/png.h>

using namespace std::literals;

namespace {
    std::ostream& print_usage(std::ostream& out) {
        return out << R"""(usage: imgdiff [{options}] <image> <reference>
    options:
      --o <filename>       output file name
       -t <threshold>      threshold for inclusion in diff image. default: 4
)""";
    }

    template <int i> std::uint_fast8_t channel(std::uint_fast32_t value) { return (value >> (i * 8)) & 0xFFU; }
    
    template <int i, typename A, typename B>
    float reldiff(A&& a, B&& b) {
        auto ref = channel<i>(b);
        float delta = std::abs(channel<i>(a) - ref);
        return ref != 0 ? delta / ref : delta;
    }
}

int main(int argc, char* argv[]) {
    try {
        std::filesystem::path file_A;
        std::filesystem::path file_B;
        std::filesystem::path output_file;
        int threshold = 4;

        for (const char* const* a = argv + 1; *a; ++a) {
            if (const char* str = argparse::parse_string_argument(a, "-o"sv))
                output_file = str;
            else if (argparse::parse_int_argument(threshold, a, "-t"sv));
            else {
                if (file_A.empty())
                    file_A = *a;
                else if (file_B.empty())
                    file_B = *a;
                else
                    throw argparse::usage_error("invalid argument");
            }
        }

        if (file_A.empty() || file_B.empty())
            throw argparse::usage_error("expected two image files to compare");

        std::cout << "comparing "sv << file_A << " to reference "sv << file_B << std::endl;

        auto A = PNG::loadImage2DR8G8B8A8(file_A);
        auto B = PNG::loadImage2DR8G8B8A8(file_B);

        if (width(A) != width(B) || height(A) != height(B))
            throw std::runtime_error("image dimensions must match!");


        auto diff = image2D<std::array<unsigned char, 3>>(width(A), height(A));

        std::transform(begin(A), end(A), begin(B), begin(diff), [](auto a, auto b) ->std::array<unsigned char, 3> {
            return {
                static_cast<unsigned char>(std::abs(channel<0>(a) - channel<0>(b))),
                static_cast<unsigned char>(std::abs(channel<1>(a) - channel<1>(b))),
                static_cast<unsigned char>(std::abs(channel<2>(a) - channel<2>(b)))
            };
        });

        auto diff_max = std::transform_reduce(begin(diff), end(diff), 0LL, [](long long a, long long b) { return std::max(a, b); }, [](const auto& color) {
                return std::max({ color[0], color[1], color[2] });
        });

        auto avg_rel_diff = std::transform_reduce(begin(A), end(A), begin(B), 0.0, [](auto a, auto b) { return a + b; }, [](auto a, auto b) {
            return (reldiff<0>(a, b) + reldiff<1>(a, b) + reldiff<2>(a, b)) / 3.0;
        }) / (width(A) * height(A));

        std::cout << "average relative difference: "sv << avg_rel_diff << std::endl;


        long long histogram[256] = {};

        for (const auto& c : diff) {
            ++histogram[c[0]];
            ++histogram[c[1]];
            ++histogram[c[2]];
        }

        std::cout << "histogram: 3x"sv << width(A) << 'x' << height(A) << " = "sv << 3 * width(A) * height(A) << " color values"sv << std::endl;

        for (int i = 0; i < std::size(histogram); ++i) {
            std::cout << std::setw(3) << i << ": "sv;

            constexpr int num_cols = 80;

            int n = histogram[i] * num_cols / (3 * width(A) * height(A));

            for (int j = 0; j < n; ++j)
                std::cout.put('#');

            for (int j = n; j < num_cols; ++j)
                std::cout.put(' ');

            std::cout << ' ' << histogram[i] << std::endl;
        }

        if (!output_file.empty()) {
            auto diff_color = image2D<uint32_t>(width(A), height(A));

            std::transform(begin(diff), end(diff), begin(diff_color), [=](const auto& color) -> std::uint_fast32_t {
                auto d = std::max({ color[0], color[1], color[2] });
                return d >= threshold ? (0xFF000000U | ((255 - d) << 8) | d) : 0xFFFFFFFFU;
            });

            PNG::saveImageR8G8B8A8(output_file, diff_color);
        }

        return histogram[std::min(std::max(threshold, 0), 255)];
    } catch (const argparse::usage_error& e) {
        std::cerr << "ERROR: " << e.what() << std::endl << print_usage;
        return -127;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "ERROR: unknown exception" << std::endl;
        return -128;
    }

    return EXIT_SUCCESS;
}
