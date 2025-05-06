#include <cstdint>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <filesystem>

#include <cuda_runtime_api.h>

#include "utils/argparse.h"
#include "utils/image.h"
#include "utils/io/png.h"
#include "utils/cuda/array.h"
#include "utils/cuda/error.h"
#include "utils/cuda/event.h"
#include "utils/cuda/device.h"
#include "utils/cuda/memory.h"

#include "envmap.h"
#include "hdr_pipeline.h"

using namespace std::literals;

namespace {
    std::ostream& print_usage(std::ostream& out) {
        return out << R"""(usage: hdr_pipeline [{options}] {<input-file>}
    options:
      --device <i>          use CUDA device <i>, default: 0
      --exposure <v>        set exposure value to <v>, default: 0.0
      --brightpass <v>      set brightpass threshold to <v>, default: 0.9
      --test-runs <N>       average timings over <N> test runs, default: 1
)""";
    }
}

namespace {
    std::ostream& pad(std::ostream& out, int n) {
        for (int i = n; i > 0; --i)
            out.put(' ');
        return out;
    }
}

void run(const std::filesystem::path& output_file, const std::filesystem::path& envmap_path, int cuda_device, float exposure_value, float brightpass_threshold, int test_runs) {
    CUDA::print_device_properties(std::cout, cuda_device) << std::endl << std::flush;
    throw_error(cudaSetDevice(cuda_device));

    std::cout << std::endl << "reading " << envmap_path << std::endl << std::flush;

    float exposure = std::exp2(exposure_value);
    auto envmap = load_envmap(envmap_path, false);

    int image_width = static_cast<int>(width(envmap));
    int image_height = static_cast<int>(height(envmap));

    auto hdr_frame = CUDA::create_array(width(envmap), height(envmap), { 32, 32, 32, 32, cudaChannelFormatKindFloat });
    auto ldr_frame = CUDA::create_array(width(envmap), height(envmap), { 8, 8, 8, 8, cudaChannelFormatKindUnsigned });

    throw_error(cudaMemcpy2DToArray(hdr_frame.get(), 0, 0, data(envmap), image_width * 16U, image_width * 16U, image_height, cudaMemcpyHostToDevice));

    HDRPipeline pipeline(image_width, image_height);


    auto pipeline_begin = CUDA::create_event();
    auto pipeline_end = CUDA::create_event();

    float pipeline_time = 0.0f;

    std::cout << std::endl << test_runs << " test run(s):" << std::endl;

    int padding = static_cast<int>(std::log10(test_runs));
    int next_padding_shift = 10;

    for (int i = 0; i < test_runs; ++i) {
        throw_error(cudaEventRecord(pipeline_begin.get()));
        pipeline.process(ldr_frame.get(), hdr_frame.get(), exposure, brightpass_threshold);
        throw_error(cudaEventRecord(pipeline_end.get()));

        throw_error(cudaEventSynchronize(pipeline_end.get()));

        auto t = CUDA::elapsed_time(pipeline_begin.get(), pipeline_end.get());


        if ((i + 1) >= next_padding_shift) {
            --padding;
            next_padding_shift *= 10;
        }

        pad(std::cout, padding) << "t_" << (i + 1) << ": " << std::setprecision(2) << std::fixed << t << " ms" << std::endl << std::flush;

        pipeline_time += t;
    }

    std::cout << "avg time: " << std::setprecision(2) << std::fixed << pipeline_time / test_runs << " ms" << std::endl << std::flush;

    image2D<uint32_t> output(image_width, image_height);
    throw_error(cudaMemcpy2DFromArray(data(output), width(output) * 4U, ldr_frame.get(), 0, 0, image_width * 4U, image_height, cudaMemcpyDeviceToHost));

    std::cout << std::endl << "saving " << output_file << std::endl << std::flush;
    PNG::saveImageR8G8B8(output_file.string().c_str(), output);
}


int main(int argc, char* argv[]) {
    try {
        std::filesystem::path envmap;
        int cuda_device = 0;
        float exposure_value = 0.0f;
        float brightpass_threshold = 0.9f;
        int test_runs = 1;

        for (const char* const* a = argv + 1; *a; ++a) {
            if (!argparse::parse_int_argument(cuda_device, a, "--device"sv))
            if (!argparse::parse_float_argument(exposure_value, a, "--exposure"sv))
            if (!argparse::parse_float_argument(brightpass_threshold, a, "--brightpass"sv))
            if (!argparse::parse_int_argument(test_runs, a, "--test-runs")) {
                std::filesystem::path input_file = *a;

                auto ext = input_file.extension();

                if (ext == ".hdr"sv || ext == ".pfm"sv)
                    envmap = std::move(input_file);
                else
                    throw argparse::usage_error("unsupported file format, only '.hdr' and '.pfm' files supported");
            }
        }

        if (envmap.empty())
            throw argparse::usage_error("expected input file");

        if (test_runs < 0)
            throw argparse::usage_error("number of test runs cannot be negative");

        run(envmap.filename().replace_extension(".png"), envmap, cuda_device, exposure_value, brightpass_threshold, test_runs);
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
