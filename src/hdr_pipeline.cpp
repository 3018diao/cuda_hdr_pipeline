#include "utils/cuda/error.h"
#include "utils/cuda/memory.h"

#include "hdr_pipeline.h"


HDRPipeline::HDRPipeline(int width, int height)
    : frame_width(width),
      frame_height(height),
      d_input_image(CUDA::malloc<float>(width * height * 4)),
      d_output_image(CUDA::malloc_zeroed<uint32_t>(width * height)) {
}


void tonemap(uint32_t* out, const float* in, int width, int height, float exposure, float brightpass_threshold);

void HDRPipeline::process(cudaArray_t out, cudaArray_t in, float exposure, float brightpass_threshold) {
    throw_error(cudaMemcpy2DFromArray(d_input_image.get(), frame_width * 16U, in, 0, 0, frame_width * 16U, frame_height, cudaMemcpyDeviceToDevice));

    tonemap(d_output_image.get(), d_input_image.get(), frame_width, frame_height, exposure, brightpass_threshold);

    throw_error(cudaMemcpy2DToArray(out, 0, 0, d_output_image.get(), frame_width * 4U, frame_width * 4U, frame_height, cudaMemcpyDeviceToDevice));
}
