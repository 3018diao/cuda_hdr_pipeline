#ifndef INCLUDED_HDR_PIPELINE
#define INCLUDED_HDR_PIPELINE

#include <cstdint>

#include <cuda_runtime_api.h>

#include "utils/cuda/memory.h"


class HDRPipeline {
    int frame_width;
    int frame_height;

    CUDA::unique_ptr<float> d_input_image;
    CUDA::unique_ptr<uint32_t> d_output_image;
    
    // 光晕效果所需的缓冲区
    CUDA::unique_ptr<float> d_bright_pass;      // 亮通道图像
    CUDA::unique_ptr<float> d_blur_temp;        // 临时模糊缓冲区
    CUDA::unique_ptr<float> d_blur_result;      // 最终模糊结果
    CUDA::unique_ptr<float> d_composite;        // 预分配的合成缓冲区

public:
    HDRPipeline(int width, int height);

    void process(cudaArray_t out, cudaArray_t in, float exposure, float brightpass_threshold);
};

#endif // INCLUDED_HDR_PIPELINE
