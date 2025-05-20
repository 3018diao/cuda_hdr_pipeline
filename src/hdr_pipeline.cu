#include <cstdint>
#include "utils/cuda/error.h"


__device__ float tone_mapping(float v) {
    float numerator = v * (0.9036f * v + 0.018f);
    float denominator = v * (0.8748f * v + 0.354f) + 0.14f;
    return numerator / denominator;
}

__device__ float srgb_gamma(float u) {
    const float threshold = 0.0031308f;
    float low = 12.92f * u;
    float high = 1.055f * __powf(u, 1.0f/2.4f) - 0.055f;
    float mask = float(u > threshold);
    return mask * high + (1.0f - mask) * low;
}

// Helper device function for sum reduction within a warp using shuffle down
__device__ __inline__ float warpReduceSum(float val) {
    // __shfl_down_sync will transfer data from a higher lane to a lower lane
    // The mask 0xFFFFFFFF indicates all threads in the warp participate.
    // On each iteration, 'val' in the current thread is added with 'val' from 'threadIdx.x + offset'
    // This effectively sums up values in a tree-like fashion within the warp.
    // warpSize is a built-in variable (typically 32).
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    // The final sum for the warp resides in lane 0 (thread with tid % warpSize == 0)
    return val;
}

__global__ void  reduce_log_luminance(float* output, const float* input, int size) {
    // Shared memory to store partial sums from each warp.
    // We need one entry per warp in the block.
    // blockDim.x is 256, warpSize is 32. So, 256/32 = 8 entries.
    // The host code allocates blockDim.x * sizeof(float), which is more than enough.
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x; // Thread ID within the block (0-255)
    unsigned int bid = blockIdx.x;  // Block ID
    unsigned int bdim = blockDim.x; // Block dimension (should be 256 from host)
    unsigned int global_idx = bid * bdim + tid; // Global thread index

    // 1. Load initial value for this thread
    float my_val = 0.0f;
    if (global_idx < size) {
        my_val = input[global_idx];
    }

    // 2. Intra-warp reduction: Each warp reduces its values.
    // The sum for each warp will be in its lane 0.
    float warp_sum = warpReduceSum(my_val);

    // 3. Inter-warp reduction:
    // Lane 0 of each warp writes its partial sum to shared memory.
    unsigned int warp_id = tid / warpSize; // ID of the warp this thread belongs to
    unsigned int lane_id = tid % warpSize; // Lane ID within its warp (0-31)

    if (lane_id == 0) {
        sdata[warp_id] = warp_sum;
    }

    // Synchronize to ensure all warp sums are written to shared memory
    // before the first warp reads them.
    __syncthreads();

    // The first warp (warp_id == 0, i.e., threads 0 to warpSize-1)
    // reduces the partial sums from shared memory.
    // These sums are in sdata[0], sdata[1], ..., sdata[(bdim/warpSize) - 1].
    float block_total_sum = 0.0f;
    if (warp_id == 0) { // Only threads in the first warp participate
        // Check if the lane_id is less than the number of warps,
        // to avoid reading out of bounds from sdata if num_warps < warpSize.
        // For bdim=256, num_warps = 8. So lanes 0-7 of the first warp will load.
        if (lane_id < (bdim / warpSize)) {
            block_total_sum = sdata[lane_id];
        }
        // Else, block_total_sum remains 0.0f for other lanes in the first warp,
        // which is the correct identity for the sum reduction.

        // The first warp reduces these (up to) warpSize values.
        // The result will be in lane_id 0 of warp_id 0 (i.e., tid 0).
        block_total_sum = warpReduceSum(block_total_sum);
    }

    // Thread 0 of the block writes the final sum for this block to global memory.
    if (tid == 0) {
        output[bid] = block_total_sum;
    }
}

__global__ void compute_luminance(float* log_lum, const float* in ,int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 4;
        float r = in[idx];
        float g = in[idx + 1];
        float b = in[idx + 2];

        float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;

        lum = fmaxf(lum, 1e-6f);

        log_lum[y * width + x] = logf(lum);
    }
}

__global__ void tonemap_kernel(uint32_t* out, const float* in, int width, int height, float exposure) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int in_index = (y * width + x) * 4;
        
        
        float r = in[in_index];
        float g = in[in_index + 1];
        float b = in[in_index + 2];

        r *= exposure;
        g *= exposure;
        b *= exposure;

        r = tone_mapping(r);
        g = tone_mapping(g);
        b = tone_mapping(b);

        r = fminf(1.0f, fmaxf(0.0f, r));
        g = fminf(1.0f, fmaxf(0.0f, g));
        b = fminf(1.0f, fmaxf(0.0f, b));

        r = srgb_gamma(r);
        g = srgb_gamma(g);
        b = srgb_gamma(b);

        unsigned char r_byte = static_cast<unsigned char>(r * 255.0f + 0.5f);
        unsigned char g_byte = static_cast<unsigned char>(g * 255.0f + 0.5f);
        unsigned char b_byte = static_cast<unsigned char>(b * 255.0f + 0.5f);

        uint32_t pixel = r_byte | (g_byte << 8) | (b_byte << 16);
        out[y * width + x] = pixel;
    }
}

void tonemap(uint32_t* out, const float* in, int width, int height, float exposure, float brightpass_threshold) {
    int num_pixels = width * height;
    float* d_log_lum;

    throw_error(cudaMalloc(&d_log_lum, num_pixels * sizeof(float)));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    compute_luminance<<<grid, block>>>(d_log_lum, in, width, height);

    float* d_temp;
    int reduce_size = num_pixels;
    int block_size = 256;
    int num_blocks = (reduce_size + block_size - 1) / block_size;
    cudaMalloc(&d_temp, num_blocks * sizeof(float));

    reduce_log_luminance<<<num_blocks, block_size, block_size * sizeof(float)>>>(d_temp, d_log_lum, reduce_size);

    while (num_blocks > 1) {
        reduce_size = num_blocks;
        num_blocks = (reduce_size + block_size - 1) / block_size;
        float* d_temp2;
        cudaMalloc(&d_temp2, num_blocks * sizeof(float));
        reduce_log_luminance<<<num_blocks, block_size, block_size * sizeof(float)>>>(d_temp2, d_temp, reduce_size);
        cudaFree(d_temp);
        d_temp = d_temp2;
    }

    float log_sum;
    cudaMemcpy(&log_sum, d_temp, sizeof(float), cudaMemcpyDeviceToHost);

    float log_avg = log_sum / num_pixels;
    float avg_luminance = expf(log_avg);
    
    
    float adapted_exposure = exposure * (0.18f / avg_luminance);
    tonemap_kernel<<<grid, block>>>(out, in, width, height, adapted_exposure);

    cudaFree(d_log_lum);
    cudaFree(d_temp);

    throw_error(cudaDeviceSynchronize());
    throw_error(cudaPeekAtLastError());
}
