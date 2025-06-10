#include <cstdint>
#include "utils/cuda/error.h"
#include "bloom_kernel.h"
#include "performance_timer.h"

// Filter kernel in constant memory
__constant__ float d_bloom_kernel[BLOOM_KERNEL_SIZE];

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

// Saturation function: clamp value to [0,1] range
__device__ float saturate(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

// Compute bright channel contribution factor β(v)
__device__ float compute_bright_contribution(float v, float threshold) {
    float tone_mapped = tone_mapping(v);
    float factor = (tone_mapped - 0.8f * threshold) / (0.2f * threshold);
    float saturated = saturate(factor);
    return saturated * saturated; // Quadratic function
}

// Extract bright pass kernel function
__global__ void extract_bright_pass(float* bright_pass, const float* input, 
                                   int width, int height, float exposure, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 4;
        
        float r = input[idx] * exposure;
        float g = input[idx + 1] * exposure;
        float b = input[idx + 2] * exposure;
        
        // Compute contribution factor for each channel
        float beta_r = compute_bright_contribution(r, threshold);
        float beta_g = compute_bright_contribution(g, threshold);
        float beta_b = compute_bright_contribution(b, threshold);
        
        // Extract bright pass: B(x,y) = β(eI(x,y)) · I(x,y)
        bright_pass[idx] = beta_r * input[idx];
        bright_pass[idx + 1] = beta_g * input[idx + 1];
        bright_pass[idx + 2] = beta_b * input[idx + 2];
        bright_pass[idx + 3] = input[idx + 3]; // Alpha channel
    }
}

// Horizontal 1D Gaussian blur
__global__ void blur_horizontal(float* output, const float* input, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int out_idx = (y * width + x) * 4;
        
        for (int c = 0; c < 3; c++) { // Process RGB channels only
            float sum = 0.0f;
            
            for (int i = 0; i < BLOOM_KERNEL_SIZE; i++) {
                int offset = i - BLOOM_KERNEL_RADIUS;
                int sample_x = x + offset;
                
                // Boundary handling: pixels outside boundary are 0
                if (sample_x >= 0 && sample_x < width) {
                    int in_idx = (y * width + sample_x) * 4 + c;
                    sum += d_bloom_kernel[i] * input[in_idx];
                }
            }
            
            output[out_idx + c] = sum;
        }
        output[out_idx + 3] = input[out_idx + 3]; // Alpha channel
    }
}

// Vertical 1D Gaussian blur
__global__ void blur_vertical(float* output, const float* input, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int out_idx = (y * width + x) * 4;
        
        for (int c = 0; c < 3; c++) { // Process RGB channels only
            float sum = 0.0f;
            
            for (int i = 0; i < BLOOM_KERNEL_SIZE; i++) {
                int offset = i - BLOOM_KERNEL_RADIUS;
                int sample_y = y + offset;
                
                // Boundary handling: pixels outside boundary are 0
                if (sample_y >= 0 && sample_y < height) {
                    int in_idx = (sample_y * width + x) * 4 + c;
                    sum += d_bloom_kernel[i] * input[in_idx];
                }
            }
            
            output[out_idx + c] = sum;
        }
        output[out_idx + 3] = input[out_idx + 3]; // Alpha channel
    }
}

// Bloom effect composition kernel function
__global__ void composite_bloom(float* output, const float* input, const float* bloom,
                               int width, int height, float exposure, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 4;
        
        float r = input[idx] * exposure;
        float g = input[idx + 1] * exposure;
        float b = input[idx + 2] * exposure;
        
        // Compute contribution factors
        float beta_r = compute_bright_contribution(r, threshold);
        float beta_g = compute_bright_contribution(g, threshold);
        float beta_b = compute_bright_contribution(b, threshold);
        
        // Composition: O(x,y) = (1 - β(eI(x,y))) I(x,y) + B̄(x,y)
        output[idx] = (1.0f - beta_r) * input[idx] + bloom[idx];
        output[idx + 1] = (1.0f - beta_g) * input[idx + 1] + bloom[idx + 1];
        output[idx + 2] = (1.0f - beta_b) * input[idx + 2] + bloom[idx + 2];
        output[idx + 3] = input[idx + 3]; // Alpha channel
    }
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

void tonemap(uint32_t* out, const float* in, int width, int height, float exposure, float brightpass_threshold, 
            float* bright_pass, float* blur_temp, float* blur_result, float* composite) {
    
    // Copy filter kernel to constant memory
    static bool kernel_copied = false;
    if (!kernel_copied) {
        throw_error(cudaMemcpyToSymbol(d_bloom_kernel, bloom_kernel, sizeof(bloom_kernel)));
        kernel_copied = true;
    }
    
    int num_pixels = width * height;
    float* d_log_lum;

    throw_error(cudaMalloc(&d_log_lum, num_pixels * sizeof(float)));

    // Use optimized thread configurations
    dim3 block_compute(32, 8);  // 256 threads, better occupancy
    dim3 grid_compute((width + block_compute.x - 1) / block_compute.x, 
                     (height + block_compute.y - 1) / block_compute.y);
    
    dim3 block_memory(16, 16);  // Suitable for memory-intensive operations
    dim3 grid_memory((width + block_memory.x - 1) / block_memory.x, 
                    (height + block_memory.y - 1) / block_memory.y);

    compute_luminance<<<grid_memory, block_memory>>>(d_log_lum, in, width, height);

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
    
    // Bloom effect processing pipeline - using optimized thread configurations
    // 1. Extract bright pass
    extract_bright_pass<<<grid_compute, block_compute>>>(bright_pass, in, width, height, adapted_exposure, brightpass_threshold);
    
    // 2. Horizontal blur - memory-intensive operation
    blur_horizontal<<<grid_memory, block_memory>>>(blur_temp, bright_pass, width, height);
    
    // 3. Vertical blur - memory-intensive operation
    blur_vertical<<<grid_memory, block_memory>>>(blur_result, blur_temp, width, height);
    
    // 4. Composite bloom effect - using pre-allocated buffer
    composite_bloom<<<grid_compute, block_compute>>>(composite, in, blur_result, width, height, adapted_exposure, brightpass_threshold);
    
    // 5. Tone mapping
    tonemap_kernel<<<grid_compute, block_compute>>>(out, composite, width, height, 1.0f); // Exposure already applied in composition stage

    cudaFree(d_log_lum);
    cudaFree(d_temp);
    // No need to free d_composite since it's pre-allocated

    throw_error(cudaDeviceSynchronize());
    throw_error(cudaPeekAtLastError());
}
