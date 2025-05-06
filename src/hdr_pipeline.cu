#include <cstdint>

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
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    tonemap_kernel<<<grid, block>>>(out, in, width, height, exposure);
}
