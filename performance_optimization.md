# HDR管线光晕效果性能优化指南

## 🚀 性能瓶颈分析

### 当前实现的主要性能问题

1. **内存分配开销**
   - 每次处理都要分配临时内存 `d_composite`
   - GPU内存分配是昂贵的操作

2. **内存访问模式**
   - 滤波器操作中存在非连续内存访问
   - 缓存未充分利用

3. **线程配置**
   - 16×16的线程块可能不是最优选择
   - 占用率可能未达到最大

4. **数据布局**
   - RGBA交错存储可能导致内存访问效率低下

## ⚡ 优化方案

### 1. 内存管理优化

#### 1.1 预分配内存缓冲区
```cpp
// 在hdr_pipeline.h中添加
class HDRPipeline {
    // ... 现有成员
    CUDA::unique_ptr<float> d_composite;  // 预分配合成缓冲区
    
public:
    HDRPipeline(int width, int height);
};

// 在hdr_pipeline.cpp构造函数中
HDRPipeline::HDRPipeline(int width, int height)
    : frame_width(width),
      frame_height(height),
      d_input_image(CUDA::malloc<float>(width * height * 4)),
      d_output_image(CUDA::malloc_zeroed<uint32_t>(width * height)),
      d_bright_pass(CUDA::malloc<float>(width * height * 4)),
      d_blur_temp(CUDA::malloc<float>(width * height * 4)),
      d_blur_result(CUDA::malloc<float>(width * height * 4)),
      d_composite(CUDA::malloc<float>(width * height * 4)) {  // 预分配
}
```

#### 1.2 移除动态内存分配
```cpp
// 在hdr_pipeline.cu的tonemap函数中移除
// float* d_composite;
// throw_error(cudaMalloc(&d_composite, width * height * 4 * sizeof(float)));
// 替换为使用预分配的缓冲区
composite_bloom<<<grid, block>>>(composite, in, blur_result, width, height, adapted_exposure, brightpass_threshold);
// cudaFree(d_composite);  // 移除这行
```

### 2. 共享内存优化滤波器操作

#### 2.1 优化的水平模糊核函数
```cpp
__global__ void blur_horizontal_optimized(float* output, const float* input, int width, int height) {
    // 使用共享内存缓存输入数据
    __shared__ float shared_data[16 + 62][16 * 4];  // 16个线程 + 62个边界像素
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    if (y < height) {
        // 协作加载数据到共享内存
        int shared_x_start = blockIdx.x * blockDim.x - BLOOM_KERNEL_RADIUS;
        
        // 每个线程加载多个数据点
        for (int offset = 0; offset < (blockDim.x + 2 * BLOOM_KERNEL_RADIUS + blockDim.x - 1) / blockDim.x; offset++) {
            int load_x = shared_x_start + tx + offset * blockDim.x;
            int shared_idx = tx + offset * blockDim.x;
            
            if (shared_idx < 16 + 62 && load_x >= 0 && load_x < width) {
                int global_idx = (y * width + load_x) * 4;
                shared_data[shared_idx][ty * 4] = input[global_idx];
                shared_data[shared_idx][ty * 4 + 1] = input[global_idx + 1];
                shared_data[shared_idx][ty * 4 + 2] = input[global_idx + 2];
                shared_data[shared_idx][ty * 4 + 3] = input[global_idx + 3];
            } else if (shared_idx < 16 + 62) {
                // 边界填充0
                shared_data[shared_idx][ty * 4] = 0.0f;
                shared_data[shared_idx][ty * 4 + 1] = 0.0f;
                shared_data[shared_idx][ty * 4 + 2] = 0.0f;
                shared_data[shared_idx][ty * 4 + 3] = 0.0f;
            }
        }
        
        __syncthreads();
        
        if (x < width) {
            int out_idx = (y * width + x) * 4;
            
            for (int c = 0; c < 3; c++) {
                float sum = 0.0f;
                
                for (int i = 0; i < BLOOM_KERNEL_SIZE; i++) {
                    int shared_idx = tx + BLOOM_KERNEL_RADIUS + i - BLOOM_KERNEL_RADIUS;
                    sum += d_bloom_kernel[i] * shared_data[shared_idx][ty * 4 + c];
                }
                
                output[out_idx + c] = sum;
            }
            output[out_idx + 3] = shared_data[tx + BLOOM_KERNEL_RADIUS][ty * 4 + 3];
        }
    }
}
```

### 3. 线程配置优化

#### 3.1 调整线程块大小
```cpp
// 针对不同操作使用不同的线程块配置
void tonemap_optimized(uint32_t* out, const float* in, int width, int height, float exposure, float brightpass_threshold, 
            float* bright_pass, float* blur_temp, float* blur_result, float* composite) {
    
    // 对于计算密集型操作使用更大的线程块
    dim3 block_compute(32, 8);  // 256个线程，更好的占用率
    dim3 grid_compute((width + block_compute.x - 1) / block_compute.x, 
                     (height + block_compute.y - 1) / block_compute.y);
    
    // 对于内存密集型操作使用不同配置
    dim3 block_memory(16, 16);  // 保持原配置，适合内存操作
    dim3 grid_memory((width + block_memory.x - 1) / block_memory.x, 
                    (height + block_memory.y - 1) / block_memory.y);
    
    // 使用优化的配置
    extract_bright_pass<<<grid_compute, block_compute>>>(bright_pass, in, width, height, adapted_exposure, brightpass_threshold);
    blur_horizontal_optimized<<<grid_memory, block_memory>>>(blur_temp, bright_pass, width, height);
    blur_vertical_optimized<<<grid_memory, block_memory>>>(blur_result, blur_temp, width, height);
    composite_bloom<<<grid_compute, block_compute>>>(composite, in, blur_result, width, height, adapted_exposure, brightpass_threshold);
    tonemap_kernel<<<grid_compute, block_compute>>>(out, composite, width, height, 1.0f);
}
```

### 4. 数据布局优化

#### 4.1 分离的RGB通道存储
```cpp
// 考虑使用分离的通道存储而不是交错存储
class HDRPipelineOptimized {
    CUDA::unique_ptr<float> d_input_r, d_input_g, d_input_b;
    CUDA::unique_ptr<float> d_bright_r, d_bright_g, d_bright_b;
    // ... 其他分离通道
};

// 这样可以提高缓存利用率，减少内存带宽需求
```

### 5. 滤波器核优化

#### 5.2 预计算优化
```cpp
// 将常用的滤波器操作展开，减少循环开销
__global__ void blur_horizontal_unrolled(float* output, const float* input, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int out_idx = (y * width + x) * 4;
        
        // 展开关键的滤波器操作
        float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
        
        // 中心权重最大的几个点
        if (x >= 3 && x < width - 3) {
            int center_idx = (y * width + x) * 4;
            sum_r += d_bloom_kernel[31] * input[center_idx];
            sum_g += d_bloom_kernel[31] * input[center_idx + 1];
            sum_b += d_bloom_kernel[31] * input[center_idx + 2];
            
            // 对称权重
            for (int i = 1; i <= 3; i++) {
                int left_idx = (y * width + x - i) * 4;
                int right_idx = (y * width + x + i) * 4;
                float weight = d_bloom_kernel[31 - i];
                
                sum_r += weight * (input[left_idx] + input[right_idx]);
                sum_g += weight * (input[left_idx + 1] + input[right_idx + 1]);
                sum_b += weight * (input[left_idx + 2] + input[right_idx + 2]);
            }
        }
        
        // 完整循环处理边界情况
        if (x < 3 || x >= width - 3) {
            // 原有的完整循环逻辑
            for (int i = 0; i < BLOOM_KERNEL_SIZE; i++) {
                // ... 边界处理逻辑
            }
        }
        
        output[out_idx] = sum_r;
        output[out_idx + 1] = sum_g;
        output[out_idx + 2] = sum_b;
        output[out_idx + 3] = input[out_idx + 3];
    }
}
```

### 6. 流处理优化

#### 6.1 使用CUDA流并行处理
```cpp
void HDRPipeline::process_with_streams(cudaArray_t out, cudaArray_t in, float exposure, float brightpass_threshold) {
    // 创建多个CUDA流
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // 异步内存传输
    throw_error(cudaMemcpy2DFromArrayAsync(d_input_image.get(), frame_width * 16U, in, 0, 0, 
                                          frame_width * 16U, frame_height, cudaMemcpyDeviceToDevice, stream1));
    
    // 在不同流中并行执行
    tonemap_async(d_output_image.get(), d_input_image.get(), frame_width, frame_height, 
                 exposure, brightpass_threshold, d_bright_pass.get(), d_blur_temp.get(), 
                 d_blur_result.get(), d_composite.get(), stream1);
    
    // 异步内存传输输出
    throw_error(cudaMemcpy2DToArrayAsync(out, 0, 0, d_output_image.get(), frame_width * 4U, 
                                        frame_width * 4U, frame_height, cudaMemcpyDeviceToDevice, stream2));
    
    // 同步流
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}
```

## 📊 性能提升预期

| 优化项目 | 预期性能提升 | 实现难度 |
|---------|-------------|----------|
| 内存预分配 | 5-10% | 简单 |
| 共享内存优化 | 15-25% | 中等 |
| 线程配置调整 | 10-15% | 简单 |
| 数据布局优化 | 20-30% | 复杂 |
| 滤波器展开 | 8-12% | 中等 |
| 流处理 | 10-20% | 中等 |

**总体预期提升：50-80%的性能改进**

## 🔧 实施建议

### 阶段1：简单优化（立即可实施）
1. 内存预分配
2. 线程配置调整
3. 移除不必要的同步点

### 阶段2：中等优化
1. 共享内存优化
2. 滤波器核展开
3. 流处理引入

### 阶段3：高级优化
1. 数据布局重构
2. 自定义内存分配器
3. 多GPU支持

## 🎯 性能监控

```cpp
// 添加性能计时器
class PerformanceTimer {
    cudaEvent_t start, stop;
public:
    PerformanceTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    
    void startTimer() {
        cudaEventRecord(start);
    }
    
    float stopTimer() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// 在关键函数中使用
void tonemap_with_profiling(...) {
    PerformanceTimer timer;
    
    timer.startTimer();
    extract_bright_pass<<<grid, block>>>(...);
    printf("Bright pass: %.2f ms\n", timer.stopTimer());
    
    timer.startTimer();
    blur_horizontal<<<grid, block>>>(...);
    printf("Horizontal blur: %.2f ms\n", timer.stopTimer());
    
    // ... 其他操作的计时
}
```

这些优化可以根据具体需求和硬件配置逐步实施，预计能带来显著的性能提升！ 