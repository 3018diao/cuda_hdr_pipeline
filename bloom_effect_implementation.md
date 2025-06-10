# HDR管线光晕效果实现文档

## 📋 概述

本文档详细说明了在现有CUDA HDR管线基础上添加光晕效果（Bloom Effect）的实现。光晕效果模拟现实世界成像系统中非常亮的区域光线渗透到相邻较暗区域的现象，增强HDR图像的视觉真实感。

## 🎯 实现目标

根据作业要求，实现以下核心功能：
1. **亮通道提取**：识别并提取图像中的高亮区域
2. **可分离高斯模糊**：对亮通道进行高效的2D模糊处理
3. **图像合成**：将模糊的亮通道与原始图像合成
4. **能量守恒**：确保光晕效果不破坏图像的整体亮度平衡

## 🔬 物理和数学原理

### 1. 光晕效果的物理原理

#### 1.1 艾里斑现象
在现实世界中，任何成像系统都不能产生完美锐利的图像，这是由于：

- **衍射限制**：光线通过有限大小的光圈时会发生衍射
- **艾里斑（Airy Disk）**：点光源在成像平面上形成的衍射图案
- **点扩散函数（PSF）**：描述点光源如何在成像系统中扩散

```
实际成像 = 理想图像 ⊗ 点扩散函数
```

其中 ⊗ 表示卷积运算。

#### 1.2 光晕效应的产生机制

当图像中存在极亮区域时：
- **线性叠加**：每个亮点都会在其周围产生艾里斑图案
- **非线性响应**：人眼和相机传感器的非线性响应放大了这种效应
- **动态范围限制**：显示设备的动态范围限制使得这种效应更加明显

### 2. 数学模型推导

#### 2.1 色调映射函数的设计

色调映射函数用于将HDR值映射到可显示的LDR范围：

```
τ(v) = v·(0.9036·v + 0.018) / (v·(0.8748·v + 0.354) + 0.14)
```

这个函数具有以下特性：
- **S形曲线**：类似人眼的感知特性
- **单调递增**：保证亮度顺序不被破坏
- **渐近行为**：τ(v) → 1 当 v → ∞

**数学推导**：
设 f(v) = av²+bv, g(v) = cv²+dv+e，则：
```
τ(v) = f(v)/g(v) = (av²+bv)/(cv²+dv+e)
```

通过匹配期望的关键点和导数值，得到系数：
- a = 0.9036, b = 0.018
- c = 0.8748, d = 0.354, e = 0.14

#### 2.2 亮通道贡献因子的数学设计

贡献因子β(v)的设计考虑了以下因素：

**1. 阈值平滑过渡**
```
β(v) = saturate((τ(v) - 0.8·Ξ) / (0.2·Ξ))²
```

- **线性区间**：[0.8·Ξ, Ξ] 内线性增长
- **二次函数**：确保二阶连续性，避免视觉伪影
- **饱和函数**：限制在[0,1]范围内

**2. 数学性质分析**
- **连续性**：β(v) 在整个定义域内连续
- **平滑性**：β'(v) 连续，避免突变
- **单调性**：β(v) 在有效区间内单调递增

#### 2.3 可分离滤波器的数学基础

**高斯函数的可分离性**：
```
G(x,y) = G₁(x) · G₁(y)
```

其中 G₁(t) = (1/√(2πσ²)) · exp(-t²/(2σ²))

**卷积的可分离性**：
```
(I ⊗ G)(x,y) = (I ⊗ G₁ ⊗ G₁)(x,y) = ((I ⊗ G₁ˣ) ⊗ G₁ʸ)(x,y)
```

**复杂度分析**：
- 2D卷积：O(N² · M²) （N为图像尺寸，M为核尺寸）
- 分离卷积：O(N² · M) 
- 性能提升：M倍（对于63×63核，提升63倍）

### 3. 能量守恒原理

#### 3.1 物理意义
在真实的光学系统中，总光能量必须守恒：
```
∫∫ I_output(x,y) dxdy = ∫∫ I_input(x,y) dxdy
```

#### 3.2 数学实现
通过调整像素贡献，确保能量守恒：
```
O(x,y) = (1 - β(eI(x,y)))·I(x,y) + B̄(x,y)
```

**能量分析**：
- 原始贡献：(1 - β)·I
- 光晕贡献：B̄（来自其他像素的β·I）
- 总能量：保持不变

### 4. 数值稳定性考虑

#### 4.1 边界条件处理
```cpp
if (sample_x >= 0 && sample_x < width) {
    // 有效采样
} else {
    // 边界外视为0，避免数组越界
}
```

#### 4.2 浮点精度处理
- 使用 `fminf/fmaxf` 确保数值范围
- 避免除零：在对数亮度计算中添加小量 `1e-6f`
- 使用单精度浮点（float）平衡精度和性能

## 📁 修改的文件

### 1. 新增文件

#### `src/bloom_kernel.h`
- **功能**：定义光晕效果的高斯滤波器核系数
- **内容**：63个预定义的滤波器系数和相关常量

```cpp
#ifndef INCLUDED_BLOOM_KERNEL
#define INCLUDED_BLOOM_KERNEL

const float bloom_kernel[] = {
    0.000000046f, // [ 0]: -31
    0.000000108f, // [ 1]: -30
    // ... 63个系数
    0.000000046f, // [62]: 31
};

const int BLOOM_KERNEL_SIZE = 63;
const int BLOOM_KERNEL_RADIUS = 31;

#endif // INCLUDED_BLOOM_KERNEL
```

### 2. 修改的文件

#### `src/hdr_pipeline.h`
- **修改内容**：添加光晕效果所需的GPU内存缓冲区

```cpp
class HDRPipeline {
    // ... 原有成员
    
    // 光晕效果所需的缓冲区
    CUDA::unique_ptr<float> d_bright_pass;      // 亮通道图像
    CUDA::unique_ptr<float> d_blur_temp;        // 临时模糊缓冲区
    CUDA::unique_ptr<float> d_blur_result;      // 最终模糊结果
    
    // ...
};
```

#### `src/hdr_pipeline.cpp`
- **修改内容**：
  1. 在构造函数中初始化新的GPU内存缓冲区
  2. 更新`tonemap`函数调用，传递光晕效果缓冲区

```cpp
HDRPipeline::HDRPipeline(int width, int height)
    : frame_width(width),
      frame_height(height),
      d_input_image(CUDA::malloc<float>(width * height * 4)),
      d_output_image(CUDA::malloc_zeroed<uint32_t>(width * height)),
      d_bright_pass(CUDA::malloc<float>(width * height * 4)),
      d_blur_temp(CUDA::malloc<float>(width * height * 4)),
      d_blur_result(CUDA::malloc<float>(width * height * 4)) {
}
```

#### `src/hdr_pipeline.cu`
- **修改内容**：这是最重要的修改文件，包含所有光晕效果的CUDA核函数

## 🧮 算法实现

### 1. 数学公式

#### 色调映射函数
```
τ(v) = v·(0.9036·v + 0.018) / (v·(0.8748·v + 0.354) + 0.14)
```

#### 亮通道贡献因子
```
β(v) = saturate((τ(v) - 0.8·Ξ) / (0.2·Ξ))²
```

#### 亮通道提取
```
B(x,y) = β(eI(x,y)) · I(x,y)
```

#### 图像合成
```
O(x,y) = (1 - β(eI(x,y)))I(x,y) + B̄(x,y)
```

### 2. 核心CUDA核函数

#### 2.1 饱和函数和贡献因子计算

```cpp
// 饱和函数：将值限制在[0,1]范围内
__device__ float saturate(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

// 计算亮通道贡献因子β(v)
__device__ float compute_bright_contribution(float v, float threshold) {
    float tone_mapped = tone_mapping(v);
    float factor = (tone_mapped - 0.8f * threshold) / (0.2f * threshold);
    float saturated = saturate(factor);
    return saturated * saturated; // 二次函数
}
```

#### 2.2 亮通道提取核函数

```cpp
__global__ void extract_bright_pass(float* bright_pass, const float* input, 
                                   int width, int height, float exposure, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 4;
        
        float r = input[idx] * exposure;
        float g = input[idx + 1] * exposure;
        float b = input[idx + 2] * exposure;
        
        // 计算每个通道的贡献因子
        float beta_r = compute_bright_contribution(r, threshold);
        float beta_g = compute_bright_contribution(g, threshold);
        float beta_b = compute_bright_contribution(b, threshold);
        
        // 提取亮通道：B(x,y) = β(eI(x,y)) · I(x,y)
        bright_pass[idx] = beta_r * input[idx];
        bright_pass[idx + 1] = beta_g * input[idx + 1];
        bright_pass[idx + 2] = beta_b * input[idx + 2];
        bright_pass[idx + 3] = input[idx + 3]; // Alpha通道
    }
}
```

#### 2.3 可分离高斯模糊核函数

**水平方向模糊：**
```cpp
__global__ void blur_horizontal(float* output, const float* input, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int out_idx = (y * width + x) * 4;
        
        for (int c = 0; c < 3; c++) { // 只处理RGB通道
            float sum = 0.0f;
            
            for (int i = 0; i < BLOOM_KERNEL_SIZE; i++) {
                int offset = i - BLOOM_KERNEL_RADIUS;
                int sample_x = x + offset;
                
                // 边界处理：超出边界的像素值为0
                if (sample_x >= 0 && sample_x < width) {
                    int in_idx = (y * width + sample_x) * 4 + c;
                    sum += d_bloom_kernel[i] * input[in_idx];
                }
            }
            
            output[out_idx + c] = sum;
        }
        output[out_idx + 3] = input[out_idx + 3]; // Alpha通道
    }
}
```

**垂直方向模糊：**
```cpp
__global__ void blur_vertical(float* output, const float* input, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int out_idx = (y * width + x) * 4;
        
        for (int c = 0; c < 3; c++) { // 只处理RGB通道
            float sum = 0.0f;
            
            for (int i = 0; i < BLOOM_KERNEL_SIZE; i++) {
                int offset = i - BLOOM_KERNEL_RADIUS;
                int sample_y = y + offset;
                
                // 边界处理：超出边界的像素值为0
                if (sample_y >= 0 && sample_y < height) {
                    int in_idx = (sample_y * width + x) * 4 + c;
                    sum += d_bloom_kernel[i] * input[in_idx];
                }
            }
            
            output[out_idx + c] = sum;
        }
        output[out_idx + 3] = input[out_idx + 3]; // Alpha通道
    }
}
```

#### 2.4 光晕合成核函数

```cpp
__global__ void composite_bloom(float* output, const float* input, const float* bloom,
                               int width, int height, float exposure, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 4;
        
        float r = input[idx] * exposure;
        float g = input[idx + 1] * exposure;
        float b = input[idx + 2] * exposure;
        
        // 计算贡献因子
        float beta_r = compute_bright_contribution(r, threshold);
        float beta_g = compute_bright_contribution(g, threshold);
        float beta_b = compute_bright_contribution(b, threshold);
        
        // 合成：O(x,y) = (1 - β(eI(x,y))) I(x,y) + B̄(x,y)
        output[idx] = (1.0f - beta_r) * input[idx] + bloom[idx];
        output[idx + 1] = (1.0f - beta_g) * input[idx + 1] + bloom[idx + 1];
        output[idx + 2] = (1.0f - beta_b) * input[idx + 2] + bloom[idx + 2];
        output[idx + 3] = input[idx + 3]; // Alpha通道
    }
}
```

### 3. 主处理流程

修改后的`tonemap`函数实现了完整的光晕效果处理流程：

```cpp
void tonemap(uint32_t* out, const float* in, int width, int height, float exposure, float brightpass_threshold, 
            float* bright_pass, float* blur_temp, float* blur_result) {
    
    // 1. 初始化滤波器核到常量内存
    static bool kernel_copied = false;
    if (!kernel_copied) {
        throw_error(cudaMemcpyToSymbol(d_bloom_kernel, bloom_kernel, sizeof(bloom_kernel)));
        kernel_copied = true;
    }
    
    // 2. 计算自适应曝光（原有逻辑）
    // ... 对数亮度计算和归约逻辑 ...
    
    float adapted_exposure = exposure * (0.18f / avg_luminance);
    
    // 3. 光晕效果处理流程
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    // 3.1 提取亮通道
    extract_bright_pass<<<grid, block>>>(bright_pass, in, width, height, adapted_exposure, brightpass_threshold);
    
    // 3.2 水平方向模糊
    blur_horizontal<<<grid, block>>>(blur_temp, bright_pass, width, height);
    
    // 3.3 垂直方向模糊
    blur_vertical<<<grid, block>>>(blur_result, blur_temp, width, height);
    
    // 3.4 合成光晕效果
    float* d_composite;
    throw_error(cudaMalloc(&d_composite, width * height * 4 * sizeof(float)));
    composite_bloom<<<grid, block>>>(d_composite, in, blur_result, width, height, adapted_exposure, brightpass_threshold);
    
    // 3.5 色调映射和输出
    tonemap_kernel<<<grid, block>>>(out, d_composite, width, height, 1.0f);
    
    // 清理临时内存
    cudaFree(d_composite);
}
```

## 🔧 技术特点

### 1. 可分离滤波器优化
- **原理**：利用高斯核的可分离性，将2D卷积分解为两个1D卷积
- **复杂度**：从O(N²)优化到O(N)，显著提升性能
- **实现**：先进行水平方向模糊，再进行垂直方向模糊

### 2. 常量内存使用
- **优势**：将63个滤波器系数存储在常量内存中，提供缓存优化的访问
- **实现**：使用`__constant__`修饰符和`cudaMemcpyToSymbol`

### 3. 边界处理
- **策略**：图像边界外的像素值假设为0
- **实现**：在核函数中进行边界检查，超出范围的像素不参与计算

### 4. 能量守恒
- **原理**：通过`(1 - β)`因子确保原始像素的贡献相应减少
- **效果**：避免光晕效果导致图像整体过亮

## 📊 算法流程图

```
                    ┌─────────────────────────────────┐
                    │         HDR输入图像             │
                    │    (RGBA float32 格式)          │
                    └─────────────┬───────────────────┘
                                  │
                    ┌─────────────▼───────────────────┐
                    │      计算对数亮度分布           │
                    │   compute_luminance()           │
                    │  ┌─────────────────────────────┐ │
                    │  │ L = 0.2126R + 0.7152G +    │ │
                    │  │     0.0722B                 │ │
                    │  │ log_lum = log(max(L,1e-6))  │ │
                    │  └─────────────────────────────┘ │
                    └─────────────┬───────────────────┘
                                  │
                    ┌─────────────▼───────────────────┐
                    │      并行归约求和               │
                    │   reduce_log_luminance()        │
                    │  ┌─────────────────────────────┐ │
                    │  │ Warp内归约 → Warp间归约     │ │
                    │  │ 多轮迭代直到单一结果        │ │
                    │  └─────────────────────────────┘ │
                    └─────────────┬───────────────────┘
                                  │
                    ┌─────────────▼───────────────────┐
                    │      计算自适应曝光             │
                    │  ┌─────────────────────────────┐ │
                    │  │ avg_lum = exp(log_sum/N)    │ │
                    │  │ adapted_exp = exp*(0.18/avg)│ │
                    │  └─────────────────────────────┘ │
                    └─────────────┬───────────────────┘
                                  │
        ┌─────────────────────────▼─────────────────────────┐
        │                  光晕效果处理流程                  │
        └─────────────────────────┬─────────────────────────┘
                                  │
                    ┌─────────────▼───────────────────┐
                    │        提取亮通道               │
                    │   extract_bright_pass()         │
                    │  ┌─────────────────────────────┐ │
                    │  │ β(v) = saturate((τ(v)-      │ │
                    │  │        0.8*Ξ)/(0.2*Ξ))²     │ │
                    │  │ B(x,y) = β(eI(x,y)) * I(x,y)│ │
                    │  └─────────────────────────────┘ │
                    └─────────────┬───────────────────┘
                                  │
                    ┌─────────────▼───────────────────┐
                    │      水平方向模糊               │
                    │     blur_horizontal()           │
                    │  ┌─────────────────────────────┐ │
                    │  │  使用63点高斯核             │ │
                    │  │  Bₓ(x,y) = Σ wᵢ*B(x+i,y)   │ │
                    │  │  边界处理：零填充           │ │
                    │  └─────────────────────────────┘ │
                    └─────────────┬───────────────────┘
                                  │
                    ┌─────────────▼───────────────────┐
                    │      垂直方向模糊               │
                    │      blur_vertical()            │
                    │  ┌─────────────────────────────┐ │
                    │  │  使用63点高斯核             │ │
                    │  │  B̄(x,y) = Σ wⱼ*Bₓ(x,y+j)   │ │
                    │  │  边界处理：零填充           │ │
                    │  └─────────────────────────────┘ │
                    └─────────────┬───────────────────┘
                                  │
                    ┌─────────────▼───────────────────┐
                    │       合成光晕效果              │
                    │    composite_bloom()            │
                    │  ┌─────────────────────────────┐ │
                    │  │ O(x,y) = (1-β(eI(x,y)))*    │ │
                    │  │          I(x,y) + B̄(x,y)     │ │
                    │  │ 能量守恒的图像合成          │ │
                    │  └─────────────────────────────┘ │
                    └─────────────┬───────────────────┘
                                  │
                    ┌─────────────▼───────────────────┐
                    │        色调映射                 │
                    │     tonemap_kernel()            │
                    │  ┌─────────────────────────────┐ │
                    │  │ τ(v) = v*(0.9036*v+0.018)/ │ │
                    │  │        (v*(0.8748*v+0.354)+ │ │
                    │  │         0.14)               │ │
                    │  └─────────────────────────────┘ │
                    └─────────────┬───────────────────┘
                                  │
                    ┌─────────────▼───────────────────┐
                    │        伽马校正                 │
                    │       srgb_gamma()              │
                    │  ┌─────────────────────────────┐ │
                    │  │ u ≤ 0.0031308:              │ │
                    │  │   out = 12.92 * u           │ │
                    │  │ u > 0.0031308:              │ │
                    │  │   out = 1.055*u^(1/2.4)-0.055│ │
                    │  └─────────────────────────────┘ │
                    └─────────────┬───────────────────┘
                                  │
                    ┌─────────────▼───────────────────┐
                    │      转换为8位整数              │
                    │  ┌─────────────────────────────┐ │
                    │  │ RGB_byte = (RGB_float *     │ │
                    │  │            255.0 + 0.5)    │ │
                    │  │ 打包为32位RGBA像素          │ │
                    │  └─────────────────────────────┘ │
                    └─────────────┬───────────────────┘
                                  │
                    ┌─────────────▼───────────────────┐
                    │         LDR输出图像             │
                    │     (8位RGBA PNG格式)           │
                    └─────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────┐
    │                    并行处理说明                         │
    ├─────────────────────────────────────────────────────────┤
    │  • GPU线程组织：16×16 thread blocks                    │
    │  • 每个线程处理一个像素                                │
    │  • 常量内存存储63个滤波器系数                          │
    │  • 共享内存用于Warp间归约                              │
    │  • 可分离滤波器：O(N²M) → O(N²M)，性能提升M倍         │
    └─────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────┐
    │                    内存布局说明                         │
    ├─────────────────────────────────────────────────────────┤
    │  d_input_image    │ 输入HDR图像 (W×H×4 float)          │
    │  d_bright_pass    │ 亮通道图像   (W×H×4 float)          │
    │  d_blur_temp      │ 水平模糊结果 (W×H×4 float)          │
    │  d_blur_result    │ 完整模糊结果 (W×H×4 float)          │
    │  d_composite      │ 合成图像     (W×H×4 float)          │
    │  d_output_image   │ 输出LDR图像  (W×H uint32)           │
    │  d_bloom_kernel   │ 滤波器核     (63个常量 float)       │
    └─────────────────────────────────────────────────────────┘
```

## 🎮 使用方法

编译并运行程序：
```bash
cd build
cmake ..
make
./bin/hdr_pipeline --exposure 1.0 --brightpass 0.8 input.hdr
```

参数说明：
- `--exposure`：曝光值，控制整体图像亮度
- `--brightpass`：光晕阈值，控制哪些区域产生光晕效果

## ✨ 实现效果

光晕效果的加入使得：
1. **明亮区域**（如天空、光源）周围产生自然的光晕
2. **视觉真实感**增强，模拟真实相机的光学特性
3. **细节保持**，暗部细节不会因光晕而丢失
4. **性能优化**，通过可分离滤波器和GPU并行计算实现高效处理

## 🔍 总结

本实现严格按照作业要求，使用CUDA并行计算技术实现了高效的光晕效果算法。通过数学公式精确实现亮通道提取、可分离高斯模糊和能量守恒的图像合成，为HDR图像处理管线添加了重要的视觉增强功能。 