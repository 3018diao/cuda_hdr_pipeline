# HDR 图像处理流水线

这是一个基于 CUDA 的高动态范围 (HDR) 图像处理流水线，用于将 HDR 图像转换为适合显示的低动态范围 (LDR) 图像。

## 代码架构

项目采用模块化设计，主要组件如下图所示：

```
+------------------------+
|       main.cpp         |  <-- 程序入口点
+------------+-----------+
             |
             v
+------------+-----------+    +------------------+
|    HDRPipeline 类      | <--+  envmap 模块     |
| (hdr_pipeline.h/cpp)   |    | (envmap.h/cpp)   |
+------------+-----------+    +------------------+
             |
             v
+------------+-----------+
|  CUDA 核心实现         |
| (hdr_pipeline.cu)      |
+------------------------+
             ^
             |
+------------+-------------------------------+
|                  工具库                    |
+---------+------------+-----------+---------+
| 参数解析 |  图像处理  |  I/O 操作  | CUDA工具 |
|(argparse)| (image.h) | (io.h/png)|  (cuda/) |
+---------+------------+-----------+---------+
```

### 核心组件描述

1. **主入口 (main.cpp)**
   - 处理命令行参数
   - 控制整体执行流程
   - 协调 HDR 流水线处理

2. **HDR 流水线 (hdr_pipeline.h/cpp)**
   - 定义 `HDRPipeline` 类及其接口
   - 实现 HDR 到 LDR 的转换流程
   - 管理 GPU 内存和数据传输

3. **CUDA 实现 (hdr_pipeline.cu)**
   - 包含 CUDA 核函数
   - 实现并行化的图像处理算法
   - 执行色调映射和亮度通道处理

4. **环境贴图处理 (envmap.h/cpp)**
   - 负责加载和处理 HDR 环境贴图
   - 支持 .hdr 和 .pfm 文件格式

5. **工具库**
   - **参数解析 (argparse.h/cpp)**: 处理命令行参数
   - **图像处理 (image.h)**: 定义图像数据结构和基本操作
   - **I/O 操作 (io.h, png.h)**: 处理文件输入输出
   - **CUDA 工具 (cuda/)**:
     - 内存管理
     - 错误处理
     - 设备信息
     - 事件处理
     - 数组操作

### 数据流

```
+-------------+     +----------------+     +----------------+
| HDR 输入文件 | --> | CPU 内存(图像) | --> | GPU 内存(图像) |
+-------------+     +----------------+     +-------+--------+
                                                  |
                                                  v
                                          +----------------+
                                          | CUDA 处理      |
                                          | - 色调映射     |
                                          | - 曝光调整     |
                                          | - 亮度处理     |
                                          +-------+--------+
                                                  |
                                                  v
+-------------+     +----------------+     +----------------+
| LDR 输出文件 | <-- | CPU 内存(图像) | <-- | GPU 内存(图像) |
+-------------+     +----------------+     +----------------+
```

## 构建指南

### 前提条件

- CUDA 工具包
- CMake (3.0+)
- C++ 编译器 (支持 C++11 或更高)

### 构建步骤

1. 创建构建目录：
   ```bash
   mkdir build && cd build
   ```

2. 配置项目：
   ```bash
   cmake ..
   ```

3. 编译：
   ```bash
   cmake --build .
   ```

## 使用方法

程序使用方法:
```
hdr_pipeline [{options}] {<input-file>}
options:
  --device <i>          使用CUDA设备<i>，默认值: 0
  --exposure <v>        设置曝光值为<v>，默认值: 0.0
  --brightpass <v>      设置亮度通道阈值为<v>，默认值: 0.9
  --test-runs <N>       平均计时使用<N>次测试运行，默认值: 1
```

### 示例

```bash
./hdr_pipeline --device 0 --exposure 1.5 --brightpass 0.85 input.hdr
```

这将处理 `input.hdr` 文件，使用 CUDA 设备 0，曝光值为 1.5，亮度通道阈值为 0.85，输出为 `input.png`。

## 处理流程

1. 加载 HDR 环境贴图
2. 将图像数据从 CPU 传输到 GPU
3. 在 GPU 上执行 HDR 到 LDR 的转换:
   - 应用曝光调整
   - 执行色调映射
   - 处理亮度通道
4. 将处理后的图像从 GPU 传回 CPU
5. 保存为 PNG 格式

## 项目结构

```
.
├── src/                  # 源代码目录
│   ├── main.cpp          # 主入口文件
│   ├── hdr_pipeline.h    # HDR 流水线类定义
│   ├── hdr_pipeline.cpp  # HDR 流水线 CPU 实现
│   ├── hdr_pipeline.cu   # HDR 流水线 CUDA 实现
│   ├── envmap.h          # 环境贴图处理定义
│   ├── envmap.cpp        # 环境贴图处理实现
│   ├── utils/            # 工具库
│   │   ├── argparse.h    # 参数解析
│   │   ├── argparse.cpp
│   │   ├── image.h       # 图像处理
│   │   ├── io.h          # I/O 操作
│   │   ├── io/           # I/O 实现
│   │   │   └── png.h     # PNG 文件支持
│   │   └── cuda/         # CUDA 相关工具
│   │       ├── array.h   # 数组操作
│   │       ├── error.h   # 错误处理
│   │       ├── event.h   # 事件处理
│   │       ├── device.h  # 设备信息
│   │       └── memory.h  # 内存管理
│   └── tools/            # 额外工具
│       └── imgdiff/      # 图像比较工具
├── CMakeLists.txt        # 根目录构建配置
└── README.md             # 本文件
``` 