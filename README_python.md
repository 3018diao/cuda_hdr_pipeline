# Python版HDR图像处理管道

这是一个用Python实现的HDR（高动态范围）图像处理管道，功能与C++/CUDA版本相同。

## 功能

- 加载HDR环境贴图（支持.hdr和.pfm格式）
- 应用色调映射（tone mapping）处理
- 支持曝光值调整
- 将结果保存为PNG格式
- 支持GPU加速处理（使用OpenCV的CUDA模块）

## 依赖项

使用前需要安装以下Python库：

```bash
pip install numpy opencv-python pillow
```

要使用GPU加速，需要安装带CUDA支持的OpenCV版本：

```bash
pip install opencv-python-headless-cuda
# 或者自行编译带CUDA支持的OpenCV
```

## 使用方法

```bash
python hdr_pipeline.py [选项] <输入文件>
```

### 命令行参数

- `<输入文件>`: 输入的HDR文件（.hdr或.pfm格式）
- `--exposure <值>`: 设置曝光值，默认为0.0
- `--brightpass <值>`: 设置亮度阈值，默认为0.9
- `--test-runs <N>`: 设置测试运行次数，默认为1
- `--cpu`: 强制使用CPU处理，即使GPU可用

### 示例

```bash
# 使用默认参数处理HDR文件（如果有GPU会自动使用）
python hdr_pipeline.py sample.hdr

# 设置曝光值为1.5
python hdr_pipeline.py sample.hdr --exposure 1.5

# 进行5次运行并计算平均处理时间
python hdr_pipeline.py sample.hdr --test-runs 5

# 强制使用CPU处理
python hdr_pipeline.py sample.hdr --cpu
```

## 输出

脚本将在当前目录下生成与输入文件同名但扩展名为.png的输出文件。同时在控制台输出处理时间统计信息。

## GPU加速说明

当检测到支持CUDA的GPU时，脚本会自动使用GPU进行处理以提高性能。但由于某些操作（如色调映射和伽马校正）在OpenCV CUDA中没有直接实现，仍需在CPU和GPU之间进行数据传输，所以加速效果可能不如纯CUDA实现明显。

如果遇到GPU处理错误，程序会自动回退到CPU处理。

## 与C++版本的区别

- 当GPU可用时，利用OpenCV的CUDA模块进行加速
- 仍有部分计算需要在CPU上进行
- 不需要CUDA编译器，仅需要预编译的OpenCV CUDA支持 