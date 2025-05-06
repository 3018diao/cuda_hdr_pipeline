#!/usr/bin/env python3
import argparse
import os
import sys
import time
import numpy as np
from PIL import Image
import cv2

class HDRPipeline:
    def __init__(self, width, height, force_cpu=False):
        self.frame_width = width
        self.frame_height = height
        
        # 检查GPU可用性
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        if cuda_devices == 0 or force_cpu:
            if force_cpu:
                print("已强制使用CPU模式")
            else:
                print("警告: 未找到支持CUDA的设备，将使用CPU处理")
            self.use_gpu = False
        else:
            print(f"找到 {cuda_devices} 个支持CUDA的设备")
            self.use_gpu = True
            # 打印设备信息
            cv2.cuda.printCudaDeviceInfo(0)
            
    def tone_mapping_cpu(self, v):
        """在CPU上实现色调映射函数"""
        numerator = v * (0.9036 * v + 0.018)
        denominator = v * (0.8748 * v + 0.354) + 0.14
        return numerator / denominator
    
    def srgb_gamma_cpu(self, u):
        """在CPU上实现sRGB伽马校正"""
        threshold = 0.0031308
        low = 12.92 * u
        high = 1.055 * np.power(u, 1.0/2.4) - 0.055
        mask = u > threshold
        return np.where(mask, high, low)
    
    def process_gpu(self, in_image, exposure, brightpass_threshold):
        """在GPU上处理HDR图像"""
        try:
            # 将图像上传到GPU
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(in_image.astype(np.float32))
            
            # 应用曝光值
            gpu_image_exposed = cv2.cuda.multiply(gpu_image, exposure)
            
            # 创建通道分离的GPU矩阵
            channels = cv2.cuda.split(gpu_image_exposed)
            result_channels = []
            
            # 对每个通道单独处理
            for i in range(3):
                # 下载到CPU进行色调映射（目前OpenCV没有直接的GPU色调映射实现）
                channel_cpu = channels[i].download()
                
                # 应用色调映射
                mapped = self.tone_mapping_cpu(channel_cpu)
                
                # 裁剪值到[0,1]
                mapped = np.clip(mapped, 0.0, 1.0)
                
                # 应用伽马校正
                gamma_corrected = self.srgb_gamma_cpu(mapped)
                
                # 转换为8位并上传回GPU
                channel_8bit = (gamma_corrected * 255.0).astype(np.uint8)
                gpu_channel = cv2.cuda_GpuMat()
                gpu_channel.upload(channel_8bit)
                result_channels.append(gpu_channel)
            
            # 合并通道
            result = cv2.cuda.merge(result_channels)
            
            # 将结果下载到CPU
            result_cpu = result.download()
            
            return result_cpu
            
        except cv2.error as e:
            print(f"GPU处理出错: {e}")
            print("回退到CPU处理...")
            return self.process_cpu(in_image, exposure, brightpass_threshold)
    
    def process_cpu(self, in_image, exposure, brightpass_threshold):
        """在CPU上处理HDR图像"""
        # 应用曝光值
        result = in_image.copy() * exposure
        
        # 应用色调映射
        result[:,:,0] = self.tone_mapping_cpu(result[:,:,0])
        result[:,:,1] = self.tone_mapping_cpu(result[:,:,1])
        result[:,:,2] = self.tone_mapping_cpu(result[:,:,2])
        
        # 裁剪值到 [0, 1] 范围
        result = np.clip(result, 0.0, 1.0)
        
        # 应用sRGB伽马校正
        result[:,:,0] = self.srgb_gamma_cpu(result[:,:,0])
        result[:,:,1] = self.srgb_gamma_cpu(result[:,:,1])
        result[:,:,2] = self.srgb_gamma_cpu(result[:,:,2])
        
        # 转换为8位整数
        result = (result * 255.0).astype(np.uint8)
        
        return result
    
    def process(self, in_image, exposure, brightpass_threshold):
        """根据GPU可用性选择处理方法"""
        if self.use_gpu:
            print("使用GPU处理...")
            return self.process_gpu(in_image, exposure, brightpass_threshold)
        else:
            print("使用CPU处理...")
            return self.process_cpu(in_image, exposure, brightpass_threshold)

def load_envmap(filename, flip=False):
    """加载HDR环境贴图，支持.hdr和.pfm格式"""
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == '.hdr':
        # 使用OpenCV加载HDR文件
        img = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif ext == '.pfm':
        # 简单的PFM加载实现
        with open(filename, 'rb') as f:
            header = f.readline().decode('utf-8').strip()
            if header != 'PF':
                raise ValueError(f"不支持的PFM格式: {header}")
            
            dims = f.readline().decode('utf-8').strip().split()
            width, height = int(dims[0]), int(dims[1])
            
            scale = float(f.readline().decode('utf-8').strip())
            endian = '<' if scale < 0 else '>'
            scale = abs(scale)
            
            data = np.fromfile(f, dtype=f'{endian}f')
            img = data.reshape((height, width, 3))
            
            if scale != 1.0:
                img *= scale
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    
    if flip:
        img = np.flip(img, axis=0)
    
    return img

def run(output_file, envmap_path, exposure_value, brightpass_threshold, test_runs, force_cpu=False):
    """运行HDR处理管道"""
    print(f"读取 {envmap_path}")
    
    exposure = 2.0 ** exposure_value
    envmap = load_envmap(envmap_path, False)
    
    image_height, image_width = envmap.shape[:2]
    print(f"图像尺寸: {image_width}x{image_height}")
    
    pipeline = HDRPipeline(image_width, image_height, force_cpu)
    
    pipeline_time = 0.0
    
    print(f"\n{test_runs} 次测试运行:")
    
    for i in range(test_runs):
        start_time = time.time()
        output = pipeline.process(envmap, exposure, brightpass_threshold)
        end_time = time.time()
        
        t = (end_time - start_time) * 1000  # 转换为毫秒
        print(f"t_{i+1}: {t:.2f} ms")
        
        pipeline_time += t
    
    print(f"平均时间: {pipeline_time / test_runs:.2f} ms")
    
    print(f"\n保存 {output_file}")
    Image.fromarray(output).save(output_file)

def main():
    parser = argparse.ArgumentParser(description='HDR图像处理管道')
    parser.add_argument('input_file', help='输入HDR文件(.hdr或.pfm格式)')
    parser.add_argument('--exposure', type=float, default=0.0, help='设置曝光值, 默认: 0.0')
    parser.add_argument('--brightpass', type=float, default=0.9, help='设置亮度阈值, 默认: 0.9')
    parser.add_argument('--test-runs', type=int, default=1, help='测试运行次数, 默认: 1')
    parser.add_argument('--cpu', action='store_true', help='强制使用CPU处理，即使GPU可用')
    
    args = parser.parse_args()
    
    try:
        if not os.path.exists(args.input_file):
            raise ValueError(f"输入文件不存在: {args.input_file}")
        
        ext = os.path.splitext(args.input_file)[1].lower()
        if ext not in ['.hdr', '.pfm']:
            raise ValueError(f"不支持的文件格式, 仅支持'.hdr'和'.pfm'文件")
        
        if args.test_runs < 0:
            raise ValueError("测试运行次数不能为负")
        
        output_file = os.path.splitext(os.path.basename(args.input_file))[0] + '.png'
        
        run(output_file, args.input_file, args.exposure, args.brightpass, args.test_runs, args.cpu)
        
    except Exception as e:
        print(f"错误: {str(e)}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 