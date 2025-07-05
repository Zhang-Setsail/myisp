#!/usr/bin/env python3
"""
分析内存使用情况的脚本
比较文件大小和内存大小的差异
"""

import os
import numpy as np
import torch
import imageio
import sys
from load_data import extract_bayer_channels

def analyze_single_image(image_path):
    """分析单张图像的内存使用情况"""
    print(f"\n分析图像: {image_path}")
    print("=" * 60)
    
    # 1. 文件大小
    if os.path.exists(image_path):
        file_size = os.path.getsize(image_path)
        print(f"文件大小: {file_size / 1024:.2f} KB ({file_size} bytes)")
    else:
        print("文件不存在!")
        return
    
    # 2. 加载原始图像数据
    try:
        raw_image = imageio.imread(image_path)
        print(f"原始图像形状: {raw_image.shape}")
        print(f"原始图像数据类型: {raw_image.dtype}")
        
        # 原始图像在内存中的大小
        raw_memory_size = raw_image.nbytes
        print(f"原始图像内存大小: {raw_memory_size / 1024:.2f} KB ({raw_memory_size} bytes)")
        
        # 3. 处理后的图像数据
        if len(raw_image.shape) == 2:  # 如果是RAW图像
            processed_image = extract_bayer_channels(raw_image)
            print(f"处理后形状: {processed_image.shape}")
            print(f"处理后数据类型: {processed_image.dtype}")
            
            processed_memory_size = processed_image.nbytes
            print(f"处理后内存大小: {processed_memory_size / 1024:.2f} KB ({processed_memory_size} bytes)")
            
            # 4. 转换为PyTorch张量
            tensor_image = torch.from_numpy(processed_image.transpose((2, 0, 1)))
            tensor_memory_size = tensor_image.nelement() * tensor_image.element_size()
            print(f"PyTorch张量内存大小: {tensor_memory_size / 1024:.2f} KB ({tensor_memory_size} bytes)")
            
        else:  # 如果是RGB图像
            # 转换为float32并归一化
            processed_image = np.float32(raw_image) / 255.0
            print(f"处理后形状: {processed_image.shape}")
            print(f"处理后数据类型: {processed_image.dtype}")
            
            processed_memory_size = processed_image.nbytes
            print(f"处理后内存大小: {processed_memory_size / 1024:.2f} KB ({processed_memory_size} bytes)")
            
            # 转换为PyTorch张量
            tensor_image = torch.from_numpy(processed_image.transpose((2, 0, 1)))
            tensor_memory_size = tensor_image.nelement() * tensor_image.element_size()
            print(f"PyTorch张量内存大小: {tensor_memory_size / 1024:.2f} KB ({tensor_memory_size} bytes)")
        
        # 5. 计算压缩比和内存膨胀比
        compression_ratio = file_size / raw_memory_size
        memory_expansion_ratio = tensor_memory_size / file_size
        
        print(f"\n压缩比 (文件/原始内存): {compression_ratio:.2f}x")
        print(f"内存膨胀比 (张量内存/文件): {memory_expansion_ratio:.2f}x")
        
        return {
            'file_size': file_size,
            'raw_memory_size': raw_memory_size,
            'processed_memory_size': processed_memory_size,
            'tensor_memory_size': tensor_memory_size,
            'compression_ratio': compression_ratio,
            'memory_expansion_ratio': memory_expansion_ratio
        }
        
    except Exception as e:
        print(f"处理图像时出错: {e}")
        return None


def analyze_dataset_memory():
    """分析整个数据集的内存使用情况"""
    print("\n" + "=" * 60)
    print("数据集内存使用分析")
    print("=" * 60)
    
    dataset_dir = "./raw_images"
    
    # 分析几个样本
    sample_files = []
    
    # 检查训练集
    train_raw_dir = os.path.join(dataset_dir, 'train', 'mediatek_raw')
    train_dslr_dir = os.path.join(dataset_dir, 'train', 'fujifilm')
    
    if os.path.exists(train_raw_dir):
        for i in range(min(3, len(os.listdir(train_raw_dir)))):  # 分析前3个文件
            raw_file = os.path.join(train_raw_dir, f"{i}.png")
            dslr_file = os.path.join(train_dslr_dir, f"{i}.png")
            
            if os.path.exists(raw_file):
                sample_files.append(('RAW训练', raw_file))
            if os.path.exists(dslr_file):
                sample_files.append(('DSLR训练', dslr_file))
    
    # 分析样本
    total_file_size = 0
    total_memory_size = 0
    valid_samples = 0
    
    for sample_type, file_path in sample_files:
        print(f"\n{sample_type}样本:")
        result = analyze_single_image(file_path)
        if result:
            total_file_size += result['file_size']
            total_memory_size += result['tensor_memory_size']
            valid_samples += 1
    
    if valid_samples > 0:
        avg_file_size = total_file_size / valid_samples
        avg_memory_size = total_memory_size / valid_samples
        avg_expansion = avg_memory_size / avg_file_size
        
        print(f"\n" + "=" * 60)
        print("平均统计")
        print("=" * 60)
        print(f"平均文件大小: {avg_file_size / 1024:.2f} KB")
        print(f"平均内存大小: {avg_memory_size / 1024:.2f} KB")
        print(f"平均内存膨胀比: {avg_expansion:.2f}x")
        
        # 估算完整数据集
        train_size = 24161
        val_size = 2258
        total_samples = train_size + val_size
        
        estimated_file_size = total_samples * avg_file_size / 1024 / 1024  # MB
        estimated_memory_size = total_samples * avg_memory_size / 1024 / 1024  # MB
        
        print(f"\n完整数据集估算:")
        print(f"文件总大小: {estimated_file_size:.1f} MB")
        print(f"内存总大小: {estimated_memory_size:.1f} MB")
        print(f"内存需求: {estimated_memory_size / 1024:.1f} GB")


def explain_memory_differences():
    """解释内存差异的原因"""
    print("\n" + "=" * 60)
    print("内存差异原因分析")
    print("=" * 60)
    
    explanations = [
        "1. 文件压缩:",
        "   - PNG/JPEG文件在磁盘上使用压缩算法",
        "   - 内存中存储的是未压缩的原始像素数据",
        "   - 压缩比通常为 2-10x",
        "",
        "2. 数据类型转换:",
        "   - 文件中: uint8 (1字节/像素)",
        "   - 内存中: float32 (4字节/像素)",
        "   - 内存膨胀: 4x",
        "",
        "3. 数据预处理:",
        "   - 拜耳模式处理: 单通道 → 4通道",
        "   - 归一化: 除以255.0",
        "   - 通道重排: HWC → CHW",
        "",
        "4. PyTorch张量开销:",
        "   - 张量元数据",
        "   - 内存对齐",
        "   - 梯度信息存储空间",
        "",
        "5. 系统内存分配:",
        "   - 内存碎片",
        "   - 操作系统页面对齐",
        "   - Python对象开销"
    ]
    
    for line in explanations:
        print(line)


def main():
    print("内存使用情况分析工具")
    print("=" * 60)
    
    # 解释内存差异
    explain_memory_differences()
    
    # 分析数据集
    analyze_dataset_memory()
    
    # 给出优化建议
    print("\n" + "=" * 60)
    print("内存优化建议")
    print("=" * 60)
    
    suggestions = [
        "1. 使用更高效的数据类型:",
        "   - float16 代替 float32 (减少50%内存)",
        "   - 量化训练",
        "",
        "2. 数据压缩:",
        "   - 使用内存映射文件",
        "   - 实时解压缩",
        "",
        "3. 批处理优化:",
        "   - 动态批大小",
        "   - 梯度累积",
        "",
        "4. 内存管理:",
        "   - 及时释放不用的张量",
        "   - 使用torch.no_grad()减少内存",
        "   - 定期垃圾回收"
    ]
    
    for line in suggestions:
        print(line)


if __name__ == "__main__":
    main() 