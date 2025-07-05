#!/usr/bin/env python3
"""
测试内存预加载功能的脚本
"""

import time
import torch
from torch.utils.data import DataLoader
from load_data import LoadData, LoadDataMemory
import os

def test_loading_performance():
    """测试不同数据加载方式的性能差异"""
    
    dataset_dir = "./raw_images"
    test_size = 50  # 使用较小的测试集
    batch_size = 4
    
    print("=" * 60)
    print("数据加载性能测试")
    print("=" * 60)
    
    # 测试传统磁盘加载
    print("\n1. 测试传统磁盘加载...")
    start_time = time.time()
    
    disk_dataset = LoadData(dataset_dir, test_size, val=False)
    disk_loader = DataLoader(dataset=disk_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=1, pin_memory=True)
    
    disk_times = []
    for i, (raw, dslr) in enumerate(disk_loader):
        batch_start = time.time()
        # 模拟一些处理
        _ = raw.shape, dslr.shape
        batch_time = time.time() - batch_start
        disk_times.append(batch_time)
        
        if i >= 10:  # 只测试前10个batch
            break
    
    disk_total_time = time.time() - start_time
    disk_avg_time = sum(disk_times) / len(disk_times)
    
    print(f"磁盘加载总时间: {disk_total_time:.3f}s")
    print(f"平均每个batch时间: {disk_avg_time:.3f}s")
    
    # 测试内存预加载
    print("\n2. 测试内存预加载...")
    start_time = time.time()
    
    memory_dataset = LoadDataMemory(dataset_dir, test_size, val=False, enable_memory_cache=True)
    memory_loader = DataLoader(dataset=memory_dataset, batch_size=batch_size, 
                             shuffle=False, num_workers=1, pin_memory=True)
    
    memory_times = []
    for i, (raw, dslr) in enumerate(memory_loader):
        batch_start = time.time()
        # 模拟一些处理
        _ = raw.shape, dslr.shape
        batch_time = time.time() - batch_start
        memory_times.append(batch_time)
        
        if i >= 10:  # 只测试前10个batch
            break
    
    memory_total_time = time.time() - start_time
    memory_avg_time = sum(memory_times) / len(memory_times)
    
    print(f"内存加载总时间: {memory_total_time:.3f}s")
    print(f"平均每个batch时间: {memory_avg_time:.3f}s")
    
    # 性能比较
    print("\n" + "=" * 60)
    print("性能比较结果")
    print("=" * 60)
    
    if disk_avg_time > 0:
        speedup = disk_avg_time / memory_avg_time
        print(f"内存预加载相对于磁盘加载的加速比: {speedup:.2f}x")
    
    total_speedup = disk_total_time / memory_total_time
    print(f"总体加速比: {total_speedup:.2f}x")
    
    print(f"\n磁盘加载平均batch时间: {disk_avg_time*1000:.2f}ms")
    print(f"内存加载平均batch时间: {memory_avg_time*1000:.2f}ms")
    print(f"性能提升: {((disk_avg_time - memory_avg_time) / disk_avg_time * 100):.1f}%")


def check_memory_usage():
    """检查内存使用情况"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"当前进程内存使用: {memory_info.rss / 1024 / 1024:.1f} MB")
        
        system_memory = psutil.virtual_memory()
        print(f"系统总内存: {system_memory.total / 1024 / 1024 / 1024:.1f} GB")
        print(f"系统可用内存: {system_memory.available / 1024 / 1024:.1f} MB")
        print(f"内存使用率: {system_memory.percent:.1f}%")
        
    except ImportError:
        print("请安装psutil库来查看内存使用情况: pip install psutil")


if __name__ == "__main__":
    print("开始测试内存预加载功能...")
    
    # 检查数据目录是否存在
    if not os.path.exists("./raw_images"):
        print("错误: 找不到数据目录 './raw_images'")
        print("请确保数据目录存在并包含训练数据")
        exit(1)
    
    # 检查内存使用情况
    print("\n检查系统内存...")
    check_memory_usage()
    
    # 运行性能测试
    try:
        test_loading_performance()
        print("\n测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        print("请检查数据目录和文件是否正确")
        
    # 最终内存使用情况
    print("\n最终内存使用情况:")
    check_memory_usage() 