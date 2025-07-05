#!/usr/bin/env python3
"""
快速内存测试脚本
"""

import os
import psutil
from load_data import LoadDataMemory
import gc

def quick_test():
    """快速测试内存使用情况"""
    
    dataset_dir = "./raw_images"
    test_size = 10  # 只测试10个样本
    
    print("快速内存测试")
    print("=" * 40)
    
    def get_memory():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    # 测试Float32
    print("\n测试Float32...")
    gc.collect()
    initial_memory = get_memory()
    
    dataset_f32 = LoadDataMemory(dataset_dir, test_size, val=False, 
                               enable_memory_cache=True, use_float16=False)
    memory_f32 = get_memory() - initial_memory
    print(f"Float32内存使用: {memory_f32:.1f} MB")
    
    del dataset_f32
    gc.collect()
    
    # 测试Float16
    print("\n测试Float16...")
    gc.collect()
    initial_memory = get_memory()
    
    dataset_f16 = LoadDataMemory(dataset_dir, test_size, val=False, 
                               enable_memory_cache=True, use_float16=True)
    memory_f16 = get_memory() - initial_memory
    print(f"Float16内存使用: {memory_f16:.1f} MB")
    
    # 计算节省
    if memory_f32 > 0:
        savings = (memory_f32 - memory_f16) / memory_f32 * 100
        print(f"\n内存节省: {savings:.1f}%")
    
    # 完整数据集估算
    full_size = 24161 + 2258
    scale = full_size / test_size
    
    estimated_f32 = memory_f32 * scale / 1024  # GB
    estimated_f16 = memory_f16 * scale / 1024  # GB
    
    print(f"\n完整数据集估算:")
    print(f"Float32: {estimated_f32:.1f} GB")
    print(f"Float16: {estimated_f16:.1f} GB")
    print(f"节省: {estimated_f32 - estimated_f16:.1f} GB")
    
    del dataset_f16
    gc.collect()

if __name__ == "__main__":
    if not os.path.exists("./raw_images"):
        print("错误: 找不到数据目录")
        exit(1)
        
    quick_test()
    print("\n✅ 测试完成！") 