#!/usr/bin/env python3
"""
测试float16内存预加载功能
"""

import time
import torch
from torch.utils.data import DataLoader
from load_data import LoadData, LoadDataMemory
import os
import gc

def test_float16_memory():
    """测试float16和float32的内存使用差异"""
    
    dataset_dir = "./raw_images"
    test_size = 100  # 使用更多样本进行测试
    batch_size = 4
    
    print("=" * 60)
    print("Float16 vs Float32 内存使用测试")
    print("=" * 60)
    
    def get_memory_usage():
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0
    
    # 测试Float32内存预加载
    print("\n1. 测试Float32内存预加载...")
    gc.collect()
    initial_memory = get_memory_usage()
    
    start_time = time.time()
    memory_dataset_f32 = LoadDataMemory(dataset_dir, test_size, val=False, 
                                      enable_memory_cache=True, use_float16=False)
    load_time_f32 = time.time() - start_time
    
    memory_f32 = get_memory_usage()
    data_memory_f32 = memory_f32 - initial_memory
    
    print(f"Float32 - 加载时间: {load_time_f32:.2f}s")
    print(f"Float32 - 数据内存使用: {data_memory_f32:.1f} MB")
    
    # 测试数据加载速度
    f32_loader = DataLoader(dataset=memory_dataset_f32, batch_size=batch_size, 
                           shuffle=False, num_workers=1, pin_memory=True)
    
    start_time = time.time()
    for i, (raw, dslr) in enumerate(f32_loader):
        if i >= 10:  # 只测试前10个batch
            break
    f32_batch_time = time.time() - start_time
    
    print(f"Float32 - 10个batch加载时间: {f32_batch_time:.3f}s")
    
    # 清理内存
    del memory_dataset_f32, f32_loader
    gc.collect()
    
    # 测试Float16内存预加载
    print("\n2. 测试Float16内存预加载...")
    gc.collect()
    initial_memory = get_memory_usage()
    
    start_time = time.time()
    memory_dataset_f16 = LoadDataMemory(dataset_dir, test_size, val=False, 
                                      enable_memory_cache=True, use_float16=True)
    load_time_f16 = time.time() - start_time
    
    memory_f16 = get_memory_usage()
    data_memory_f16 = memory_f16 - initial_memory
    
    print(f"Float16 - 加载时间: {load_time_f16:.2f}s")
    print(f"Float16 - 数据内存使用: {data_memory_f16:.1f} MB")
    
    # 测试数据加载速度
    f16_loader = DataLoader(dataset=memory_dataset_f16, batch_size=batch_size, 
                           shuffle=False, num_workers=1, pin_memory=True)
    
    start_time = time.time()
    for i, (raw, dslr) in enumerate(f16_loader):
        if i >= 10:  # 只测试前10个batch
            break
    f16_batch_time = time.time() - start_time
    
    print(f"Float16 - 10个batch加载时间: {f16_batch_time:.3f}s")
    
    # 检查数据类型
    sample_raw, sample_dslr = next(iter(f16_loader))
    print(f"Float16 - 输出数据类型: Raw={sample_raw.dtype}, DSLR={sample_dslr.dtype}")
    
    # 性能比较
    print("\n" + "=" * 60)
    print("性能比较结果")
    print("=" * 60)
    
    memory_reduction = (data_memory_f32 - data_memory_f16) / data_memory_f32 * 100
    print(f"内存节省: {memory_reduction:.1f}%")
    print(f"内存使用比较: Float32={data_memory_f32:.1f}MB vs Float16={data_memory_f16:.1f}MB")
    
    time_diff = (load_time_f16 - load_time_f32) / load_time_f32 * 100
    print(f"加载时间变化: {time_diff:+.1f}%")
    
    batch_time_diff = (f16_batch_time - f32_batch_time) / f32_batch_time * 100
    print(f"批处理时间变化: {batch_time_diff:+.1f}%")
    
    # 完整数据集估算
    print("\n完整数据集估算:")
    full_dataset_size = 24161 + 2258
    scale_factor = full_dataset_size / test_size
    
    estimated_f32_memory = data_memory_f32 * scale_factor
    estimated_f16_memory = data_memory_f16 * scale_factor
    
    print(f"Float32完整数据集内存需求: {estimated_f32_memory:.1f} MB ({estimated_f32_memory/1024:.1f} GB)")
    print(f"Float16完整数据集内存需求: {estimated_f16_memory:.1f} MB ({estimated_f16_memory/1024:.1f} GB)")
    print(f"Float16可节省内存: {(estimated_f32_memory - estimated_f16_memory)/1024:.1f} GB")
    
    # 清理内存
    del memory_dataset_f16, f16_loader
    gc.collect()


def main():
    print("Float16内存优化测试")
    print("=" * 60)
    
    # 检查数据目录
    if not os.path.exists("./raw_images"):
        print("错误: 找不到数据目录 './raw_images'")
        exit(1)
    
    # 运行测试
    try:
        test_float16_memory()
        print("\n✅ 测试完成！")
        
        print("\n推荐配置:")
        print("- 如果内存充足(>16GB): 使用Float32获得最佳精度")
        print("- 如果内存有限(8-16GB): 使用Float16节省内存")
        print("- 如果内存很少(<8GB): 禁用内存预加载")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 