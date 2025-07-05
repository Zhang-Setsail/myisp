# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from torch.utils.data import Dataset
from torchvision import transforms
from scipy import misc
import numpy as np
import imageio
import torch
import os
import glob
import gc
import psutil
from tqdm import tqdm

to_tensor = transforms.Compose([
    transforms.ToTensor()
])


def extract_bayer_channels(raw):

    # Reshape the input bayer image

    ch_B  = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    RAW_norm = RAW_combined.astype(np.float32) / (4 * 255.0)

    return RAW_norm


def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB


class LoadData(Dataset):

    def __init__(self, dataset_dir, dataset_size, val=False):

        if val:
            self.raw_dir = os.path.join(dataset_dir, 'val', 'mediatek_raw')
            self.dslr_dir = os.path.join(dataset_dir, 'val', 'fujifilm')
        else:
            self.raw_dir = os.path.join(dataset_dir, 'train', 'mediatek_raw')
            self.dslr_dir = os.path.join(dataset_dir, 'train', 'fujifilm')

        self.dataset_size = dataset_size
        self.val = val

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        raw_image = np.asarray(imageio.imread(os.path.join(self.raw_dir, str(idx) + '.png')))
        raw_image = extract_bayer_channels(raw_image)
        raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))

        dslr_image = np.asarray(imageio.imread(os.path.join(self.dslr_dir, str(idx) + ".png")))
        dslr_image = np.float32(dslr_image) / 255.0
        dslr_image = torch.from_numpy(dslr_image.transpose((2, 0, 1)))

        return raw_image, dslr_image


class LoadDataMemory(Dataset):
    """将所有数据预加载到内存中的高性能数据加载器"""

    def __init__(self, dataset_dir, dataset_size, val=False, enable_memory_cache=True, use_float16=False):
        
        if val:
            self.raw_dir = os.path.join(dataset_dir, 'val', 'mediatek_raw')
            self.dslr_dir = os.path.join(dataset_dir, 'val', 'fujifilm')
        else:
            self.raw_dir = os.path.join(dataset_dir, 'train', 'mediatek_raw')
            self.dslr_dir = os.path.join(dataset_dir, 'train', 'fujifilm')

        self.dataset_size = dataset_size
        self.val = val
        self.enable_memory_cache = enable_memory_cache
        self.use_float16 = use_float16
        
        # 预加载所有数据到内存
        if self.enable_memory_cache:
            self._preload_data()
        
    def _preload_data(self):
        """预加载所有数据到内存"""
        dtype_str = "float16" if self.use_float16 else "float32"
        print(f"开始预加载数据到内存... 数据集大小: {self.dataset_size}, 数据类型: {dtype_str}")
        
        # 记录开始时的内存使用
        initial_memory = get_memory_usage()
        print(f"初始内存使用: {initial_memory:.1f} MB")
        
        # 预分配存储空间
        self.raw_images = []
        self.dslr_images = []
        
        # 使用tqdm显示进度
        for idx in tqdm(range(self.dataset_size), desc="预加载数据"):
            try:
                # 加载原始图像
                raw_image = np.asarray(imageio.imread(os.path.join(self.raw_dir, str(idx) + '.png')))
                raw_image = extract_bayer_channels(raw_image)
                raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))
                
                # 加载DSLR图像
                dslr_image = np.asarray(imageio.imread(os.path.join(self.dslr_dir, str(idx) + ".png")))
                if self.use_float16:
                    dslr_image = np.float16(dslr_image) / 255.0
                    raw_image = raw_image.half()  # 转换为float16
                else:
                    dslr_image = np.float32(dslr_image) / 255.0
                dslr_image = torch.from_numpy(dslr_image.transpose((2, 0, 1)))
                
                self.raw_images.append(raw_image)
                self.dslr_images.append(dslr_image)
                
            except Exception as e:
                print(f"加载图像 {idx} 时出错: {e}")
                # 如果某个图像加载失败，使用零张量占位
                if len(self.raw_images) > 0:
                    self.raw_images.append(torch.zeros_like(self.raw_images[0]))
                    self.dslr_images.append(torch.zeros_like(self.dslr_images[0]))
                else:
                    # 如果第一个图像就失败，创建默认尺寸
                    self.raw_images.append(torch.zeros((4, 256, 256)))
                    self.dslr_images.append(torch.zeros((3, 256, 256)))
        
        # 记录结束时的内存使用
        final_memory = get_memory_usage()
        memory_used = final_memory - initial_memory
        print(f"数据预加载完成!")
        print(f"最终内存使用: {final_memory:.1f} MB")
        print(f"数据占用内存: {memory_used:.1f} MB")
        print(f"平均每张图像占用内存: {memory_used / self.dataset_size:.2f} MB")
        
        # 强制垃圾回收
        gc.collect()

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.enable_memory_cache:
            # 从内存中直接返回数据
            raw_img, dslr_img = self.raw_images[idx], self.dslr_images[idx]
            
            # 如果使用float16存储，在使用时转换回float32
            if self.use_float16:
                raw_img = raw_img.float()
                dslr_img = dslr_img.float()
            
            return raw_img, dslr_img
        else:
            # 退回到原始的磁盘加载方式
            raw_image = np.asarray(imageio.imread(os.path.join(self.raw_dir, str(idx) + '.png')))
            raw_image = extract_bayer_channels(raw_image)
            raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))

            dslr_image = np.asarray(imageio.imread(os.path.join(self.dslr_dir, str(idx) + ".png")))
            dslr_image = np.float32(dslr_image) / 255.0
            dslr_image = torch.from_numpy(dslr_image.transpose((2, 0, 1)))

            return raw_image, dslr_image


class LoadVisualData(Dataset):

    def __init__(self, data_dir, size):

        self.raw_dir = data_dir
        self.dataset_size = size
        self.test_images = glob.glob(os.path.join(data_dir, "*.png"))


    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        raw_image = np.asarray(imageio.imread(self.test_images[idx]))
        raw_image = extract_bayer_channels(raw_image)
        raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))

        return raw_image


if __name__ == "__main__":
    dataset = LoadData("./raw_images", 10, val=False)
    print(dataset[0])
    visual_dataset = LoadVisualData("./visual", 4)
    print(visual_dataset[0])
    
    # 测试内存预加载
    print("\n测试内存预加载数据集...")
    memory_dataset = LoadDataMemory("./raw_images", 10, val=False)
    print(f"内存数据集大小: {len(memory_dataset)}")
    print(f"第一个样本shape: {memory_dataset[0][0].shape}, {memory_dataset[0][1].shape}")