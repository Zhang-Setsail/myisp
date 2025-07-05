# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam

import torch
import imageio
import numpy as np
import math
import sys
from tqdm import tqdm
import cv2
import os

from load_data import LoadData, LoadVisualData, LoadDataMemory
from msssim import MSSSIM
# from model import PyNET
from model import MicroISP
from vgg import vgg_19
from utils import normalize_batch

to_image = transforms.Compose([transforms.ToPILImage()])

np.random.seed(0)
torch.manual_seed(0)

# Processing command arguments

# batch_size, learning_rate, num_train_epochs, dataset_dir, visual_dir = process_command_args(sys.argv)
batch_size = 16
learning_rate = 0.001
num_train_epochs = 100
dataset_dir = "./raw_images"
visual_dir = "./visual"

# Dataset size

TRAIN_SIZE = 24161
VAL_SIZE = 2258
# TRAIN_SIZE = 100
# VAL_SIZE = 10

VISUAL_SIZE = 1

# 优化参数
EVAL_INTERVAL = 10  # 每10个epoch评估一次，而不是每个epoch
SAVE_INTERVAL = 10  # 每10个epoch保存一次模型
VISUAL_INTERVAL = 20  # 每20个epoch保存一次可视化结果

# 性能优化选项
USE_MIXED_PRECISION = True  # 是否使用混合精度训练
REDUCE_VGG_FREQUENCY = True  # 是否减少VGG损失计算频率
VGG_LOSS_FREQUENCY = 5  # 每5个batch计算一次VGG损失（如果启用）
USE_MEMORY_CACHE = True  # 是否将数据集预加载到内存中
USE_FLOAT16_CACHE = False  # 是否使用float16存储预加载数据（减少50%内存）
MEMORY_CACHE_WARNING_THRESHOLD = 16000  # 内存缓存警告阈值（MB）


def train_model():

    # torch.backends.cudnn.deterministic = True
    # device = torch.device("cuda")

    # print("CUDA visible devices: " + str(torch.cuda.device_count()))
    # print("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # 内存检查
    use_memory_cache = USE_MEMORY_CACHE  # 创建局部变量
    if use_memory_cache:
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / 1024 / 1024  # MB
            
            # 更准确的内存估算（基于实际测量）
            # RAW图像: 256KB, DSLR图像: 768KB, 平均: 512KB
            memory_per_sample = 0.5 if not USE_FLOAT16_CACHE else 0.25  # float16减少50%内存
            estimated_memory_usage = (TRAIN_SIZE + VAL_SIZE) * memory_per_sample
            
            print(f"可用内存: {available_memory:.1f} MB")
            print(f"估计数据集内存需求: {estimated_memory_usage:.1f} MB ({estimated_memory_usage/1024:.1f} GB)")
            
            if estimated_memory_usage > available_memory * 0.7:
                print("⚠️  警告: 估计的内存需求可能超过可用内存的70%")
                print("考虑:")
                print("1. 设置 USE_MEMORY_CACHE = False 来禁用内存预加载")
                print("2. 减少 TRAIN_SIZE 和 VAL_SIZE")
                print("3. 增加系统内存")
                print("4. 使用 float16 精度训练（可减少50%内存）")
                
                response = input("是否继续使用内存预加载? (y/n): ")
                if response.lower() != 'y':
                    use_memory_cache = False
                    print("已禁用内存预加载")
            else:
                print("✅ 内存充足，建议使用内存预加载以提高性能")
                    
        except ImportError:
            print("警告: 未安装psutil，无法进行内存检查")
    
    # 优化数据加载器性能
    if use_memory_cache:
        # 使用内存预加载时，减少worker数量（因为不需要频繁的磁盘I/O）
        num_workers = min(2, os.cpu_count())
        persistent_workers = False
        print("使用内存预加载数据集...")
    else:
        # 使用磁盘加载时，增加worker数量来提高I/O并发
        num_workers = min(8, os.cpu_count())
        persistent_workers = True
        print("使用磁盘加载数据集...")
    
    # Creating dataset loaders
    if use_memory_cache:
        train_dataset = LoadDataMemory(dataset_dir, TRAIN_SIZE, val=False, 
                                     enable_memory_cache=True, use_float16=USE_FLOAT16_CACHE)
        val_dataset = LoadDataMemory(dataset_dir, VAL_SIZE, val=True, 
                                   enable_memory_cache=True, use_float16=USE_FLOAT16_CACHE)
    else:
        train_dataset = LoadData(dataset_dir, TRAIN_SIZE, val=False)
        val_dataset = LoadData(dataset_dir, VAL_SIZE, val=True)
        
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, persistent_workers=persistent_workers)

    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True, persistent_workers=persistent_workers)

    visual_dataset = LoadVisualData(visual_dir, VISUAL_SIZE)
    visual_loader = DataLoader(dataset=visual_dataset, batch_size=1, shuffle=False, num_workers=1,
                               pin_memory=True)

    # Creating image processing network and optimizer
    generator = MicroISP(input_channels=4, output_channels=3).to(device)

    optimizer = Adam(params=generator.parameters(), lr=learning_rate)

    # generator.load_state_dict(torch.load("models/model_86.pth", map_location=device))

    # Losses

    VGG_19 = vgg_19(device)
    MSE_loss = torch.nn.MSELoss()
    MS_SSIM = MSSSIM()

    # 预分配一些变量以减少内存分配开销
    generator.train()
    
    # 开启一些PyTorch优化
    torch.backends.cudnn.benchmark = True  # 优化cudnn性能
    
    # 使用混合精度训练可以进一步提升性能
    scaler = torch.cuda.amp.GradScaler() if (device.type == 'cuda' and USE_MIXED_PRECISION) else None

    # Train the network

    for epoch in range(num_train_epochs):

        train_iter = iter(train_loader)
        epoch_loss = 0.0
        
        for i in tqdm(range(len(train_loader)), desc=f"Epoch {epoch+1}/{num_train_epochs}"):

            optimizer.zero_grad()
            x, y = next(train_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # 使用混合精度训练（如果可用）
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    enhanced = generator(x)

                    # MSE Loss
                    loss_mse = MSE_loss(enhanced, y)

                    # VGG Loss (可选择性计算以提高性能)
                    if not REDUCE_VGG_FREQUENCY or (i % VGG_LOSS_FREQUENCY == 0):
                        enhanced_vgg = VGG_19(normalize_batch(enhanced))
                        target_vgg = VGG_19(normalize_batch(y))
                        loss_content = MSE_loss(enhanced_vgg, target_vgg)
                    else:
                        loss_content = torch.tensor(0.0, device=device)

                    # Total Loss
                    loss_ssim = MS_SSIM(enhanced, y)
                    total_loss = loss_mse + loss_content + (1 - loss_ssim) * 0.4

                # Perform the optimization step
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                enhanced = generator(x)

                # MSE Loss
                loss_mse = MSE_loss(enhanced, y)

                # VGG Loss (可选择性计算以提高性能)
                if not REDUCE_VGG_FREQUENCY or (i % VGG_LOSS_FREQUENCY == 0):
                    enhanced_vgg = VGG_19(normalize_batch(enhanced))
                    target_vgg = VGG_19(normalize_batch(y))
                    loss_content = MSE_loss(enhanced_vgg, target_vgg)
                else:
                    loss_content = torch.tensor(0.0, device=device)

                # Total Loss
                loss_ssim = MS_SSIM(enhanced, y)
                total_loss = loss_mse + loss_content + (1 - loss_ssim) * 0.4

                # Perform the optimization step
                total_loss.backward()
                optimizer.step()

            epoch_loss += total_loss.item()

        # 打印每个epoch的平均损失
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        # 只在指定间隔进行评估和保存
        if (epoch + 1) % SAVE_INTERVAL == 0:
            # 保存模型时不需要切换到CPU
            torch.save(generator.state_dict(), f"models/isp_model_epoch_{epoch+1}.pth")
            print(f"Model saved at epoch {epoch+1}")

        if (epoch + 1) % VISUAL_INTERVAL == 0:
            # Save visual results for several test images
            generator.eval()
            with torch.no_grad():
                visual_iter = iter(visual_loader)
                for j in range(len(visual_loader)):
                    raw_image = next(visual_iter)
                    raw_image = raw_image.to(device, non_blocking=True)

                    enhanced = generator(raw_image)
                    enhanced = np.asarray(to_image(torch.squeeze(enhanced.detach().cpu())))

                    cv2.imwrite(f"results/isp_img_{j}_epoch_{epoch+1}.jpg", enhanced[::,::,::-1])
            generator.train()

        if (epoch + 1) % EVAL_INTERVAL == 0:
            # Evaluate the model
            loss_mse_eval = 0
            loss_psnr_eval = 0
            loss_vgg_eval = 0
            loss_ssim_eval = 0

            generator.eval()
            with torch.no_grad():
                val_iter = iter(val_loader)
                for j in range(len(val_loader)):
                    x, y = next(val_iter)
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    enhanced = generator(x)

                    loss_mse_temp = MSE_loss(enhanced, y).item()

                    loss_mse_eval += loss_mse_temp * enhanced.shape[0]
                    loss_psnr_eval += 20 * math.log10(1.0 / math.sqrt(loss_mse_temp)) * enhanced.shape[0]

                    loss_ssim_eval += MS_SSIM(y, enhanced) * enhanced.shape[0]

                    enhanced_vgg_eval = VGG_19(normalize_batch(enhanced))
                    target_vgg_eval = VGG_19(normalize_batch(y))

                    loss_vgg_eval += MSE_loss(enhanced_vgg_eval, target_vgg_eval).item() * enhanced.shape[0]

            loss_mse_eval = loss_mse_eval / VAL_SIZE
            loss_psnr_eval = loss_psnr_eval / VAL_SIZE
            loss_vgg_eval = loss_vgg_eval / VAL_SIZE
            loss_ssim_eval = loss_ssim_eval / VAL_SIZE

            print("Epoch %d, mse: %.4f, psnr: %.4f, vgg: %.4f, ms-ssim: %.4f" % (epoch+1,
                    loss_mse_eval, loss_psnr_eval, loss_vgg_eval, loss_ssim_eval))

            generator.train()


if __name__ == '__main__':
    train_model()

