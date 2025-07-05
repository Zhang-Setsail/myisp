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

from load_data import LoadData, LoadVisualData
from msssim import MSSSIM
# from model import PyNET
from model import MicroISP
from vgg import vgg_19
from utils import normalize_batch, process_command_args

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

# TRAIN_SIZE = 24161
# VAL_SIZE = 2258
TRAIN_SIZE = 100
VAL_SIZE = 10

VISUAL_SIZE = 1


def train_model():

    # torch.backends.cudnn.deterministic = True
    # device = torch.device("cuda")

    # print("CUDA visible devices: " + str(torch.cuda.device_count()))
    # print("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Creating dataset loaders

    train_dataset = LoadData(dataset_dir, TRAIN_SIZE, val=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                              pin_memory=True)

    val_dataset = LoadData(dataset_dir, VAL_SIZE, val=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=1,
                             pin_memory=True)

    visual_dataset = LoadVisualData(visual_dir, VISUAL_SIZE)
    visual_loader = DataLoader(dataset=visual_dataset, batch_size=1, shuffle=False, num_workers=0,
                               pin_memory=True)

    # Creating image processing network and optimizer
    generator = MicroISP(input_channels=4, output_channels=3).to(device)

    optimizer = Adam(params=generator.parameters(), lr=learning_rate)

    # Losses

    VGG_19 = vgg_19(device)
    MSE_loss = torch.nn.MSELoss()
    MS_SSIM = MSSSIM()

    # Train the network

    for epoch in range(num_train_epochs):

        train_iter = iter(train_loader)
        for i in tqdm(range(len(train_loader))):

            optimizer.zero_grad()
            x, y = next(train_iter)

            x = x.to(device)
            y = y.to(device)

            enhanced = generator(x)

            # MSE Loss
            loss_mse = MSE_loss(enhanced, y)

            # VGG Loss
            enhanced_vgg = VGG_19(normalize_batch(enhanced))
            target_vgg = VGG_19(normalize_batch(y))
            loss_content = MSE_loss(enhanced_vgg, target_vgg)

            # Total Loss
            loss_ssim = MS_SSIM(enhanced, y)
            total_loss = loss_mse + loss_content + (1 - loss_ssim) * 0.4

            # Perform the optimization step

            total_loss.backward()
            optimizer.step()

            if i == 0 and epoch > 0:

                # Save the model that corresponds to the current epoch

                generator.eval().cpu()
                torch.save(generator.state_dict(), "models/isp_model_epoch_" + str(epoch) + ".pth")
                generator.to(device).train()

                # Save visual results for several test images

                generator.eval()
                with torch.no_grad():

                    visual_iter = iter(visual_loader)
                    for j in range(len(visual_loader)):

                        torch.cuda.empty_cache()

                        raw_image = next(visual_iter)
                        raw_image = raw_image.to(device, non_blocking=True)

                        enhanced = generator(raw_image.detach())
                        enhanced = np.asarray(to_image(torch.squeeze(enhanced.detach().cpu())))

                        cv2.imwrite("results/isp_img_" + str(j) + "_epoch_" +
                                        str(epoch) + ".jpg", enhanced[::,::,::-1])

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

                        loss_mse_eval += loss_mse_temp
                        loss_psnr_eval += 20 * math.log10(1.0 / math.sqrt(loss_mse_temp))

                        loss_ssim_eval += MS_SSIM(y, enhanced)

                        enhanced_vgg_eval = VGG_19(normalize_batch(enhanced)).detach()
                        target_vgg_eval = VGG_19(normalize_batch(y)).detach()

                        loss_vgg_eval += MSE_loss(enhanced_vgg_eval, target_vgg_eval).item()

                loss_mse_eval = loss_mse_eval / VAL_SIZE
                loss_psnr_eval = loss_psnr_eval / VAL_SIZE
                loss_vgg_eval = loss_vgg_eval / VAL_SIZE
                loss_ssim_eval = loss_ssim_eval / VAL_SIZE

                print("Epoch %d, mse: %.4f, psnr: %.4f, vgg: %.4f, ms-ssim: %.4f" % (epoch,
                        loss_mse_eval, loss_psnr_eval, loss_vgg_eval, loss_ssim_eval))

                generator.train()


if __name__ == '__main__':
    train_model()

