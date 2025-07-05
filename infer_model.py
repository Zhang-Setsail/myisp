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

to_image = transforms.Compose([transforms.ToPILImage()])

visual_dir = "./visual"

VISUAL_SIZE = 4

visual_dataset = LoadVisualData(visual_dir, VISUAL_SIZE)
visual_loader = DataLoader(dataset=visual_dataset, batch_size=1, shuffle=False, num_workers=0,
                           pin_memory=True)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("mps")

print(f"Using device: {device}")

model = MicroISP().to(device)
model.load_state_dict(torch.load("models/model_86.pth", map_location=device))
model.eval()

for i, visual_data in enumerate(visual_loader):
    visual_data = visual_data.to(device)
    output = model(visual_data)
    output = output.cpu().clamp(0, 1)
    output = output.detach().numpy()
    output = output.squeeze(0)
    output = output.transpose(1, 2, 0)
    output = output * 255.0
    output = output.astype(np.uint8)
    imageio.imwrite(f"results/output_{i}.png", output)
