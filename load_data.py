# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from torch.utils.data import Dataset
from torchvision import transforms
from scipy import misc
import numpy as np
import imageio
import torch
import os
import glob

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