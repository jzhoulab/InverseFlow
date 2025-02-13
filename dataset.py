import torch
from torch.utils.data import Dataset
import numpy as np


class NoisyDataset(Dataset):
    def __init__(self, file_path, gt_tag, noisy_tag, transform=None):
        super().__init__()
        self.data = np.load(file_path)
        self.gt = self.data[gt_tag]
        self.noisy_data = self.data[noisy_tag]
        self.transform = transform

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        img_gt = torch.from_numpy(self.gt[idx])
        img_noisy = torch.from_numpy(self.noisy_data[idx])

        if self.transform is not None:
            channel = img_gt.shape[0]
            img_combine = torch.concat((img_gt, img_noisy), dim=0)
            img_combine = self.transform(img_combine)
            img_gt = img_combine[:channel, :, :]
            img_noisy = img_combine[channel:, :, :]

        return img_noisy, img_gt
