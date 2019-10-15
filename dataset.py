import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class ADE20K(Dataset):
    def __init__(self, path, split, transform=None):
        split_path = {'train': 'training', 'valid': 'validation'}
        self.img_path = os.path.join(path, 'images', split_path[split])
        self.annot_path = os.path.join(path, 'annotations', split_path[split])
        files = os.listdir(self.img_path)
        self.ids = []

        for file in files:
            name, ext = os.path.splitext(file)
            if ext.lower() == '.jpg':
                self.ids.append(name)

        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        img = Image.open(os.path.join(self.img_path, id) + '.jpg').convert('RGB')
        annot = Image.open(os.path.join(self.annot_path, id) + '.png')

        if self.transform is not None:
            img, annot = self.transform(img, annot)

        return img, annot


def collate_data(batch):
    max_height = max([b[0].shape[1] for b in batch])
    max_width = max([b[0].shape[2] for b in batch])
    batch_size = len(batch)

    img_batch = torch.zeros(batch_size, 3, max_height, max_width, dtype=torch.float32)
    annot_batch = torch.zeros(batch_size, max_height, max_width, dtype=torch.int64)

    for i, (img, annot) in enumerate(batch):
        _, height, width = img.shape
        img_batch[i, :, :height, :width] = img
        annot_batch[i, :height, :width] = annot

    return img_batch, annot_batch
