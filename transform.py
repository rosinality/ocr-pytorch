import random

import numpy as np
from PIL import Image
import torch
from torch.nn.functional import pad
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)

        return img, target

    def __repr__(self):
        format_str = self.__class__.__name__ + '('
        for t in self.transforms:
            format_str += '\n'
            format_str += f'    {t}'
        format_str += '\n)'

        return format_str


class Resize:
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)

        self.min_size = min_size
        self.max_size = max_size

    def get_size(self, img_size):
        w, h = img_size
        size = random.choice(self.min_size)
        max_size = self.max_size

        if max_size is not None:
            min_orig = float(min((w, h)))
            max_orig = float(max((w, h)))

            if max_orig / min_orig * size > max_size:
                size = int(round(max_size * min_orig / max_orig))

        if (w <= h and w == size) or (h <= w and h == size):
            return h, w

        if w < h:
            ow = size
            oh = int(size * h / w)

        else:
            oh = size
            ow = int(size * w / h)

        return oh, ow

    def __call__(self, img, target):
        size = self.get_size(img.size)
        img = F.resize(img, size)
        target = F.resize(target, size, interpolation=Image.NEAREST)

        return img, target  


class RandomScale:
    def __init__(self, min_scale, max_scale):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img, target):
        w, h = img.size
        scale = random.uniform(self.min_scale, self.max_scale)
        h *= scale
        w *= scale
        size = (round(h), round(w))

        img = F.resize(img, size)
        target = F.resize(target, size, interpolation=Image.NEAREST)

        return img, target


class RandomBrightness:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img, target):
        factor = random.uniform(-self.factor, self.factor)
        img = F.adjust_brightness(img, 1 + factor)

        return img, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = F.hflip(img)
            target = F.hflip(target)

        return img, target


class RandomCrop:
    def __init__(self, size):
        if not isinstance(size, (list, tuple)):
            size = (size, size)

        self.size = size

    def __call__(self, img, target):
        w, h = img.size
        w_range = w - self.size[0]
        h_range = h - self.size[1]
        if w_range > 0:
            left = random.randint(0, w_range - 1)

        else:
            left = 0

        if h_range > 0:
            top = random.randint(0, h_range - 1)

        else:
            top = 0
            
        height = min(h - top, self.size[1])
        width = min(w - left, self.size[0])

        img = F.crop(img, top, left, height, width)
        target = F.crop(target, top, left, height, width)

        return img, target


class ToTensor:
    def __call__(self, img, target):
        target = torch.from_numpy(np.array(target, dtype=np.int64, copy=False))
        return F.to_tensor(img), target
    
    
class Pad:
    def __init__(self, size):
        self.size = size
        
    def __call__(self, img, target):
        _, h, w = img.shape
        
        if h == self.size and w == self.size:
            return img, target
        
        h_pad = self.size - h
        w_pad = self.size - w
        
        img = pad(img, [0, w_pad, 0, h_pad])
        target = pad(target, [0, w_pad, 0, h_pad])
        
        return img, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        img = F.normalize(img, mean=self.mean, std=self.std)

        return img, target
