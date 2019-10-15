from PIL import Image
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.utils import make_grid


@torch.no_grad()
def intersection_union(gt, pred, correct, n_class):
    intersect = pred * correct

    area_intersect = torch.histc(intersect, bins=n_class, min=1, max=n_class)
    area_pred = torch.histc(pred, bins=n_class, min=1, max=n_class)
    area_gt = torch.histc(gt, bins=n_class, min=1, max=n_class)

    # intersect = intersect.detach().to('cpu').numpy()
    # pred = pred.detach().to('cpu').numpy()
    # gt = gt.detach().to('cpu').numpy()
    # area_intersect, _ = np.histogram(intersect, bins=n_class, range=(1, n_class))
    # area_pred, _ = np.histogram(pred, bins=n_class, range=(1, n_class))
    # area_gt, _ = np.histogram(gt, bins=n_class, range=(1, n_class))

    area_union = area_pred + area_gt - area_intersect

    return area_intersect, area_union


def get_colormap(filename):
    colors = np.load(filename)
    colors = np.pad(colors, [(1, 0), (0, 0)], 'constant', constant_values=0)
    colors = torch.from_numpy(colors).type(torch.float32)

    return colors


@torch.no_grad()
def show_segmentation(img, gt, pred, mean, std, colormap):
    colormap = colormap.to(img.device)
    gt = F.embedding(gt, colormap).permute(2, 0, 1).div(255)
    pred = F.embedding(pred, colormap).permute(2, 0, 1).div(255)
    mean = torch.as_tensor(mean, dtype=torch.float32, device=img.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=img.device)
    img = img * std[:, None, None] + mean[:, None, None]
    grid = torch.stack([img, gt, pred], 0)
    grid = make_grid(grid, nrow=3)
    grid = (
        grid.mul_(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to('cpu', torch.uint8)
        .numpy()
    )
    img = Image.fromarray(grid)

    return img
