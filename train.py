import argparse
import os
import math

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm

from model import OCR
from vovnet import vovnet39, vovnet57
import transform
from dataset import ADE20K, collate_data
from util import get_colormap, show_segmentation, intersection_union
from scheduler import CycleScheduler
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    DistributedSampler,
    all_gather,
)


def train(args, epoch, loader, model, optimizer, scheduler):
    torch.backends.cudnn.benchmark = True

    model.train()

    if get_rank() == 0:
        pbar = tqdm(loader, dynamic_ncols=True)

    else:
        pbar = loader

    for i, (img, annot) in enumerate(pbar):
        img = img.to('cuda')
        annot = annot.to('cuda')

        loss, _ = model(img, annot)
        loss_sum = loss['loss'] + args.aux_weight * loss['aux']
        model.zero_grad()
        loss_sum.backward()
        optimizer.step()
        scheduler.step()

        loss_dict = reduce_loss_dict(loss)
        loss = loss_dict['loss'].mean().item()
        aux_loss = loss_dict['aux'].mean().item()

        if get_rank() == 0:
            lr = optimizer.param_groups[0]['lr']

            pbar.set_description(
                f'epoch: {epoch + 1}; loss: {loss:.5f}; aux loss: {aux_loss:.5f}; lr: {lr:.5f}'
            )


@torch.no_grad()
def valid(args, epoch, loader, model, show):
    torch.backends.cudnn.benchmark = False

    model.eval()

    if get_rank() == 0:
        pbar = tqdm(loader, dynamic_ncols=True)

    else:
        pbar = loader

    intersect_sum = None
    union_sum = None
    correct_sum = 0
    total_sum = 0

    for i, (img, annot) in enumerate(pbar):
        img = img.to('cuda')
        annot = annot.to('cuda')
        _, out = model(img)
        _, pred = out.max(1)

        if get_rank() == 0 and i % 10 == 0:
            result = show(img[0], annot[0], pred[0])
            result.save(f'sample/{str(epoch + 1).zfill(3)}-{str(i).zfill(4)}.png')

        pred = (annot > 0) * pred
        correct = (pred > 0) * (pred == annot)
        correct_sum += correct.sum().float().item()
        total_sum += (annot > 0).sum().float()

        for g, p, c in zip(annot, pred, correct):
            intersect, union = intersection_union(g, p, c, args.n_class)

            if intersect_sum is None:
                intersect_sum = intersect

            else:
                intersect_sum += intersect

            if union_sum is None:
                union_sum = union

            else:
                union_sum += union

        all_intersect = sum(all_gather(intersect_sum.to('cpu')))
        all_union = sum(all_gather(union_sum.to('cpu')))

        if get_rank() == 0:
            iou = all_intersect / (all_union + 1e-10)
            m_iou = iou.mean().item()

            pbar.set_description(
                f'acc: {correct_sum / total_sum:.5f}; mIoU: {m_iou:.5f}'
            )


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return sampler.RandomSampler(dataset)

    else:
        return sampler.SequentialSampler(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--size', type=int, default=520)
    parser.add_argument('--arch', type=str, default='vovnet39')
    parser.add_argument('--aux_weight', type=float, default=0.4)
    parser.add_argument('--n_class', type=int, default=150)
    parser.add_argument('--lr', type=float, default=2e-2)
    parser.add_argument('--l2', type=float, default=1e-4)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('path', metavar='PATH')

    args = parser.parse_args()

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    device = 'cuda'
    # torch.backends.cudnn.deterministic = True

    train_trans = transform.Compose(
        [
            transform.RandomScale(0.5, 2.0),
            # transform.Resize(args.size, None),
            transform.RandomHorizontalFlip(),
            transform.RandomCrop(args.size),
            transform.RandomBrightness(0.04),
            transform.ToTensor(),
            transform.Normalize(img_mean, img_std),
            transform.Pad(args.size)
        ]
    )

    valid_trans = transform.Compose(
        [transform.ToTensor(), transform.Normalize(img_mean, img_std)]
    )

    train_set = ADE20K(args.path, 'train', train_trans)
    valid_set = ADE20K(args.path, 'valid', valid_trans)

    arch_map = {'vovnet39': vovnet39, 'vovnet57': vovnet57}
    backbone = arch_map[args.arch](pretrained=True)
    model = OCR(args.n_class + 1, backbone).to(device)

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr / 25,
        momentum=0.9,
        weight_decay=args.l2,
        nesterov=True,
    )

    max_iter = math.ceil(len(train_set) / (n_gpu * args.batch)) * args.epoch

    scheduler = CycleScheduler(
        optimizer,
        args.lr,
        n_iter=max_iter,
        warmup_proportion=0.01,
        phase=('linear', 'poly'),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        num_workers=2,
        sampler=data_sampler(train_set, shuffle=True, distributed=args.distributed),
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch,
        num_workers=2,
        sampler=data_sampler(valid_set, shuffle=False, distributed=args.distributed),
        collate_fn=collate_data,
    )

    colormap = get_colormap('color150.npy')

    def show_result(img, gt, pred):
        return show_segmentation(img, gt, pred, img_mean, img_std, colormap)

    for i in range(args.epoch):
        train(args, i, train_loader, model, optimizer, scheduler)
        valid(args, i, valid_loader, model, show_result)

        if get_rank() == 0:
            torch.save(
                {'model': model.module.state_dict(), 'args': args},
                f'checkpoint/epoch-{str(i + 1).zfill(3)}.pt',
            )

