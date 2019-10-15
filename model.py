import torch
from torch import nn
from torch.nn import functional as F


def conv2d(in_channel, out_channel, kernel_size):
    layers = [
        nn.Conv2d(
            in_channel, out_channel, kernel_size, padding=kernel_size // 2, bias=False
        ),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
    ]

    return nn.Sequential(*layers)


def conv1d(in_channel, out_channel):
    layers = [
        nn.Conv1d(in_channel, out_channel, 1, bias=False),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(),
    ]

    return nn.Sequential(*layers)


class OCR(nn.Module):
    def __init__(self, n_class, backbone, feat_channels=[768, 1024]):
        super().__init__()

        self.backbone = backbone

        ch16, ch32 = feat_channels

        self.L = nn.Conv2d(ch16, n_class, 1)
        self.X = conv2d(ch32, 512, 3)

        self.phi = conv1d(512, 256)
        self.psi = conv1d(512, 256)
        self.delta = conv1d(512, 256)
        self.rho = conv1d(256, 512)
        self.g = conv2d(512 + 512, 512, 1)

        self.out = nn.Conv2d(512, n_class, 1)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, input, target=None):
        input_size = input.shape[2:]
        stg16, stg32 = self.backbone(input)[-2:]

        X = self.X(stg32)
        L = self.L(stg16)
        batch, n_class, height, width = L.shape
        l_flat = L.view(batch, n_class, -1)
        # M: NKL
        M = torch.softmax(l_flat, -1)
        channel = X.shape[1]
        X_flat = X.view(batch, channel, -1)
        # f_k: NCK
        f_k = (M @ X_flat.transpose(1, 2)).transpose(1, 2)

        # query: NKD
        query = self.phi(f_k).transpose(1, 2)
        # key: NDL
        key = self.psi(X_flat)
        logit = query @ key
        # attn: NKL
        attn = torch.softmax(logit, 1)

        # delta: NDK
        delta = self.delta(f_k)
        # attn_sum: NDL
        attn_sum = delta @ attn
        # x_obj = NCHW
        X_obj = self.rho(attn_sum).view(batch, -1, height, width)

        concat = torch.cat([X, X_obj], 1)
        X_bar = self.g(concat)
        out = self.out(X_bar)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)

        if self.training:
            aux_out = F.interpolate(
                L, size=input_size, mode='bilinear', align_corners=False
            )

            loss = self.criterion(out, target)
            aux_loss = self.criterion(aux_out, target)

            return {'loss': loss, 'aux': aux_loss}, None

        else:
            return {}, out
