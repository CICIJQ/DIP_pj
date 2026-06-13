import torch
from torch import nn
import torch.nn.functional as F

from .default import DRNet


class RefineBlock(nn.Module):
    def __init__(self, channels):
        super(RefineBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1))

    def forward(self, x):
        return x + self.body(x) * 0.1


class RefinementNet(nn.Module):
    def __init__(self, in_channels=9, out_channels=3, n_feats=64, n_blocks=6):
        super(RefinementNet, self).__init__()
        blocks = [
            nn.Conv2d(in_channels, n_feats, kernel_size=3, padding=1),
            nn.ReLU(True),
        ]
        blocks.extend([RefineBlock(n_feats) for _ in range(n_blocks)])
        blocks.extend([
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(n_feats, out_channels, kernel_size=3, padding=1),
        ])
        self.body = nn.Sequential(*blocks)

    def forward(self, x):
        return self.body(x)


class CascadeRefinementNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_refine_feats=64, n_refine_blocks=6):
        super(CascadeRefinementNet, self).__init__()
        self.coarse_net = DRNet(
            in_channels,
            out_channels,
            256,
            13,
            norm=None,
            res_scale=0.1,
            se_reduction=8,
            bottom_kernel_size=1,
            pyramid=True)
        self.refine_net = RefinementNet(
            in_channels=out_channels * 3,
            out_channels=out_channels,
            n_feats=n_refine_feats,
            n_blocks=n_refine_blocks)
        self.last_coarse = None

    def forward(self, x):
        coarse = self.coarse_net(x)
        rgb_input = x[:, :coarse.shape[1]]
        if coarse.shape[2:] != rgb_input.shape[2:]:
            coarse = F.interpolate(coarse, size=rgb_input.shape[2:], mode='bilinear', align_corners=False)
        residual = rgb_input - coarse
        refine_input = torch.cat([rgb_input, coarse, residual], dim=1)
        delta = self.refine_net(refine_input)
        output = coarse + delta
        self.last_coarse = coarse
        return output
