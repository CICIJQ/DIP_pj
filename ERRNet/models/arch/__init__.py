# Add your custom network here
from .default import DRNet
from .cascade import CascadeRefinementNet
from .transformer import TransformerCascadeNet
import torch.nn as nn


def basenet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, bottom_kernel_size=1, **kwargs)


def errnet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True, **kwargs)


def errnet_cascade(in_channels, out_channels, **kwargs):
    return CascadeRefinementNet(in_channels, out_channels, **kwargs)


def errnet_transformer(in_channels, out_channels, **kwargs):
    return TransformerCascadeNet(in_channels, out_channels, **kwargs)
