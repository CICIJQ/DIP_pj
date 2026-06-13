import torch
from torch import nn
import torch.nn.functional as F

from .default import ConvLayer, DRNet


class PatchTransformerBlock(nn.Module):
    def __init__(self, channels, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super(PatchTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)

        hidden_dim = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
        )

    def forward(self, x):
        # x: B x C x H x W
        b, c, h, w = x.size()
        x_flat = x.flatten(2).transpose(1, 2)  # B x N x C

        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_out = x_flat + attn_out
        x_out = x_out + self.mlp(self.norm2(x_out))

        x_out = x_out.transpose(1, 2).view(b, c, h, w)
        return x_out


class TransformerRefineNet(nn.Module):
    def __init__(self, in_channels=9, feat_channels=128, n_blocks=3, n_heads=8, max_attn_size=112):
        super(TransformerRefineNet, self).__init__()
        self.max_attn_size = max_attn_size
        self.embed = nn.Sequential(
            ConvLayer(nn.Conv2d, in_channels, feat_channels, kernel_size=3, stride=1, padding=1, norm=None, act=nn.ReLU(True)),
            ConvLayer(nn.Conv2d, feat_channels, feat_channels, kernel_size=3, stride=2, padding=1, norm=None, act=nn.ReLU(True)),
        )

        self.transformer_blocks = nn.Sequential(*[
            PatchTransformerBlock(feat_channels, num_heads=n_heads) for _ in range(n_blocks)
        ])

        self.tail = nn.Sequential(
            ConvLayer(nn.Conv2d, feat_channels, feat_channels, kernel_size=3, stride=1, padding=1, norm=None, act=nn.ReLU(True)),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvLayer(nn.Conv2d, feat_channels, feat_channels, kernel_size=3, stride=1, padding=1, norm=None, act=nn.ReLU(True)),
            ConvLayer(nn.Conv2d, feat_channels, 3, kernel_size=1, stride=1, norm=None, act=None),
        )

    def forward(self, x):
        x = self.embed(x)
        embed_size = x.shape[2:]

        if self.max_attn_size is not None and max(embed_size) > self.max_attn_size:
            scale = float(self.max_attn_size) / float(max(embed_size))
            attn_size = (
                max(1, int(round(embed_size[0] * scale))),
                max(1, int(round(embed_size[1] * scale))),
            )
            x = F.interpolate(x, size=attn_size, mode='bilinear', align_corners=False)

        x = self.transformer_blocks(x)

        if x.shape[2:] != embed_size:
            x = F.interpolate(x, size=embed_size, mode='bilinear', align_corners=False)

        x = self.tail(x)
        return x


class TransformerCascadeNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_refine_feats=128, n_refine_blocks=3, n_refine_heads=8, max_attn_size=112):
        super(TransformerCascadeNet, self).__init__()
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
        self.refine_net = TransformerRefineNet(
            in_channels=out_channels * 3,
            feat_channels=n_refine_feats,
            n_blocks=n_refine_blocks,
            n_heads=n_refine_heads,
            max_attn_size=max_attn_size)
        self.last_coarse = None

    def forward(self, x):
        coarse = self.coarse_net(x)
        rgb_input = x[:, :coarse.shape[1]]
        if coarse.shape[2:] != rgb_input.shape[2:]:
            coarse = F.interpolate(coarse, size=rgb_input.shape[2:], mode='bilinear', align_corners=False)

        residual = rgb_input - coarse
        refine_input = torch.cat([rgb_input, coarse, residual], dim=1)
        delta = self.refine_net(refine_input)
        if delta.shape[2:] != coarse.shape[2:]:
            delta = F.interpolate(delta, size=coarse.shape[2:], mode='bilinear', align_corners=False)
        output = coarse + delta
        self.last_coarse = coarse
        return output


def errnet_transformer(in_channels, out_channels, **kwargs):
    return TransformerCascadeNet(in_channels, out_channels, **kwargs)
