import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    def forward(self, x):
        first, second = x.chunk(2, dim=1)
        return first * second


class NAFBlock(nn.Module):
    def __init__(self, channels, dw_expand=2, ffn_expand=2, dropout=0.0):
        super(NAFBlock, self).__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, 1)
        self.conv2 = nn.Conv2d(
            dw_channels,
            dw_channels,
            3,
            padding=1,
            groups=dw_channels,
        )
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, 1),
        )
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels, 1)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, 1)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        residual = x
        y = self.conv1(self.norm1(x))
        y = self.conv2(y)
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.dropout1(self.conv3(y))
        x = residual + y * self.beta

        y = self.conv4(self.norm2(x))
        y = self.sg(y)
        y = self.dropout2(self.conv5(y))
        return x + y * self.gamma


class NAFNetBackbone(nn.Module):
    def __init__(
        self,
        img_channels=9,
        out_channels=3,
        width=32,
        middle_blk_num=4,
        enc_blk_nums=(1, 1, 2),
        dec_blk_nums=(1, 1, 1),
    ):
        super(NAFNetBackbone, self).__init__()
        if len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError("enc_blk_nums and dec_blk_nums must have equal length")

        self.intro = nn.Conv2d(img_channels, width, 3, padding=1)
        self.ending = nn.Conv2d(width, out_channels, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = width
        for block_count in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(channels) for _ in range(block_count)])
            )
            self.downs.append(
                nn.Conv2d(channels, channels * 2, 2, stride=2)
            )
            channels *= 2

        self.middle = nn.Sequential(
            *[NAFBlock(channels) for _ in range(middle_blk_num)]
        )

        for block_count in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            channels //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(channels) for _ in range(block_count)])
            )

        self.padder_size = 2 ** len(enc_blk_nums)

    def _pad(self, x):
        height, width = x.shape[-2:]
        pad_height = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_width = (self.padder_size - width % self.padder_size) % self.padder_size
        return F.pad(x, (0, pad_width, 0, pad_height))

    def forward_features(self, x):
        height, width = x.shape[-2:]
        x = self._pad(x)
        x = self.intro(x)
        skips = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)

        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = up(x)
            x = x + skip
            x = decoder(x)

        return x[..., :height, :width]

    def forward(self, x):
        return self.ending(self.forward_features(x))


class NAFNetRefiner(nn.Module):
    """Lightweight NAFNet that predicts a residual correction over coarse ERRNet."""

    def __init__(
        self,
        width=32,
        middle_blk_num=4,
        enc_blk_nums=(1, 1, 2),
        dec_blk_nums=(1, 1, 1),
        delta_scale=1.0,
    ):
        super(NAFNetRefiner, self).__init__()
        self.delta_scale = delta_scale
        self.backbone = NAFNetBackbone(
            img_channels=9,
            out_channels=3,
            width=width,
            middle_blk_num=middle_blk_num,
            enc_blk_nums=enc_blk_nums,
            dec_blk_nums=dec_blk_nums,
        )
        nn.init.zeros_(self.backbone.ending.weight)
        nn.init.zeros_(self.backbone.ending.bias)

    def forward(self, input_image, errnet_output):
        residual = input_image - errnet_output
        features = torch.cat([input_image, errnet_output, residual], dim=1)
        correction = self.backbone(features)
        return errnet_output + correction * self.delta_scale
