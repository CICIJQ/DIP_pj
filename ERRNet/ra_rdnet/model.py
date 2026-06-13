import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + 0.2 * self.body(x)


class MaskGuidedResidualRefiner(nn.Module):
    """Small mask-guided residual refiner for RDNet outputs.

    Inputs are concatenated as [I, T_rd, I - T_rd, M].  The network predicts a
    bounded RGB residual and applies it through the reflection mask, so low-mask
    regions stay close to the original RDNet output.
    """

    def __init__(self, width=32, num_blocks=6, residual_scale=0.35):
        super().__init__()
        self.residual_scale = float(residual_scale)
        layers = [
            nn.Conv2d(10, width, 3, padding=1),
            nn.GELU(),
        ]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(width))
        layers.extend(
            [
                nn.Conv2d(width, width, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(width, 3, 3, padding=1),
                nn.Tanh(),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, input_image, rdnet_image, mask):
        features = torch.cat([input_image, rdnet_image, input_image - rdnet_image, mask], dim=1)
        delta = self.net(features) * self.residual_scale
        refined = (rdnet_image + mask * delta).clamp(0.0, 1.0)
        return {
            "refined": refined,
            "delta": delta,
            "mask": mask,
        }
