import math

import torch


def psnr(pred, target, eps=1e-10):
    mse = torch.mean((pred.clamp(0, 1) - target.clamp(0, 1)) ** 2).item()
    if mse <= eps:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)
