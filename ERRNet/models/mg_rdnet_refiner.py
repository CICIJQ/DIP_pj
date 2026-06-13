"""RDNet + NAFNet residual refiner.

This module supports two RDNet refinement modes:

1. ``ungated``: predict an additive residual over the coarse RDNet output.
2. ``learned_confidence``: predict the same residual, but modulate it with a
   learned confidence gate on top of the heuristic reflection mask.

The backbone still consumes the same 9-channel input
``[I, T_rd, I - T_rd]`` so we can warm-start the learned-gating model from the
existing ungated checkpoint without changing the intro convolution.
"""

import torch
import torch.nn.functional as F
from torch import nn

from models.arch.nafnet_refiner import NAFNetBackbone


def _ensure_bchw(value, name):
    if value.ndim == 3:
        return value.unsqueeze(0)
    if value.ndim != 4:
        raise ValueError("%s must be CHW or BCHW, got shape %r" % (name, tuple(value.shape)))
    return value


def parse_int_list(value):
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def build_rdnet_naf_features(input_image, rdnet_image):
    """Build the 9-channel RDNet+NAF input tensor.

    Channels are [I(3), T_rd(3), I - T_rd(3)].
    """

    input_image = _ensure_bchw(input_image, "input_image")
    rdnet_image = _ensure_bchw(rdnet_image, "rdnet_image")
    if input_image.shape != rdnet_image.shape:
        raise ValueError(
            "input_image and rdnet_image must have the same shape, got %r and %r"
            % (tuple(input_image.shape), tuple(rdnet_image.shape))
        )
    if input_image.shape[1] != 3:
        raise ValueError("input_image must have 3 channels, got %d" % input_image.shape[1])

    residual = input_image - rdnet_image
    return torch.cat([input_image, rdnet_image, residual], dim=1)


def build_mg_rdnet_features(input_image, rdnet_image, mask=None):
    """Backward-compatible alias for older callers."""

    return build_rdnet_naf_features(input_image, rdnet_image)


class MaskGuidedRDNetResidualRefiner(nn.Module):
    """RDNet + NAFNet residual refiner with optional learned confidence gating."""

    def __init__(
        self,
        width=32,
        middle_blk_num=4,
        enc_blk_nums=(1, 1, 2),
        dec_blk_nums=(1, 1, 1),
        residual_scale=0.35,
        base_channels=None,
        gate_mode="ungated",
        gate_init_bias=4.0,
    ):
        super().__init__()
        if base_channels is not None:
            width = base_channels

        width = int(width)
        if width <= 0:
            raise ValueError("width must be positive.")
        self.width = width
        self.middle_blk_num = int(middle_blk_num)
        self.enc_blk_nums = parse_int_list(enc_blk_nums)
        self.dec_blk_nums = parse_int_list(dec_blk_nums)
        self.residual_scale = float(residual_scale)
        self.gate_mode = str(gate_mode)
        self.gate_init_bias = float(gate_init_bias)
        if self.gate_mode not in {"ungated", "learned_confidence"}:
            raise ValueError("unsupported gate_mode: %s" % self.gate_mode)

        self.backbone = NAFNetBackbone(
            img_channels=9,
            out_channels=3,
            width=self.width,
            middle_blk_num=self.middle_blk_num,
            enc_blk_nums=self.enc_blk_nums,
            dec_blk_nums=self.dec_blk_nums,
        )
        nn.init.zeros_(self.backbone.ending.weight)
        nn.init.zeros_(self.backbone.ending.bias)
        self.gate_head = None
        if self.gate_mode == "learned_confidence":
            self.gate_head = nn.Conv2d(self.width, 1, 3, padding=1)
            nn.init.zeros_(self.gate_head.weight)
            nn.init.constant_(self.gate_head.bias, self.gate_init_bias)

    def _prepare_mask(self, mask, input_image, rdnet_image):
        if mask is None:
            return torch.ones(
                input_image.shape[0],
                1,
                input_image.shape[2],
                input_image.shape[3],
                dtype=input_image.dtype,
                device=input_image.device,
            )
        mask = _ensure_bchw(mask, "mask")
        if mask.shape[1] != 1:
            raise ValueError("mask must have 1 channel, got %d" % mask.shape[1])
        if mask.shape[0] != input_image.shape[0] or mask.shape[2:] != input_image.shape[2:]:
            raise ValueError(
                "mask shape %r must match input batch/spatial shape %r"
                % (tuple(mask.shape), tuple((input_image.shape[0], 1) + input_image.shape[2:]))
            )
        return mask.to(dtype=input_image.dtype, device=input_image.device).clamp(0.0, 1.0)

    def forward(self, input_image, rdnet_image, mask=None):
        input_image = _ensure_bchw(input_image, "input_image")
        rdnet_image = _ensure_bchw(rdnet_image, "rdnet_image")
        mask = self._prepare_mask(mask, input_image, rdnet_image)
        features = build_rdnet_naf_features(input_image, rdnet_image)
        shared = self.backbone.forward_features(features)
        delta = torch.tanh(self.backbone.ending(shared)) * self.residual_scale
        if self.gate_mode == "learned_confidence":
            confidence = torch.sigmoid(self.gate_head(shared))
            gate = mask * confidence
            t_out = (rdnet_image + gate * delta).clamp(0.0, 1.0)
            return t_out, delta, gate
        t_out = (rdnet_image + delta).clamp(0.0, 1.0)
        return t_out, delta, mask


MG_RDNetRefiner = MaskGuidedRDNetResidualRefiner
RDNetNAFRefiner = MaskGuidedRDNetResidualRefiner
