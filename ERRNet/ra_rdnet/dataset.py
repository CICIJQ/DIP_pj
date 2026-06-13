import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .common import collect_pairs, load_rgb_tensor


class RARefinementDataset(Dataset):
    def __init__(
        self,
        input_dir,
        rdnet_dir,
        gt_dir=None,
        rdnet_filename="xreflection_rdnet.png",
        stems=None,
        patch_size=None,
        augment=False,
        max_images=None,
    ):
        pairs = collect_pairs(
            input_dir,
            rdnet_dir,
            gt_dir=gt_dir,
            rdnet_filename=rdnet_filename,
            allow_missing=False,
        )
        if stems is not None:
            keep = set(stems)
            pairs = [pair for pair in pairs if pair["stem"] in keep]
        if max_images is not None:
            pairs = pairs[: int(max_images)]
        if not pairs:
            raise ValueError("no RA-RDNet training pairs after filtering.")
        self.pairs = pairs
        self.patch_size = patch_size
        self.augment = augment
        self.has_gt = gt_dir is not None

    def __len__(self):
        return len(self.pairs)

    def _resize_to_min_patch(self, tensors):
        if not self.patch_size:
            return tensors
        _, height, width = tensors[0].shape
        if height >= self.patch_size and width >= self.patch_size:
            return tensors
        new_height = max(height, self.patch_size)
        new_width = max(width, self.patch_size)
        return [
            F.interpolate(
                tensor.unsqueeze(0),
                size=(new_height, new_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            for tensor in tensors
        ]

    def _random_crop(self, tensors):
        if not self.patch_size:
            return tensors
        _, height, width = tensors[0].shape
        if height == self.patch_size and width == self.patch_size:
            return tensors
        top = random.randint(0, height - self.patch_size)
        left = random.randint(0, width - self.patch_size)
        return [
            tensor[:, top : top + self.patch_size, left : left + self.patch_size]
            for tensor in tensors
        ]

    def _augment(self, tensors):
        if not self.augment:
            return tensors
        if random.random() < 0.5:
            tensors = [torch.flip(tensor, dims=[2]) for tensor in tensors]
        if random.random() < 0.5:
            tensors = [torch.flip(tensor, dims=[1]) for tensor in tensors]
        return tensors

    def __getitem__(self, index):
        pair = self.pairs[index]
        gt = load_rgb_tensor(pair["gt"]) if pair["gt"] else None
        target_size = None
        if gt is not None:
            target_size = (gt.shape[2], gt.shape[1])
        input_image = load_rgb_tensor(pair["input"], size=target_size)
        rdnet_image = load_rgb_tensor(pair["rdnet"], size=target_size)
        tensors = [input_image, rdnet_image]
        if gt is not None:
            tensors.append(gt)
        tensors = self._resize_to_min_patch(tensors)
        tensors = self._random_crop(tensors)
        tensors = self._augment(tensors)
        item = {
            "stem": pair["stem"],
            "input": tensors[0],
            "rdnet": tensors[1],
        }
        if gt is not None:
            item["gt"] = tensors[2]
        return item
