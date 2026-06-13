import io
import math
import os
import random
from os.path import join

import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageChops, ImageFilter
import torchvision.transforms.functional as TF

from data.image_folder import make_dataset


def _resize_long_edge(img, max_long_edge, round_factor=1):
    if max_long_edge is None:
        return img
    ow, oh = img.size
    long_edge = max(ow, oh)
    if long_edge <= max_long_edge:
        return img
    scale = max_long_edge / float(long_edge)
    new_w = max(1, int(round(ow * scale)))
    new_h = max(1, int(round(oh * scale)))
    if round_factor > 1:
        new_w = max(round_factor, int(round(new_w / float(round_factor))) * round_factor)
        new_h = max(round_factor, int(round(new_h / float(round_factor))) * round_factor)
    return img.resize((new_w, new_h), Image.BICUBIC)


def _scale_width(img, target_width):
    ow, oh = img.size
    if ow == target_width:
        return img
    h = int(target_width * oh / ow)
    h = int(math.ceil(h / 2.0) * 2)
    return img.resize((target_width, h), Image.BICUBIC)


def _scale_height(img, target_height):
    ow, oh = img.size
    if oh == target_height:
        return img
    w = int(target_height * ow / oh)
    w = int(math.ceil(w / 2.0) * 2)
    return img.resize((w, target_height), Image.BICUBIC)


def paired_data_transforms(img_1, img_2, crop_size=224):
    target_size = int(random.randint(224, 448) / 2.0) * 2
    ow, oh = img_1.size
    if ow >= oh:
        img_1 = _scale_height(img_1, target_size)
        img_2 = _scale_height(img_2, target_size)
    else:
        img_1 = _scale_width(img_1, target_size)
        img_2 = _scale_width(img_2, target_size)

    if random.random() < 0.5:
        img_1 = TF.hflip(img_1)
        img_2 = TF.hflip(img_2)

    w, h = img_1.size
    th = min(crop_size, h)
    tw = min(crop_size, w)
    i = 0 if h == th else random.randint(0, h - th)
    j = 0 if w == tw else random.randint(0, w - tw)
    return TF.crop(img_1, i, j, th, tw), TF.crop(img_2, i, j, th, tw)


def _to_float(img):
    return np.asarray(img, dtype=np.float32) / 255.0


def _to_image(arr):
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


class PILReflectionSynthesis(object):
    def __init__(self, low_sigma=2.0, high_sigma=5.0, low_gamma=1.3, high_gamma=1.3):
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma
        self.low_gamma = low_gamma
        self.high_gamma = high_gamma
        print("[i] PIL reflection synthesis: %s" % {
            "low_sigma": low_sigma,
            "high_sigma": high_sigma,
            "low_gamma": low_gamma,
            "high_gamma": high_gamma,
        })

    def __call__(self, b_img, r_img):
        sigma = random.uniform(self.low_sigma, self.high_sigma)
        gamma = random.uniform(self.low_gamma, self.high_gamma)
        b = _to_float(b_img)
        r = _to_float(r_img.filter(ImageFilter.GaussianBlur(radius=sigma)))
        mix = b + r
        if mix.max() > 1.0:
            overflow = mix[mix > 1.0]
            offset = (overflow.mean() - 1.0) * gamma if overflow.size else 0.0
            r = np.clip(r - offset, 0.0, 1.0)
            mix = np.clip(b + r, 0.0, 1.0)
        return b.astype(np.float32), r.astype(np.float32), mix.astype(np.float32)


class PILRealisticReflectionSynthesis(object):
    def __init__(
            self,
            low_sigma=1.0,
            high_sigma=6.0,
            alpha_min=0.55,
            alpha_max=0.9,
            ghost_prob=0.6,
            max_ghost_shift=8,
            noise_std=0.01,
            jpeg_prob=0.25):
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.ghost_prob = ghost_prob
        self.max_ghost_shift = max_ghost_shift
        self.noise_std = noise_std
        self.jpeg_prob = jpeg_prob
        print("[i] PIL realistic reflection synthesis: %s" % {
            "low_sigma": low_sigma,
            "high_sigma": high_sigma,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "ghost_prob": ghost_prob,
            "max_ghost_shift": max_ghost_shift,
            "noise_std": noise_std,
            "jpeg_prob": jpeg_prob,
        })

    @staticmethod
    def _smooth_mask(h, w, low, high):
        small_h = max(2, int(math.ceil(h / 32.0)))
        small_w = max(2, int(math.ceil(w / 32.0)))
        mask = np.random.uniform(low, high, (small_h, small_w)).astype(np.float32)
        mask_img = Image.fromarray(np.uint8(np.clip(mask * 255.0, 0, 255)), mode="L")
        mask_img = mask_img.resize((w, h), Image.BICUBIC)
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=max(h, w) / 12.0))
        mask = np.asarray(mask_img, dtype=np.float32) / 255.0
        return np.clip(mask[..., None], low, high).astype(np.float32)

    @staticmethod
    def _jpeg_roundtrip(arr):
        img = _to_image(arr)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=random.randint(70, 95))
        buffer.seek(0)
        return _to_float(Image.open(buffer).convert("RGB"))

    def __call__(self, b_img, r_img):
        b = _to_float(b_img)
        sigma = random.uniform(self.low_sigma, self.high_sigma)
        r_blur_img = r_img.filter(ImageFilter.GaussianBlur(radius=sigma))
        if random.random() < self.ghost_prob:
            dx = random.randint(-self.max_ghost_shift, self.max_ghost_shift)
            dy = random.randint(-self.max_ghost_shift, self.max_ghost_shift)
            ghost = ImageChops.offset(r_blur_img, dx, dy)
            r_blur = 0.75 * _to_float(r_blur_img) + 0.25 * _to_float(ghost)
        else:
            r_blur = _to_float(r_blur_img)

        color_gain = np.random.uniform(0.85, 1.15, (1, 1, 3)).astype(np.float32)
        r_blur = np.clip(r_blur * color_gain, 0.0, 1.0)
        h, w = b.shape[:2]
        alpha_t = self._smooth_mask(h, w, self.alpha_min, self.alpha_max)
        alpha_r = self._smooth_mask(h, w, 0.12, 0.45)

        gamma = random.uniform(1.8, 2.4)
        mix = alpha_t * np.power(b, gamma) + alpha_r * np.power(r_blur, gamma)
        mix = np.power(np.clip(mix, 0.0, 1.0), 1.0 / gamma)
        mix = np.clip(mix * random.uniform(0.95, 1.08), 0.0, 1.0)
        if self.noise_std > 0:
            mix = np.clip(mix + np.random.normal(0, self.noise_std, mix.shape).astype(np.float32), 0.0, 1.0)
        if random.random() < self.jpeg_prob:
            mix = self._jpeg_roundtrip(mix)
        reflection = np.clip(alpha_r * r_blur, 0.0, 1.0)
        return b.astype(np.float32), reflection.astype(np.float32), mix.astype(np.float32)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, *args, **kwargs)
        self.shuffle = shuffle

    def reset(self):
        if self.shuffle and hasattr(self.dataset, "reset"):
            print("Reset Dataset...")
            self.dataset.reset()


class PriorCEILDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            datadir,
            fns=None,
            size=None,
            enable_transforms=True,
            synthesis="realistic",
            low_sigma=2.0,
            high_sigma=5.0,
            low_gamma=1.3,
            high_gamma=1.3,
            alpha_min=0.55,
            alpha_max=0.9,
            ghost_prob=0.6,
            max_ghost_shift=8,
            noise_std=0.01,
            jpeg_prob=0.25):
        self.size = size
        self.datadir = datadir
        self.enable_transforms = enable_transforms
        sortkey = lambda key: os.path.split(key)[-1]
        self.paths = sorted(make_dataset(datadir, fns), key=sortkey)
        if size is not None:
            self.paths = self.paths[:size]
        if synthesis == "realistic":
            self.syn_model = PILRealisticReflectionSynthesis(
                low_sigma=low_sigma,
                high_sigma=high_sigma,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                ghost_prob=ghost_prob,
                max_ghost_shift=max_ghost_shift,
                noise_std=noise_std,
                jpeg_prob=jpeg_prob)
        else:
            self.syn_model = PILReflectionSynthesis(
                low_sigma=low_sigma,
                high_sigma=high_sigma,
                low_gamma=low_gamma,
                high_gamma=high_gamma)
        self.reset(shuffle=False)

    def reset(self, shuffle=True):
        if shuffle:
            random.shuffle(self.paths)
        num_paths = len(self.paths) // 2
        self.B_paths = self.paths[:num_paths]
        self.R_paths = self.paths[num_paths:2 * num_paths]

    def __getitem__(self, index):
        b_path = self.B_paths[index % len(self.B_paths)]
        r_path = self.R_paths[index % len(self.R_paths)]
        b_img = Image.open(b_path).convert("RGB")
        r_img = Image.open(r_path).convert("RGB")
        if self.enable_transforms:
            b_img, r_img = paired_data_transforms(b_img, r_img)
        b, r, m = self.syn_model(b_img, r_img)
        return {
            "input": TF.to_tensor(m),
            "target_t": TF.to_tensor(b),
            "target_r": TF.to_tensor(r),
            "fn": os.path.basename(b_path),
            "real": False,
            "unaligned": False,
        }

    def __len__(self):
        size = max(len(self.B_paths), len(self.R_paths))
        return min(size, self.size) if self.size is not None else size


class PriorCEILTestDataset(torch.utils.data.Dataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=False, max_long_edge=None):
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir, "blended"))
        self.enable_transforms = enable_transforms
        self.max_long_edge = max_long_edge
        if size is not None:
            self.fns = self.fns[:size]

    def __getitem__(self, index):
        fn = self.fns[index]
        t_img = Image.open(join(self.datadir, "transmission_layer", fn)).convert("RGB")
        m_img = Image.open(join(self.datadir, "blended", fn)).convert("RGB")
        if self.enable_transforms:
            t_img, m_img = paired_data_transforms(t_img, m_img)
        else:
            t_img = _resize_long_edge(t_img, self.max_long_edge)
            m_img = _resize_long_edge(m_img, self.max_long_edge)
        return {
            "input": TF.to_tensor(m_img),
            "target_t": TF.to_tensor(t_img),
            "target_r": TF.to_tensor(t_img),
            "fn": fn,
            "real": True,
            "unaligned": False,
        }

    def __len__(self):
        return len(self.fns)


class PriorRealDataset(torch.utils.data.Dataset):
    def __init__(self, datadir, fns=None, size=None, max_long_edge=None):
        self.datadir = datadir
        self.fns = fns or os.listdir(datadir)
        self.max_long_edge = max_long_edge
        if size is not None:
            self.fns = self.fns[:size]

    def __getitem__(self, index):
        fn = self.fns[index]
        m_img = Image.open(join(self.datadir, fn)).convert("RGB")
        m_img = _resize_long_edge(m_img, self.max_long_edge)
        return {
            "input": TF.to_tensor(m_img),
            "target_t": -1,
            "fn": fn,
            "real": True,
            "unaligned": False,
        }

    def __len__(self):
        return len(self.fns)


class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, fusion_ratios=None):
        self.datasets = datasets
        self.size = sum(len(dataset) for dataset in datasets)
        self.fusion_ratios = fusion_ratios or [1.0 / len(datasets)] * len(datasets)
        print("[i] using a fusion dataset: %d %s imgs fused with ratio %s" % (
            self.size, [len(dataset) for dataset in datasets], self.fusion_ratios))

    def reset(self):
        for dataset in self.datasets:
            if hasattr(dataset, "reset"):
                dataset.reset()

    def __getitem__(self, index):
        residual = 1.0
        for i, ratio in enumerate(self.fusion_ratios):
            if random.random() < ratio / residual or i == len(self.fusion_ratios) - 1:
                dataset = self.datasets[i]
                return dataset[index % len(dataset)]
            residual -= ratio

    def __len__(self):
        return self.size
