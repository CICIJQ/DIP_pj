import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def fail(message):
    raise SystemExit("ERROR: %s" % message)


def is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def resolve_image_root(path, kind):
    path = Path(path).expanduser().resolve()
    if kind == "input" and (path / "blended").is_dir():
        return path / "blended"
    if kind == "gt" and (path / "transmission_layer").is_dir():
        return path / "transmission_layer"
    return path


def list_images(root):
    root = Path(root)
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if is_image(path))


def build_stem_map(root):
    mapping = {}
    duplicates = []
    for path in list_images(root):
        if path.stem in mapping:
            duplicates.append(path.stem)
            continue
        mapping[path.stem] = path
    if duplicates:
        print("[w] duplicate stems ignored under %s: %s" % (root, sorted(set(duplicates))[:8]))
    return mapping


def find_method_image(root, stem, filename):
    root = Path(root)
    if filename:
        nested = root / stem / filename
        if nested.is_file():
            return nested
        flat = root / (stem + Path(filename).suffix)
        if flat.is_file():
            return flat

    preferred = (
        "ra_rdnet_mg.png",
        "ra_rdnet_rca.png",
        "xreflection_rdnet.png",
        "rdnet.png",
        "clean.png",
        "output.png",
        "pred.png",
        "result.png",
    )
    candidates = []
    for path in list_images(root):
        if path.parent.name == stem or path.stem == stem:
            candidates.append(path)
    if not candidates:
        return None

    def score(path):
        value = 0
        lower = path.name.lower()
        if path.parent.name == stem:
            value += 50
        if path.stem == stem:
            value += 25
        if lower in preferred:
            value += 100 - preferred.index(lower)
        if "mask" in path.stem.lower() or "delta" in path.stem.lower():
            value -= 100
        return value

    return sorted(candidates, key=score, reverse=True)[0]


def collect_pairs(input_dir, rdnet_dir, gt_dir=None, rdnet_filename="xreflection_rdnet.png", allow_missing=False):
    input_root = resolve_image_root(input_dir, "input")
    rdnet_root = Path(rdnet_dir).expanduser().resolve()
    gt_root = resolve_image_root(gt_dir, "gt") if gt_dir else None

    if not input_root.exists():
        fail("input image root does not exist: %s" % input_root)
    if not rdnet_root.exists():
        fail("RDNet result directory does not exist: %s" % rdnet_root)
    if gt_root is not None and not gt_root.exists():
        fail("GT image root does not exist: %s" % gt_root)

    input_map = build_stem_map(input_root)
    gt_map = build_stem_map(gt_root) if gt_root else {}
    if not input_map:
        fail("no input images found under: %s" % input_root)

    stems = sorted(gt_map) if gt_map else sorted(input_map)
    pairs = []
    missing = []
    for stem in stems:
        input_path = input_map.get(stem)
        rdnet_path = find_method_image(rdnet_root, stem, rdnet_filename)
        gt_path = gt_map.get(stem) if gt_map else None
        if input_path is None or rdnet_path is None or (gt_root is not None and gt_path is None):
            missing.append(stem)
            continue
        pairs.append(
            {
                "stem": stem,
                "input": input_path,
                "rdnet": rdnet_path,
                "gt": gt_path,
            }
        )
    if missing and not allow_missing:
        fail(
            "missing input/RDNet/GT images for %d stem(s), first missing: %s"
            % (len(missing), ", ".join(missing[:10]))
        )
    if not pairs:
        fail("no matched image pairs found.")
    return pairs


def load_rgb_tensor(path, size=None):
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def resize_tensor(tensor, size_hw):
    if tensor.shape[-2:] == tuple(size_hw):
        return tensor
    return F.interpolate(
        tensor.unsqueeze(0),
        size=tuple(size_hw),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def save_rgb_tensor(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(array).save(path)


def save_mask_tensor(mask, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = mask.detach().float().cpu().clamp(0.0, 1.0)
    if mask.ndim == 3:
        mask = mask[0]
    array = (mask.numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def save_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["stem", "input_path", "rdnet_path", "output_path", "mask_path"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
