#!/usr/bin/env python3
"""Run RDNet + NAFNet refiner inference on precomputed RDNet outputs."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.mg_rdnet_refiner import RDNetNAFRefiner  # noqa: E402
from ra_rdnet.mask import estimate_reflection_mask  # noqa: E402


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}
GENERIC_IMAGE_STEMS = {
    "blended",
    "clean",
    "image",
    "input",
    "m_input",
    "mg_rdnet_refiner",
    "output",
    "pred",
    "ra_rdnet_mg",
    "ra_rdnet_rca",
    "rdnet",
    "reflection",
    "result",
    "xreflection_rdnet",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Infer with a trained RDNet + NAFNet refiner checkpoint.")
    parser.add_argument("--input_dir", required=True, type=Path, help="Reflection input I directory.")
    parser.add_argument("--rdnet_dir", required=True, type=Path, help="Precomputed RDNet output T_rd directory.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="RDNet + NAFNet checkpoint, e.g. best_psnr.pth.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory for refined outputs.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save_mask", action="store_true", help="Save heuristic soft reflection masks to output_dir/masks.")
    parser.add_argument(
        "--posthoc_mask_gate",
        action="store_true",
        help="Apply post-hoc mask gating at inference: T_out = T_rd + M * delta.",
    )
    parser.add_argument("--rdnet_filename", default="xreflection_rdnet.png", help="Preferred RDNet filename in stem subdirs.")
    parser.add_argument("--output_filename", default="mg_rdnet_refiner.png")
    parser.add_argument("--mask_blur_radius", default=5, type=int)
    parser.add_argument("--mask_sensitivity", default=1.35, type=float)
    parser.add_argument("--mask_gamma", default=0.70, type=float)
    parser.add_argument("--mask_diff_weight", default=0.55, type=float)
    parser.add_argument("--mask_bright_weight", default=0.30, type=float)
    parser.add_argument("--mask_edge_weight", default=0.15, type=float)
    parser.add_argument("--mask_floor", default=0.02, type=float)
    parser.add_argument("--base_channels", default=None, type=int, help="Override model width from checkpoint.")
    parser.add_argument("--residual_scale", default=None, type=float, help="Override residual scale from checkpoint.")
    parser.add_argument(
        "--gate_mode",
        choices=["ungated", "learned_confidence"],
        default=None,
        help="Optional override for checkpoint gate mode.",
    )
    parser.add_argument(
        "--gate_init_bias",
        default=None,
        type=float,
        help="Optional override for checkpoint gate head initialization bias.",
    )
    parser.add_argument("--max_images", default=None, type=int)
    parser.add_argument("--allow_missing", action="store_true")
    return parser.parse_args()


def fail(message):
    raise SystemExit("ERROR: %s" % message)


def resolve_device(device_name):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[w] CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def resolve_input_root(path):
    root = Path(path).expanduser().resolve()
    if (root / "blended").is_dir():
        return root / "blended"
    return root


def list_images(root):
    root = Path(root)
    if not root.exists():
        return []
    images = []
    for path in root.rglob("*"):
        if not is_image(path):
            continue
        try:
            rel_parts = path.relative_to(root).parts[:-1]
        except ValueError:
            rel_parts = ()
        if any(part.startswith("_") for part in rel_parts):
            continue
        images.append(path)
    return sorted(images)


def load_rdnet_manifest_map(root):
    manifest_path = Path(root) / "xreflection_outputs_manifest.csv"
    if not manifest_path.is_file():
        return {}
    with manifest_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if "image" not in reader.fieldnames or "normalized_path" not in reader.fieldnames:
            return {}
        mapping = {}
        for row in reader:
            key = str(row.get("image", "")).strip()
            normalized_path = str(row.get("normalized_path", "")).strip()
            if not key or not normalized_path:
                continue
            path = Path(normalized_path)
            if not path.is_absolute():
                path = (manifest_path.parent / path).resolve()
            if path.is_file():
                mapping[key] = path
        return mapping


def image_key(path, preferred_filename=None):
    if preferred_filename and path.name == preferred_filename:
        return path.parent.name
    if path.stem.lower() in GENERIC_IMAGE_STEMS:
        return path.parent.name
    return path.stem


def image_score(path, key, role, preferred_filename=None):
    score = 0
    lower_name = path.name.lower()
    lower_stem = path.stem.lower()
    if preferred_filename and path.name == preferred_filename:
        score += 100
    if path.parent.name == key:
        score += 25
    if path.stem == key:
        score += 20
    if role == "rdnet" and ("rdnet" in lower_name or "xreflection" in lower_name):
        score += 30
    if "mask" in lower_stem or "delta" in lower_stem:
        score -= 100
    return score


def build_image_map(root, role, preferred_filename=None):
    root = Path(root)
    if role == "rdnet":
        manifest_map = load_rdnet_manifest_map(root)
        if manifest_map:
            return manifest_map

    mapping = {}
    scores = {}
    duplicate_keys = set()
    for path in list_images(root):
        key = image_key(path, preferred_filename=preferred_filename if role == "rdnet" else None)
        score = image_score(path, key, role, preferred_filename=preferred_filename)
        if key in mapping:
            duplicate_keys.add(key)
            if score <= scores[key]:
                continue
        mapping[key] = path
        scores[key] = score
    if duplicate_keys:
        print("[w] duplicate %s stems resolved under %s: %s" % (role, root, sorted(duplicate_keys)[:8]))
    return mapping


def collect_pairs(input_dir, rdnet_dir, rdnet_filename="xreflection_rdnet.png", allow_missing=False):
    input_root = resolve_input_root(input_dir)
    rdnet_root = Path(rdnet_dir).expanduser().resolve()
    if not input_root.exists():
        fail("input_dir does not exist: %s" % input_root)
    if not rdnet_root.exists():
        fail("rdnet_dir does not exist: %s" % rdnet_root)
    input_map = build_image_map(input_root, "input")
    rdnet_map = build_image_map(rdnet_root, "rdnet", preferred_filename=rdnet_filename)
    if not input_map:
        fail("no input images found under: %s" % input_root)
    if not rdnet_map:
        fail("no RDNet images found under: %s" % rdnet_root)

    pairs = []
    missing = []
    for stem in sorted(input_map):
        rdnet_path = rdnet_map.get(stem)
        if rdnet_path is None:
            missing.append(stem)
            continue
        pairs.append({"stem": stem, "input": input_map[stem], "rdnet": rdnet_path})
    if missing and not allow_missing:
        fail("missing RDNet outputs for %d input stem(s); first missing: %s" % (len(missing), ", ".join(missing[:10])))
    if not pairs:
        fail("no matched input/RDNet pairs found.")
    return pairs


def load_rgb_tensor(path, size=None):
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


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


def make_reflection_mask(input_image, rdnet_image, args):
    return estimate_reflection_mask(
        input_image,
        rdnet_image,
        sensitivity=args.mask_sensitivity,
        gamma=args.mask_gamma,
        blur_radius=args.mask_blur_radius,
        diff_weight=args.mask_diff_weight,
        bright_weight=args.mask_bright_weight,
        edge_weight=args.mask_edge_weight,
        floor=args.mask_floor,
    ).clamp(0.0, 1.0)


def torch_load_checkpoint(path):
    if not path.is_file():
        fail("checkpoint does not exist: %s" % path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        return checkpoint.get("state_dict", checkpoint.get("model_state", checkpoint))
    return checkpoint


def checkpoint_model_config(checkpoint):
    if not isinstance(checkpoint, dict):
        return {}
    config = checkpoint.get("model_config", {})
    return dict(config) if isinstance(config, dict) else {}


def load_model(args, device):
    checkpoint = torch_load_checkpoint(args.checkpoint.expanduser().resolve())
    config = checkpoint_model_config(checkpoint)
    base_channels = args.base_channels or int(config.get("base_channels", config.get("width", 48)))
    residual_scale = args.residual_scale or float(config.get("residual_scale", 0.35))
    gate_mode = args.gate_mode or str(config.get("gate_mode", "ungated"))
    gate_init_bias = args.gate_init_bias
    if gate_init_bias is None:
        gate_init_bias = float(config.get("gate_init_bias", 4.0))
    model = RDNetNAFRefiner(
        base_channels=base_channels,
        residual_scale=residual_scale,
        gate_mode=gate_mode,
        gate_init_bias=gate_init_bias,
    )
    state_dict = checkpoint_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["stem", "input_path", "rdnet_path", "output_path", "mask_path"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model = load_model(args, device)
    learned_confidence_mode = getattr(model, "gate_mode", "ungated") == "learned_confidence"
    if args.posthoc_mask_gate and learned_confidence_mode:
        print("[w] ignoring --posthoc_mask_gate because checkpoint already uses learned confidence gating.")
    pairs = collect_pairs(
        args.input_dir,
        args.rdnet_dir,
        rdnet_filename=args.rdnet_filename,
        allow_missing=args.allow_missing,
    )
    if args.max_images is not None:
        pairs = pairs[: int(args.max_images)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.no_grad():
        for index, pair in enumerate(pairs, start=1):
            input_tensor = load_rgb_tensor(pair["input"]).to(device)
            size = (input_tensor.shape[2], input_tensor.shape[1])
            rdnet_tensor = load_rgb_tensor(pair["rdnet"], size=size).to(device)
            input_batch = input_tensor.unsqueeze(0)
            rdnet_batch = rdnet_tensor.unsqueeze(0)
            mask = make_reflection_mask(input_batch, rdnet_batch, args)
            t_out, delta, mask = model(input_batch, rdnet_batch, mask)
            if args.posthoc_mask_gate and not learned_confidence_mode:
                t_out = (rdnet_batch + mask * delta).clamp(0.0, 1.0)

            out_path = args.output_dir / pair["stem"] / args.output_filename
            mask_path = args.output_dir / "masks" / (pair["stem"] + ".png")
            save_rgb_tensor(t_out[0], out_path)
            saved_mask = ""
            if args.save_mask:
                save_mask_tensor(mask[0], mask_path)
                saved_mask = str(mask_path)
            rows.append(
                {
                    "stem": pair["stem"],
                    "input_path": str(pair["input"]),
                    "rdnet_path": str(pair["rdnet"]),
                    "output_path": str(out_path),
                    "mask_path": saved_mask,
                }
            )
            print("[i] %d/%d %s -> %s" % (index, len(pairs), pair["stem"], out_path))

    write_manifest(args.output_dir / "mg_rdnet_refiner_manifest.csv", rows)
    print("[i] wrote RDNet + NAFNet refiner outputs under: %s" % args.output_dir)


if __name__ == "__main__":
    main()
