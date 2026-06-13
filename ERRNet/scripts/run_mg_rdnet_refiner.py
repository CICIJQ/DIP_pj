#!/usr/bin/env python3
"""Run a trained RA-RDNet MG-RDNet Refiner checkpoint."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_rdnet.common import (  # noqa: E402
    collect_pairs,
    load_rgb_tensor,
    save_manifest,
    save_mask_tensor,
    save_rgb_tensor,
)
from ra_rdnet.mask import estimate_reflection_mask  # noqa: E402
from ra_rdnet.model import MaskGuidedResidualRefiner  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run RA-RDNet MG-RDNet Refiner inference.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--rdnet_dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--rdnet_filename", default="xreflection_rdnet.png")
    parser.add_argument("--output_filename", default="ra_rdnet_mg.png")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=None, help="Override checkpoint width.")
    parser.add_argument("--num_blocks", type=int, default=None, help="Override checkpoint block count.")
    parser.add_argument("--residual_scale", type=float, default=None, help="Override checkpoint residual scale.")
    parser.add_argument("--mask_sensitivity", type=float, default=1.25)
    parser.add_argument("--mask_gamma", type=float, default=0.70)
    parser.add_argument("--mask_blur_radius", type=int, default=5)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--allow_missing", action="store_true")
    parser.add_argument("--no_save_mask", action="store_true")
    parser.add_argument("--save_delta", action="store_true", help="Save visualized residual delta as delta_vis.png.")
    return parser.parse_args()


def resolve_device(device_name):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[w] CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def load_model(args, device):
    if not args.checkpoint.is_file():
        raise SystemExit("ERROR: checkpoint does not exist: %s" % args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config", {})
    width = args.width or int(config.get("width", 32))
    num_blocks = args.num_blocks or int(config.get("num_blocks", 6))
    residual_scale = args.residual_scale or float(config.get("residual_scale", 0.35))
    model = MaskGuidedResidualRefiner(
        width=width,
        num_blocks=num_blocks,
        residual_scale=residual_scale,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model = load_model(args, device)
    pairs = collect_pairs(
        args.input_dir,
        args.rdnet_dir,
        rdnet_filename=args.rdnet_filename,
        allow_missing=args.allow_missing,
    )
    if args.max_images is not None:
        pairs = pairs[: args.max_images]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for index, pair in enumerate(pairs):
            input_tensor = load_rgb_tensor(pair["input"]).to(device)
            size = (input_tensor.shape[2], input_tensor.shape[1])
            rdnet_tensor = load_rgb_tensor(pair["rdnet"], size=size).to(device)
            input_batch = input_tensor.unsqueeze(0)
            rdnet_batch = rdnet_tensor.unsqueeze(0)
            mask = estimate_reflection_mask(
                input_batch,
                rdnet_batch,
                sensitivity=args.mask_sensitivity,
                gamma=args.mask_gamma,
                blur_radius=args.mask_blur_radius,
            )
            output = model(input_batch, rdnet_batch, mask)
            out_dir = args.output_dir / pair["stem"]
            out_path = out_dir / args.output_filename
            mask_path = out_dir / "reflection_confidence.png"
            save_rgb_tensor(output["refined"][0], out_path)
            if not args.no_save_mask:
                save_mask_tensor(mask[0], mask_path)
            if args.save_delta:
                delta_vis = (output["delta"][0] / (2.0 * model.residual_scale) + 0.5).clamp(0.0, 1.0)
                save_rgb_tensor(delta_vis, out_dir / "delta_vis.png")
            rows.append(
                {
                    "stem": pair["stem"],
                    "input_path": str(pair["input"]),
                    "rdnet_path": str(pair["rdnet"]),
                    "output_path": str(out_path),
                    "mask_path": "" if args.no_save_mask else str(mask_path),
                }
            )
            print("[i] %d/%d %s -> %s" % (index + 1, len(pairs), pair["stem"], out_path))
    save_manifest(args.output_dir / "ra_rdnet_mg_manifest.csv", rows)
    print("[i] wrote MG-RDNet Refiner outputs under: %s" % args.output_dir)


if __name__ == "__main__":
    main()
