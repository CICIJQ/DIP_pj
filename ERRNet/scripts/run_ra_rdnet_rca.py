#!/usr/bin/env python3
"""Run training-free Reflection Correction Amplification on RDNet outputs."""

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
from ra_rdnet.rca import apply_reflection_correction_amplification  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply RA-RDNet version A: training-free Reflection Correction "
            "Amplification (RCA)."
        )
    )
    parser.add_argument("--input_dir", required=True, type=Path, help="Input images or dataset root with blended/.")
    parser.add_argument("--rdnet_dir", required=True, type=Path, help="Existing XReflection-RDNet output directory.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory for RCA outputs.")
    parser.add_argument("--rdnet_filename", default="xreflection_rdnet.png", help="RDNet filename under rdnet_dir/<stem>/.")
    parser.add_argument("--output_filename", default="ra_rdnet_rca.png", help="Output filename under output_dir/<stem>/.")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N.")
    parser.add_argument("--strength", type=float, default=0.65, help="Amplification strength along RDNet correction direction.")
    parser.add_argument("--max_extra_delta", type=float, default=0.35, help="Clamp for extra RGB correction in [0,1] units.")
    parser.add_argument("--mask_sensitivity", type=float, default=1.25)
    parser.add_argument("--mask_gamma", type=float, default=0.70, help="Lower than 1 expands high-mask regions.")
    parser.add_argument("--mask_blur_radius", type=int, default=5)
    parser.add_argument("--mask_floor", type=float, default=0.0, help="Minimum mask value; >0 makes RCA more global.")
    parser.add_argument("--max_images", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--allow_missing", action="store_true", help="Skip missing RDNet outputs instead of failing.")
    parser.add_argument("--no_save_mask", action="store_true", help="Do not save reflection confidence masks.")
    return parser.parse_args()


def resolve_device(device_name):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[w] CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    pairs = collect_pairs(
        args.input_dir,
        args.rdnet_dir,
        rdnet_filename=args.rdnet_filename,
        allow_missing=args.allow_missing,
    )
    if args.max_images is not None:
        pairs = pairs[: args.max_images]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
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
                floor=args.mask_floor,
            )
            output = apply_reflection_correction_amplification(
                input_batch,
                rdnet_batch,
                mask,
                strength=args.strength,
                max_extra_delta=args.max_extra_delta,
            )

            out_dir = args.output_dir / pair["stem"]
            out_path = out_dir / args.output_filename
            mask_path = out_dir / "reflection_confidence.png"
            save_rgb_tensor(output[0], out_path)
            if not args.no_save_mask:
                save_mask_tensor(mask[0], mask_path)
            manifest_rows.append(
                {
                    "stem": pair["stem"],
                    "input_path": str(pair["input"]),
                    "rdnet_path": str(pair["rdnet"]),
                    "output_path": str(out_path),
                    "mask_path": "" if args.no_save_mask else str(mask_path),
                }
            )
            print("[i] %d/%d %s -> %s" % (index + 1, len(pairs), pair["stem"], out_path))

    save_manifest(args.output_dir / "ra_rdnet_rca_manifest.csv", manifest_rows)
    print("[i] wrote RCA outputs under: %s" % args.output_dir)


if __name__ == "__main__":
    main()
