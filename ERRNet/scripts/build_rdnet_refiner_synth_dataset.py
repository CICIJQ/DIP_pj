#!/usr/bin/env python3
"""Export a fixed synthetic paired dataset for RDNet refiner training.

This converts the original ERRNet-style online VOC reflection synthesis into a
stable `blended/` + `transmission_layer/` directory so we can precompute RDNet
coarse outputs once and train the refiner without mixing benchmark test sets
into the training split.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.image_folder import read_fns  # noqa: E402
from data.reflect_dataset import CEILDataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a fixed synthetic paired dataset for RDNet refiner training."
    )
    parser.add_argument(
        "--voc_dir",
        default=Path("./datasets/processed_data/VOCdevkit/VOC2012/PNGImages"),
        type=Path,
        help="VOC PNG root used by the original ERRNet training pipeline.",
    )
    parser.add_argument(
        "--voc_list",
        default=Path("./VOC2012_224_train_png.txt"),
        type=Path,
        help="Image id list used by ERRNet synthetic training.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Output dataset root; images are written to blended/ and transmission_layer/.",
    )
    parser.add_argument(
        "--count",
        default=None,
        type=int,
        help="Number of synthetic pairs to export. Default uses the full CEILDataset length.",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--low_sigma", default=2.0, type=float)
    parser.add_argument("--high_sigma", default=5.0, type=float)
    parser.add_argument("--low_gamma", default=1.3, type=float)
    parser.add_argument("--high_gamma", default=1.3, type=float)
    return parser.parse_args()


def fail(message):
    raise SystemExit("ERROR: %s" % message)


def save_rgb_tensor(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(array).save(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    args.voc_dir = args.voc_dir.expanduser().resolve()
    args.voc_list = args.voc_list.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if not args.voc_dir.is_dir():
        fail("voc_dir does not exist: %s" % args.voc_dir)
    if not args.voc_list.is_file():
        fail("voc_list does not exist: %s" % args.voc_list)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        fail("output_dir already exists and is not empty: %s" % args.output_dir)

    set_seed(args.seed)
    dataset = CEILDataset(
        str(args.voc_dir),
        read_fns(str(args.voc_list)),
        enable_transforms=True,
        low_sigma=args.low_sigma,
        high_sigma=args.high_sigma,
        low_gamma=args.low_gamma,
        high_gamma=args.high_gamma,
    )
    max_count = len(dataset)
    count = max_count if args.count is None else int(args.count)
    if count <= 0:
        fail("--count must be positive.")
    if count > max_count:
        fail("requested count %d exceeds maximum synthetic pairs %d." % (count, max_count))

    blended_dir = args.output_dir / "blended"
    gt_dir = args.output_dir / "transmission_layer"
    blended_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index in range(count):
        sample = dataset[index]
        stem = "voc_syn_%05d" % index
        input_path = blended_dir / ("%s.png" % stem)
        gt_path = gt_dir / ("%s.png" % stem)
        save_rgb_tensor(sample["input"], input_path)
        save_rgb_tensor(sample["target_t"], gt_path)

        rows.append(
            {
                "index": index,
                "stem": stem,
                "transmission_source": Path(dataset.B_paths[index % len(dataset.B_paths)]).name,
                "reflection_source": Path(dataset.R_paths[index % len(dataset.R_paths)]).name,
                "input_path": str(input_path.relative_to(args.output_dir)),
                "gt_path": str(gt_path.relative_to(args.output_dir)),
            }
        )
        if index == 0 or (index + 1) % 500 == 0 or index + 1 == count:
            print("[i] exported %d/%d synthetic pairs" % (index + 1, count))

    with (args.output_dir / "synthetic_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "stem",
                "transmission_source",
                "reflection_source",
                "input_path",
                "gt_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with (args.output_dir / "synthetic_meta.json").open("w") as handle:
        json.dump(
            {
                "count": count,
                "seed": args.seed,
                "voc_dir": str(args.voc_dir),
                "voc_list": str(args.voc_list),
                "low_sigma": args.low_sigma,
                "high_sigma": args.high_sigma,
                "low_gamma": args.low_gamma,
                "high_gamma": args.high_gamma,
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    print("[i] wrote synthetic paired dataset: %s" % args.output_dir)


if __name__ == "__main__":
    main()
