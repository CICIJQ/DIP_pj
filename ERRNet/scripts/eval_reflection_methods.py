#!/usr/bin/env python3
"""Evaluate reflection-removal predictions against ground truth images."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
METRICS = ("PSNR", "SSIM", "NCC", "LMSE")
NON_PRED_TOKENS = (
    "comparison",
    "coarse",
    "gt",
    "input",
    "label",
    "m_input",
    "t_label",
)
PREFERRED_PRED_NAMES = (
    "xreflection_rdnet.png",
    "naf_refiner.png",
    "baseline_eval.png",
    "output.png",
    "pred.png",
    "result.png",
    "clean.png",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute PSNR, SSIM, NCC, and LMSE for predicted images."
    )
    parser.add_argument("--pred_dir", required=True, type=Path, help="Prediction directory.")
    parser.add_argument(
        "--gt_dir",
        required=True,
        type=Path,
        help="GT image directory, or dataset root containing transmission_layer/.",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        type=Path,
        help="Per-image CSV path. Defaults to pred_dir/per_image_metrics.csv.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        type=Path,
        help="Average metric JSON path. Defaults to pred_dir/average_metrics.json.",
    )
    parser.add_argument(
        "--summary_txt",
        default=None,
        type=Path,
        help="Optional summary.txt path compatible with existing result folders.",
    )
    parser.add_argument(
        "--pred_filename",
        default=None,
        help=(
            "Prediction filename inside per-image subdirectories, e.g. "
            "xreflection_rdnet.png, baseline_eval.png, or naf_refiner.png. "
            "If omitted, the script chooses the best candidate automatically."
        ),
    )
    parser.add_argument(
        "--no_resize_pred",
        action="store_true",
        help="Fail on size mismatch instead of resizing prediction to GT size.",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Warn and continue if a GT image has no matching prediction.",
    )
    return parser.parse_args()


def fail(message):
    raise SystemExit("ERROR: %s" % message)


def is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def image_root(path, gt=False):
    path = path.expanduser().resolve()
    if gt and (path / "transmission_layer").is_dir():
        return path / "transmission_layer"
    if (not gt) and (path / "blended").is_dir():
        return path / "blended"
    return path


def list_images_recursive(root):
    return sorted(path for path in root.rglob("*") if is_image(path))


def build_stem_map(root):
    stem_map = {}
    duplicates = []
    for path in list_images_recursive(root):
        if path.stem in stem_map:
            duplicates.append(path.stem)
            continue
        stem_map[path.stem] = path
    if duplicates:
        print("[w] duplicate stems ignored under %s: %s" % (root, sorted(set(duplicates))[:10]))
    return stem_map


def candidate_score(path, stem):
    lower_name = path.name.lower()
    lower_stem = path.stem.lower()
    score = 0
    if path.parent.name == stem:
        score += 50
    if path.stem == stem:
        score += 30
    if lower_name in PREFERRED_PRED_NAMES:
        score += 100 - PREFERRED_PRED_NAMES.index(lower_name)
    if "clean" in lower_stem:
        score += 25
    if "rdnet" in lower_stem or "xreflection" in lower_stem:
        score += 20
    if any(token == lower_stem or token in lower_stem for token in NON_PRED_TOKENS):
        score -= 100
    if "reflection" in lower_stem and "xreflection" not in lower_stem:
        score -= 100
    return score


def build_prediction_map(pred_root, stems, pred_filename=None):
    pred_map = {}
    if pred_filename:
        for stem in stems:
            nested = pred_root / stem / pred_filename
            flat = pred_root / (stem + Path(pred_filename).suffix)
            if nested.is_file():
                pred_map[stem] = nested
            elif flat.is_file():
                pred_map[stem] = flat
        return pred_map

    candidates = {stem: [] for stem in stems}
    for path in list_images_recursive(pred_root):
        if path.parent.name in candidates:
            candidates[path.parent.name].append(path)
        if path.stem in candidates:
            candidates[path.stem].append(path)

    for stem, paths in candidates.items():
        if not paths:
            continue
        pred_map[stem] = sorted(
            paths,
            key=lambda item: (candidate_score(item, stem), item.stat().st_mtime),
            reverse=True,
        )[0]
    return pred_map


def read_rgb(path):
    return Image.open(path).convert("RGB")


def to_float_array(image):
    return np.asarray(image, dtype=np.float64)


def compare_ssim_rgb(gt, pred):
    try:
        return structural_similarity(gt, pred, data_range=255, channel_axis=-1)
    except TypeError:
        return structural_similarity(gt, pred, data_range=255, multichannel=True)


def compare_ncc(gt, pred):
    gt_centered = gt - np.mean(gt)
    pred_centered = pred - np.mean(pred)
    denom = np.std(gt) * np.std(pred)
    if denom < 1e-12:
        return 0.0
    return float(np.mean(gt_centered * pred_centered) / denom)


def ssq_error(correct, estimate):
    if np.sum(estimate ** 2) > 1e-5:
        alpha = np.sum(correct * estimate) / np.sum(estimate ** 2)
    else:
        alpha = 0.0
    return np.sum((correct - alpha * estimate) ** 2)


def local_error(correct, estimate, window_size=20, window_shift=10):
    height, width, channels = correct.shape
    window_size = min(window_size, height, width)
    if window_size <= 0:
        return math.nan
    window_shift = max(1, min(window_shift, window_size))

    ssq = 0.0
    total = 0.0
    for channel in range(channels):
        for row in range(0, height - window_size + 1, window_shift):
            for col in range(0, width - window_size + 1, window_shift):
                correct_curr = correct[row : row + window_size, col : col + window_size, channel]
                estimate_curr = estimate[row : row + window_size, col : col + window_size, channel]
                ssq += ssq_error(correct_curr, estimate_curr)
                total += np.sum(correct_curr ** 2)
    if total < 1e-12:
        return math.nan
    return float(ssq / total)


def quality_assess(pred, gt):
    return {
        "PSNR": float(peak_signal_noise_ratio(gt, pred, data_range=255)),
        "SSIM": float(compare_ssim_rgb(gt, pred)),
        "NCC": compare_ncc(gt, pred),
        "LMSE": local_error(gt, pred),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image", "pred_path", "gt_path", "width", "height"] + list(METRICS)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    averages = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if not math.isnan(float(row[metric]))]
        averages[metric] = float(np.mean(values)) if values else math.nan
    payload = {
        "num_images": len(rows),
        "metrics": averages,
    }
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def write_summary(path, rows, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("Images: %d\n\n" % len(rows))
        handle.write("Averages\n")
        for metric in METRICS:
            handle.write("%s %.6f\n" % (metric, payload["metrics"][metric]))


def main():
    args = parse_args()
    pred_root = args.pred_dir.expanduser().resolve()
    gt_root = image_root(args.gt_dir, gt=True)

    if not pred_root.exists():
        fail("pred_dir does not exist: %s" % pred_root)
    if not gt_root.exists():
        fail("gt_dir does not exist: %s" % gt_root)

    output_csv = args.output_csv or (pred_root / "per_image_metrics.csv")
    output_json = args.output_json or (pred_root / "average_metrics.json")
    summary_txt = args.summary_txt

    gt_map = build_stem_map(gt_root)
    if not gt_map:
        fail("no GT images found under: %s" % gt_root)

    pred_map = build_prediction_map(pred_root, gt_map.keys(), args.pred_filename)
    missing = sorted(set(gt_map) - set(pred_map))
    if missing and not args.allow_missing:
        fail(
            "missing predictions for %d GT image(s), first missing: %s"
            % (len(missing), ", ".join(missing[:10]))
        )
    if missing:
        print("[w] missing predictions skipped: %s" % ", ".join(missing[:10]))

    rows = []
    for stem in sorted(set(gt_map) & set(pred_map)):
        gt_img = read_rgb(gt_map[stem])
        pred_img = read_rgb(pred_map[stem])
        if pred_img.size != gt_img.size:
            if args.no_resize_pred:
                fail(
                    "size mismatch for %s: pred %s vs gt %s"
                    % (stem, pred_img.size, gt_img.size)
                )
            pred_img = pred_img.resize(gt_img.size, Image.BICUBIC)

        gt = to_float_array(gt_img)
        pred = to_float_array(pred_img)
        metrics = quality_assess(pred, gt)
        row = {
            "image": stem,
            "pred_path": str(pred_map[stem]),
            "gt_path": str(gt_map[stem]),
            "width": gt_img.size[0],
            "height": gt_img.size[1],
        }
        row.update(metrics)
        rows.append(row)

    if not rows:
        fail("no matched prediction/GT pairs were evaluated.")

    write_csv(Path(output_csv), rows)
    payload = write_json(Path(output_json), rows)
    if summary_txt:
        write_summary(Path(summary_txt), rows, payload)

    print("[i] evaluated %d image(s)" % len(rows))
    print("[i] wrote per-image metrics: %s" % output_csv)
    print("[i] wrote average metrics: %s" % output_json)
    if summary_txt:
        print("[i] wrote summary: %s" % summary_txt)
    print(
        "[i] averages: "
        + " ".join("%s %.6f" % (key, payload["metrics"][key]) for key in METRICS)
    )


if __name__ == "__main__":
    main()
