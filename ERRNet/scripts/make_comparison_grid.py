#!/usr/bin/env python3
"""Create paper-ready visual comparison grids."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
PREFERRED = {
    "errnet": ("baseline_eval.png", "baseline.png", "rpt_gpu_baseline.png", "errnet.png"),
    "nafnet": ("naf_refiner.png", "naf.png"),
    "xreflection": ("xreflection_rdnet.png", "rdnet.png", "clean.png"),
    "rdnet_naf": (
        "rdnet_naf_refiner.png",
        "ra_rdnet_mg.png",
        "mg_rdnet_refiner.png",
        "rdnet_naf_refiner_posthoc_mask.png",
        "output.png",
        "clean.png",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a grid with columns: Input | ERRNet | NAFNet Refiner | "
            "XReflection-RDNet | optional RDNet + NAFNet Refiner | GT. "
            "The GT column is omitted when no GT is found."
        )
    )
    parser.add_argument("--input_dir", required=True, type=Path, help="Input images or dataset root.")
    parser.add_argument("--errnet_dir", required=True, type=Path, help="ERRNet result directory.")
    parser.add_argument("--nafnet_dir", required=True, type=Path, help="NAFNet Refiner result directory.")
    parser.add_argument(
        "--xreflection_dir",
        required=True,
        type=Path,
        help="XReflection-RDNet result directory.",
    )
    parser.add_argument(
        "--rdnet_naf_dir",
        default=None,
        type=Path,
        help="Optional RDNet + NAFNet Refiner result directory.",
    )
    parser.add_argument(
        "--gt_dir",
        default=None,
        type=Path,
        help="Optional GT directory or dataset root containing transmission_layer/.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output grid image path.")
    parser.add_argument(
        "--image_stems",
        nargs="*",
        default=None,
        help="Specific image stems to include. Defaults to the first max_images inputs.",
    )
    parser.add_argument("--max_images", type=int, default=8, help="Maximum rows when image_stems is omitted.")
    parser.add_argument("--cell_width", type=int, default=320, help="Rendered width of each image cell.")
    parser.add_argument("--cell_height", type=int, default=240, help="Rendered height of each image cell.")
    parser.add_argument("--label_height", type=int, default=28, help="Height reserved for column labels.")
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Use blank cells for missing method outputs instead of skipping the row.",
    )
    return parser.parse_args()


def fail(message):
    raise SystemExit("ERROR: %s" % message)


def is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def resolve_image_root(path, kind):
    path = path.expanduser().resolve()
    if kind == "input" and (path / "blended").is_dir():
        return path / "blended"
    if kind == "gt" and (path / "transmission_layer").is_dir():
        return path / "transmission_layer"
    return path


def build_flat_map(root):
    if root is None or not root.exists():
        return {}
    mapping = {}
    for path in sorted(root.rglob("*")):
        if is_image(path) and path.stem not in mapping:
            mapping[path.stem] = path
    return mapping


def find_method_image(root, stem, method):
    for filename in PREFERRED[method]:
        nested = root / stem / filename
        if nested.is_file():
            return nested
    direct_candidates = []
    for path in root.rglob("*"):
        if not is_image(path):
            continue
        if path.stem == stem or path.parent.name == stem:
            direct_candidates.append(path)
    if not direct_candidates:
        return None

    def score(path):
        name = path.name.lower()
        value = 0
        if path.parent.name == stem:
            value += 20
        if path.stem == stem:
            value += 10
        for index, preferred in enumerate(PREFERRED[method]):
            if name == preferred:
                value += 100 - index
        if "clean" in path.stem.lower():
            value += 15
        lower_stem = path.stem.lower()
        if any(token in lower_stem for token in ("input", "gt", "label")):
            value -= 100
        if "reflection" in lower_stem and "xreflection" not in lower_stem:
            value -= 100
        return value

    return sorted(direct_candidates, key=score, reverse=True)[0]


def fit_image(path, width, height):
    image = Image.open(path).convert("RGB")
    image.thumbnail((width, height), Image.BICUBIC)
    canvas = Image.new("RGB", (width, height), "white")
    left = (width - image.width) // 2
    top = (height - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def blank_cell(width, height, text):
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), text, fill=(80, 80, 80))
    return canvas


def draw_label(draw, x, y, width, text):
    try:
        bbox = draw.textbbox((0, 0), text)
        text_width = bbox[2] - bbox[0]
    except AttributeError:
        text_width = draw.textsize(text)[0]
    draw.text((x + max(4, (width - text_width) // 2), y + 6), text, fill="black")


def main():
    args = parse_args()
    input_root = resolve_image_root(args.input_dir, "input")
    gt_root = resolve_image_root(args.gt_dir, "gt") if args.gt_dir else None
    errnet_root = args.errnet_dir.expanduser().resolve()
    nafnet_root = args.nafnet_dir.expanduser().resolve()
    xreflection_root = args.xreflection_dir.expanduser().resolve()
    rdnet_naf_root = (
        args.rdnet_naf_dir.expanduser().resolve()
        if args.rdnet_naf_dir is not None
        else None
    )

    for label, root in (
        ("input_dir", input_root),
        ("errnet_dir", errnet_root),
        ("nafnet_dir", nafnet_root),
        ("xreflection_dir", xreflection_root),
    ):
        if not root.exists():
            fail("%s does not exist: %s" % (label, root))
    if rdnet_naf_root is not None and not rdnet_naf_root.exists():
        fail("rdnet_naf_dir does not exist: %s" % rdnet_naf_root)

    input_map = build_flat_map(input_root)
    gt_map = build_flat_map(gt_root) if gt_root else {}
    if not input_map:
        fail("no input images found under: %s" % input_root)

    stems = args.image_stems or sorted(input_map)[: args.max_images]
    rows = []
    missing_messages = []
    for stem in stems:
        paths = {
            "Input": input_map.get(stem),
            "ERRNet": find_method_image(errnet_root, stem, "errnet"),
            "NAFNet Refiner": find_method_image(nafnet_root, stem, "nafnet"),
            "XReflection-RDNet": find_method_image(xreflection_root, stem, "xreflection"),
        }
        if rdnet_naf_root is not None:
            paths["RDNet + NAFNet Refiner"] = find_method_image(
                rdnet_naf_root,
                stem,
                "rdnet_naf",
            )
        if gt_map.get(stem):
            paths["GT"] = gt_map[stem]

        missing_required = [
            label for label, path in paths.items() if label != "GT" and path is None
        ]
        if missing_required and not args.allow_missing:
            missing_messages.append("%s: %s" % (stem, ", ".join(missing_required)))
            continue
        rows.append((stem, paths))

    if missing_messages:
        fail("missing required images; first cases: %s" % "; ".join(missing_messages[:5]))
    if not rows:
        fail("no complete rows to render.")

    include_gt = any("GT" in paths for _, paths in rows)
    columns = ["Input", "ERRNet", "NAFNet Refiner", "XReflection-RDNet"]
    if rdnet_naf_root is not None:
        columns.append("RDNet + NAFNet Refiner")
    if include_gt:
        columns.append("GT")

    grid_width = args.cell_width * len(columns)
    row_height = args.label_height + args.cell_height
    grid_height = args.label_height + args.cell_height * len(rows)
    canvas = Image.new("RGB", (grid_width, grid_height), "white")
    draw = ImageDraw.Draw(canvas)

    for col_index, label in enumerate(columns):
        draw_label(draw, col_index * args.cell_width, 0, args.cell_width, label)

    y = args.label_height
    for _, paths in rows:
        for col_index, label in enumerate(columns):
            path = paths.get(label)
            if path is None:
                cell = blank_cell(args.cell_width, args.cell_height, "Missing")
            else:
                cell = fit_image(path, args.cell_width, args.cell_height)
            canvas.paste(cell, (col_index * args.cell_width, y))
        y += args.cell_height

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print("[i] wrote comparison grid: %s" % args.output)


if __name__ == "__main__":
    main()
