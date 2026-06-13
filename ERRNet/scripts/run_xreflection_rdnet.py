#!/usr/bin/env python3
"""Run XReflection RDNet inference on a directory of reflection images.

The wrapper intentionally treats XReflection as an external project.  By
default it generates a small one-dataset config from the supplied RDNet config
and launches XReflection's official test entry point in a subprocess.  If the
upstream command changes, pass --command_template instead of editing this repo.
"""

import argparse
import csv
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in missing envs.
    yaml = None


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_COMMAND_TEMPLATE = (
    "{python} {xreflection_train} --config {generated_config} "
    "--test_only {checkpoint}"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-run RDNet/XReflection inference via subprocess."
    )
    parser.add_argument(
        "--xreflection_root",
        required=True,
        type=Path,
        help="Path to the cloned XReflection repository.",
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=Path,
        help=(
            "Input image directory, or a paired dataset root containing "
            "blended/ and optionally transmission_layer/."
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory where normalized RDNet outputs will be written.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to an XReflection RDNet yaml config, e.g. options/train_rdnet.yml.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to an RDNet checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device hint such as cuda:0 or cpu. cuda:N sets CUDA_VISIBLE_DEVICES=N.",
    )
    parser.add_argument(
        "--command_template",
        default=None,
        help=(
            "Optional XReflection command template. Supported placeholders: "
            "{python}, {xreflection_root}, {xreflection_train}, {input_dir}, "
            "{prepared_input_dir}, {output_dir}, {config}, {generated_config}, "
            "{checkpoint}, {device}, {run_root}, {run_name}, {dataset_name}. "
            "Values are shell-quoted automatically; use *_raw placeholders if "
            "you need unquoted text."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch XReflection.",
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        help="Dataset name used inside the generated XReflection config.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Experiment name used inside the generated XReflection config.",
    )
    parser.add_argument(
        "--normalized_filename",
        default="xreflection_rdnet.png",
        help="Filename for each normalized clean output under output_dir/<stem>/.",
    )
    parser.add_argument(
        "--copy_inputs",
        action="store_true",
        help="Copy instead of symlink flat input images when preparing pseudo paired data.",
    )
    parser.add_argument(
        "--max_long_edge",
        type=int,
        default=0,
        help=(
            "If positive, prepare resized inputs whose longer edge is at most "
            "this value before launching XReflection. Useful for large images "
            "that otherwise run out of GPU memory."
        ),
    )
    parser.add_argument(
        "--no_normalize_outputs",
        action="store_true",
        help="Only run XReflection; do not copy clean images into output_dir/<stem>/.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the resolved command and generated config path without running.",
    )
    return parser.parse_args()


def fail(message):
    raise SystemExit("ERROR: %s" % message)


def is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_images(directory):
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if is_image(path))


def resize_or_link_image(src, dst, max_long_edge, copy_inputs=False):
    if dst.exists() or dst.is_symlink():
        if dst.is_dir():
            fail("cannot replace directory while preparing input: %s" % dst)
        dst.unlink()

    if max_long_edge and max_long_edge > 0:
        image = Image.open(src).convert("RGB")
        width, height = image.size
        long_edge = max(width, height)
        if long_edge > max_long_edge:
            scale = float(max_long_edge) / float(long_edge)
            new_size = (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            image = image.resize(new_size, Image.BICUBIC)
        image.save(dst)
        return

    if copy_inputs:
        shutil.copy2(src, dst)
        return

    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def check_paths(args):
    args.xreflection_root = args.xreflection_root.expanduser().resolve()
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()

    if not args.xreflection_root.is_dir():
        fail("XReflection root does not exist: %s" % args.xreflection_root)
    if not args.input_dir.exists():
        fail("input_dir does not exist: %s" % args.input_dir)
    if not args.config.is_file():
        fail("config file does not exist: %s" % args.config)
    if not args.checkpoint.is_file():
        fail("checkpoint file does not exist: %s" % args.checkpoint)
    if yaml is None:
        fail("PyYAML is required to generate the temporary XReflection config.")

    train_py = args.xreflection_root / "xreflection" / "tools" / "train.py"
    if args.command_template is None and not train_py.is_file():
        fail("cannot find XReflection train entry point: %s" % train_py)


def safe_name(value):
    cleaned = []
    for char in str(value):
        cleaned.append(char if char.isalnum() or char in "-_." else "_")
    return "".join(cleaned).strip("_") or "dataset"


def prepared_root_for_input(args, dataset_name):
    if (args.input_dir / "blended").is_dir() and not args.max_long_edge:
        return args.input_dir

    source_root = args.input_dir / "blended" if (args.input_dir / "blended").is_dir() else args.input_dir
    input_images = list_images(source_root)
    if not input_images:
        fail(
            "input_dir must either contain images or contain a blended/ subdirectory: %s"
            % args.input_dir
        )

    prepared_root = args.output_dir / "_prepared_input" / dataset_name
    blended_dir = prepared_root / "blended"
    pseudo_gt_dir = prepared_root / "transmission_layer"
    blended_dir.mkdir(parents=True, exist_ok=True)
    pseudo_gt_dir.mkdir(parents=True, exist_ok=True)

    for src in input_images:
        for dst_dir in (blended_dir, pseudo_gt_dir):
            dst = dst_dir / src.name
            resize_or_link_image(
                src,
                dst,
                max_long_edge=args.max_long_edge,
                copy_inputs=args.copy_inputs,
            )

    if args.max_long_edge:
        print("[i] prepared resized input with max_long_edge=%d: %s" % (args.max_long_edge, prepared_root))
    else:
        print("[i] prepared flat input directory as pseudo paired data: %s" % prepared_root)
    return prepared_root


def device_to_lightning(device):
    lowered = device.lower()
    if lowered == "cpu":
        return "cpu", 1, None
    if lowered.startswith("cuda"):
        parts = lowered.split(":", 1)
        visible = parts[1] if len(parts) == 2 and parts[1] else None
        return "gpu", 1, visible
    return "auto", "auto", None


def load_yaml(path):
    with path.open("r") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        fail("config is not a YAML mapping: %s" % path)
    return data


def write_generated_config(
    args,
    prepared_input_dir,
    dataset_name,
    run_name,
    run_root,
    accelerator,
    devices,
):
    config = load_yaml(args.config)

    config["name"] = run_name
    config["test_only"] = True
    config["accelerator"] = accelerator
    config["devices"] = devices

    config.setdefault("path", {})
    config["path"]["experiments_root"] = str(run_root)

    config.setdefault("logger", {})
    config["logger"].setdefault("wandb", {})
    config["logger"]["wandb"]["enable"] = False

    config.setdefault("lightning", {})
    config["lightning"]["strategy"] = "auto"

    config.setdefault("val", {})
    config["val"]["save_img"] = True
    config["val"]["save_img_top_n"] = 1000000

    config.setdefault("datasets", {})
    config["datasets"]["val_datasets"] = [
        {
            "name": dataset_name,
            "type": "DSRTestDataset",
            "mode": "eval",
            "datadir": str(prepared_input_dir),
            "io_backend": {"type": "disk"},
            "use_shuffle": False,
            "num_worker_per_gpu": 0,
            "batch_size_per_gpu": 1,
        }
    ]

    generated_config = args.output_dir / "_xreflection_rdnet_config.yml"
    generated_config.parent.mkdir(parents=True, exist_ok=True)
    with generated_config.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return generated_config


def command_mapping(
    args,
    prepared_input_dir,
    generated_config,
    run_root,
    run_name,
    dataset_name,
):
    train_py = args.xreflection_root / "xreflection" / "tools" / "train.py"
    raw = {
        "python": args.python,
        "xreflection_root": str(args.xreflection_root),
        "xreflection_train": str(train_py),
        "input_dir": str(args.input_dir),
        "prepared_input_dir": str(prepared_input_dir),
        "output_dir": str(args.output_dir),
        "config": str(args.config),
        "generated_config": str(generated_config),
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "run_root": str(run_root),
        "run_name": run_name,
        "dataset_name": dataset_name,
    }
    quoted = {key: shlex.quote(str(value)) for key, value in raw.items()}
    quoted.update({key + "_raw": str(value) for key, value in raw.items()})
    return quoted


class SafeFormatDict(dict):
    def __missing__(self, key):
        fail("unknown command_template placeholder: {%s}" % key)


def build_command(template, mapping):
    command_string = template.format_map(SafeFormatDict(mapping))
    try:
        return shlex.split(command_string)
    except ValueError as exc:
        fail("could not parse command_template: %s" % exc)


def run_command(args, command, cuda_visible):
    print("[i] running XReflection command:")
    print("    " + " ".join(shlex.quote(part) for part in command))

    if args.dry_run:
        print("[i] dry run requested; command was not executed.")
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(args.xreflection_root)
        if not env.get("PYTHONPATH")
        else str(args.xreflection_root) + os.pathsep + env["PYTHONPATH"]
    )
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
    env.setdefault("TORCH_HOME", "/tmp/torch_cache")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    if cuda_visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    elif args.device.lower() == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""

    subprocess.run(command, cwd=str(args.xreflection_root), env=env, check=True)


def input_stems(prepared_input_dir):
    image_root = prepared_input_dir / "blended"
    return {path.stem for path in list_images(image_root)}


def score_candidate(path, stem):
    name = path.name.lower()
    path_stem = path.stem.lower()
    score = 0
    if path.parent.name == stem:
        score += 40
    if path.stem == stem:
        score += 30
    if "clean" in path_stem:
        score += 50
    if "xreflection" in path_stem or "rdnet" in path_stem:
        score += 20
    if name in {"output.png", "pred.png", "result.png"}:
        score += 10
    if any(token in path_stem for token in ("input", "gt", "label")):
        score -= 80
    if "reflection" in path_stem and "xreflection" not in path_stem:
        score -= 80
    return score


def collect_candidates(source_root, stems):
    candidates = {stem: [] for stem in stems}
    if not source_root.exists():
        return candidates

    for path in source_root.rglob("*"):
        if not is_image(path):
            continue
        parent_stem = path.parent.name
        file_stem = path.stem
        if parent_stem in candidates:
            candidates[parent_stem].append(path)
        if file_stem in candidates:
            candidates[file_stem].append(path)
    return candidates


def normalize_outputs(args, prepared_input_dir, run_root, run_name, dataset_name):
    if args.no_normalize_outputs:
        return

    stems = input_stems(prepared_input_dir)
    if not stems:
        fail("no input images found under prepared blended directory.")

    expected_visual = run_root / run_name / "visualization" / dataset_name
    search_roots = [expected_visual]
    if not expected_visual.exists():
        search_roots.append(args.output_dir)

    candidates = {stem: [] for stem in stems}
    for root in search_roots:
        for stem, paths in collect_candidates(root, stems).items():
            candidates[stem].extend(paths)

    manifest_rows = []
    missing = []
    for stem in sorted(stems):
        paths = candidates.get(stem, [])
        if not paths:
            missing.append(stem)
            continue
        best = sorted(
            paths,
            key=lambda path: (score_candidate(path, stem), path.stat().st_mtime),
            reverse=True,
        )[0]
        target_dir = args.output_dir / stem
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / args.normalized_filename
        if best.resolve() != target_path.resolve():
            shutil.copy2(best, target_path)
        manifest_rows.append(
            {
                "image": stem,
                "source_path": str(best),
                "normalized_path": str(target_path),
            }
        )

    manifest_path = args.output_dir / "xreflection_outputs_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image", "source_path", "normalized_path"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("[i] normalized %d RDNet outputs into %s" % (len(manifest_rows), args.output_dir))
    print("[i] wrote manifest: %s" % manifest_path)
    if missing:
        print(
            "[w] no clean output was found for %d image(s): %s"
            % (len(missing), ", ".join(missing[:10]))
        )


def main():
    args = parse_args()
    check_paths(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = safe_name(args.dataset_name or args.input_dir.name)
    run_name = safe_name(args.run_name or ("rdnet_" + dataset_name))
    run_root = args.output_dir / "_xreflection_experiments"

    prepared_input_dir = prepared_root_for_input(args, dataset_name)
    accelerator, devices, cuda_visible = device_to_lightning(args.device)
    generated_config = write_generated_config(
        args,
        prepared_input_dir,
        dataset_name,
        run_name,
        run_root,
        accelerator,
        devices,
    )

    template = args.command_template or DEFAULT_COMMAND_TEMPLATE
    mapping = command_mapping(
        args,
        prepared_input_dir,
        generated_config,
        run_root,
        run_name,
        dataset_name,
    )
    command = build_command(template, mapping)

    print("[i] generated XReflection config: %s" % generated_config)
    run_command(args, command, cuda_visible)
    if not args.dry_run:
        normalize_outputs(args, prepared_input_dir, run_root, run_name, dataset_name)


if __name__ == "__main__":
    main()
