#!/usr/bin/env python3
"""Train an RDNet + NAFNet residual refiner on top of precomputed RDNet outputs."""

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.mg_rdnet_refiner import RDNetNAFRefiner  # noqa: E402
from ra_rdnet.mask import estimate_reflection_mask  # noqa: E402

try:
    from skimage.metrics import structural_similarity
except Exception:  # pragma: no cover - exercised only if skimage is absent.
    structural_similarity = None


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}
GENERIC_IMAGE_STEMS = {
    "blended",
    "clean",
    "gt",
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
    "t_label",
    "transmission",
    "transmission_layer",
    "xreflection_rdnet",
}
DEFAULT_AUTO_TRAIN_DATASETS = [
    "real_train",
    "rdnet_refiner_voc_synth_train",
    "real20",
    "testdata_CEILNET_table2",
    "sir2_withgt",
    "objects",
    "postcard",
    "wild",
    "reflection_pairs_tight_eval/testdata_CEILNET_table2",
]
AUTO_DISCOVERY_EXCLUDES = {
    "reflection_pairs_tight_nogt_eval/testdata_CEILNET_table2",
    "rdnet_refiner_smoke4",
}
BENCHMARK_EVAL_DATASETS = {
    "real20",
    "testdata_CEILNET_table2",
    "sir2_withgt",
    "objects",
    "postcard",
    "wild",
    "reflection_pairs_tight_eval/testdata_CEILNET_table2",
    "reflection_pairs_tight_nogt_eval/testdata_CEILNET_table2",
}
DATASET_RDNET_ALIASES = {
    "real_train": ["real_train"],
    "real20": ["real20"],
    "testdata_CEILNET_table2": ["CEILNet_table2", "testdata_CEILNET_table2"],
    "sir2_withgt": ["sir2_withgt"],
    "objects": ["SIR2_objects", "objects"],
    "postcard": ["SIR2_postcard", "postcard"],
    "wild": ["SIR2_wild", "wild"],
    "reflection_pairs_tight_eval/testdata_CEILNET_table2": [
        "self_collected_gt",
        "self_collected",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a lightweight RDNet + NAFNet residual refiner on top of "
            "precomputed XReflection-RDNet outputs."
        )
    )
    parser.add_argument("--train_input_dir", default=None, type=Path, help="Training reflection input I directory.")
    parser.add_argument("--train_gt_dir", default=None, type=Path, help="Training ground-truth transmission directory.")
    parser.add_argument("--train_rdnet_dir", default=None, type=Path, help="Training RDNet output T_rd directory.")
    parser.add_argument(
        "--train_data_root",
        default=Path("./datasets/processed_data"),
        type=Path,
        help="Root directory that contains paired processed datasets.",
    )
    parser.add_argument(
        "--train_rdnet_root",
        default=Path("./results/xreflection_rdnet"),
        type=Path,
        help="Root directory that contains per-dataset RDNet outputs.",
    )
    parser.add_argument(
        "--train_dataset_names",
        default="fair_train",
        help=(
            "Comma-separated dataset names under train_data_root. "
            "Default fair_train keeps training-only paired datasets and excludes "
            "benchmark/eval sets. Use all_paired only with --allow_benchmark_train_mix."
        ),
    )
    parser.add_argument(
        "--exclude_train_datasets",
        default="",
        help="Comma-separated dataset names to exclude when auto-discovering paired training data.",
    )
    parser.add_argument(
        "--allow_benchmark_train_mix",
        action="store_true",
        help=(
            "Explicitly allow benchmark/eval datasets such as real20, CEILNet table2, "
            "SIR2, or self-collected GT to enter the training split. This is not "
            "benchmark-fair and is disabled by default."
        ),
    )
    parser.add_argument("--val_input_dir", default=None, type=Path, help="Validation reflection input I directory.")
    parser.add_argument("--val_gt_dir", default=None, type=Path, help="Validation ground-truth transmission directory.")
    parser.add_argument("--val_rdnet_dir", default=None, type=Path, help="Validation RDNet output T_rd directory.")
    parser.add_argument("--save_dir", default=None, type=Path, help="Directory for checkpoints and train_log.csv.")
    parser.add_argument(
        "--checkpoint_dir",
        dest="save_dir",
        default=None,
        type=Path,
        help="Backward-compatible alias for --save_dir.",
    )
    parser.add_argument(
        "--resume_checkpoint",
        default=None,
        type=Path,
        help="Optional checkpoint to resume refiner training from. Omit to start from scratch.",
    )
    parser.add_argument(
        "--init_checkpoint",
        default=None,
        type=Path,
        help=(
            "Optional checkpoint used only to initialize model weights. "
            "This warm-starts training but does not restore optimizer/epoch."
        ),
    )
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=4, type=int, help="Per-GPU batch size.")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lambda_ref", default=4.0, type=float)
    parser.add_argument("--lambda_bg", default=0.25, type=float)
    parser.add_argument("--lambda_grad", default=0.1, type=float)
    parser.add_argument("--device", default="cuda:0", help="Single-process device. Under torchrun, LOCAL_RANK is used automatically.")
    parser.add_argument("--num_workers", default=4, type=int, help="DataLoader workers per GPU.")
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--patch_size", default=256, type=int, help="Random crop size. Use 0 to disable crops.")
    parser.add_argument("--mask_blur_radius", default=5, type=int)
    parser.add_argument("--mask_sensitivity", default=1.35, type=float)
    parser.add_argument("--mask_gamma", default=0.70, type=float)
    parser.add_argument("--mask_diff_weight", default=0.55, type=float)
    parser.add_argument("--mask_bright_weight", default=0.30, type=float)
    parser.add_argument("--mask_edge_weight", default=0.15, type=float)
    parser.add_argument("--mask_floor", default=0.02, type=float)
    parser.add_argument("--target_mask_gamma", default=0.85, type=float)
    parser.add_argument("--rdnet_filename", default="xreflection_rdnet.png", help="Preferred RDNet filename in stem subdirs.")
    parser.add_argument("--base_channels", default=48, type=int)
    parser.add_argument("--residual_scale", default=0.35, type=float)
    parser.add_argument(
        "--gate_mode",
        choices=["ungated", "learned_confidence"],
        default="ungated",
        help="Refinement mode. learned_confidence uses heuristic mask times a learned confidence map.",
    )
    parser.add_argument(
        "--gate_init_bias",
        default=4.0,
        type=float,
        help="Initial bias for the learned confidence head; larger values start closer to mask-only gating.",
    )
    parser.add_argument(
        "--sample_mode",
        choices=["balanced_dataset", "sqrt_balanced", "by_image"],
        default="balanced_dataset",
        help="How to sample the mixed full dataset during training.",
    )
    parser.add_argument(
        "--epoch_size",
        default=None,
        type=int,
        help="Number of samples drawn per epoch when using weighted sampling. Default uses dataset length.",
    )
    parser.add_argument("--val_ratio", default=0.1, type=float, help="Used only when validation dirs are omitted.")
    parser.add_argument("--max_train_images", default=None, type=int)
    parser.add_argument("--max_val_images", default=None, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--print_freq", default=20, type=int)
    return parser.parse_args()


def fail(message):
    raise SystemExit("ERROR: %s" % message)


def torch_load_checkpoint(path):
    path = Path(path).expanduser().resolve()
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


def load_matching_state_dict(model, state_dict, label):
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in model_state:
            skipped.append("%s (unexpected)" % key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped.append(
                "%s (shape %r != %r)"
                % (key, tuple(value.shape), tuple(model_state[key].shape))
            )
            continue
        compatible[key] = value

    missing = [key for key in model_state.keys() if key not in compatible]
    model.load_state_dict(compatible, strict=False)

    log_info(
        "[i] initialized %s with %d/%d matching tensors"
        % (label, len(compatible), len(model_state))
    )
    if skipped:
        log_info("[i] skipped %d tensor(s): %s" % (len(skipped), ", ".join(skipped[:8])))
    if missing:
        log_info("[i] left %d tensor(s) at fresh init: %s" % (len(missing), ", ".join(missing[:8])))


def resolve_device(device_name):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[w] CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def dist_ready():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if dist_ready() else 0


def get_world_size():
    return dist.get_world_size() if dist_ready() else 1


def is_main_process():
    return get_rank() == 0


def log_info(message):
    if is_main_process():
        print(message)


def sync_workers():
    if dist_ready():
        dist.barrier()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def init_distributed_mode(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        args.rank = 0
        args.local_rank = 0
        args.world_size = 1
        args.distributed = False
        return resolve_device(args.device)

    if not torch.cuda.is_available():
        fail("WORLD_SIZE > 1 but CUDA is not available.")
    if not str(args.device).startswith("cuda"):
        fail("multi-GPU training requires a CUDA device setting.")

    args.rank = int(os.environ.get("RANK", "0"))
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.world_size = world_size
    args.distributed = True

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return torch.device("cuda:%d" % args.local_rank)


def cleanup_distributed():
    if dist_ready():
        dist.destroy_process_group()


def is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def parse_name_list(value):
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def resolve_image_root(path, kind):
    root = Path(path).expanduser().resolve()
    if kind == "input" and (root / "blended").is_dir():
        return root / "blended"
    if kind == "gt" and (root / "transmission_layer").is_dir():
        return root / "transmission_layer"
    return root


def dataset_sort_key(name):
    preferred = {dataset: index for index, dataset in enumerate(DEFAULT_AUTO_TRAIN_DATASETS)}
    return (preferred.get(name, len(preferred)), name)


def is_benchmark_eval_dataset(dataset_name):
    return Path(dataset_name).as_posix() in BENCHMARK_EVAL_DATASETS


def discover_paired_dataset_roots(data_root):
    data_root = Path(data_root).expanduser().resolve()
    if not data_root.is_dir():
        fail("train_data_root does not exist: %s" % data_root)
    roots = {}
    for blended_dir in sorted(data_root.rglob("blended")):
        root = blended_dir.parent
        if not (root / "transmission_layer").is_dir():
            continue
        rel_name = root.relative_to(data_root).as_posix()
        if rel_name == ".":
            continue
        if rel_name in AUTO_DISCOVERY_EXCLUDES or "nogt" in rel_name.lower():
            continue
        roots[rel_name] = root
    if not roots:
        fail("no paired datasets with blended/ and transmission_layer/ were found under: %s" % data_root)
    return roots


def candidate_rdnet_aliases(dataset_name):
    aliases = []
    for alias in DATASET_RDNET_ALIASES.get(dataset_name, []):
        aliases.append(alias)
    rel_path = Path(dataset_name)
    aliases.extend(
        [
            dataset_name,
            rel_path.name,
            dataset_name.replace("/", "_"),
            dataset_name.replace("/", "-"),
        ]
    )
    deduped = []
    seen = set()
    for alias in aliases:
        if alias and alias not in seen:
            deduped.append(alias)
            seen.add(alias)
    return deduped


def resolve_rdnet_dir_for_dataset(dataset_name, rdnet_root):
    rdnet_root = Path(rdnet_root).expanduser().resolve()
    if not rdnet_root.is_dir():
        fail("train_rdnet_root does not exist: %s" % rdnet_root)
    tried = []
    for alias in candidate_rdnet_aliases(dataset_name):
        candidate = rdnet_root / alias
        tried.append(str(candidate))
        if candidate.is_dir():
            return candidate
    fail(
        "could not find RDNet outputs for dataset '%s' under %s; tried: %s"
        % (dataset_name, rdnet_root, ", ".join(tried))
    )


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


def collect_triplets(input_dir, gt_dir, rdnet_dir, rdnet_filename="xreflection_rdnet.png", allow_missing=False, dataset_name=None):
    input_root = resolve_image_root(input_dir, "input")
    gt_root = resolve_image_root(gt_dir, "gt")
    rdnet_root = Path(rdnet_dir).expanduser().resolve()
    if dataset_name is None:
        dataset_name = Path(input_dir).expanduser().resolve().name
    if not input_root.exists():
        fail("train/val input directory does not exist: %s" % input_root)
    if not gt_root.exists():
        fail("train/val GT directory does not exist: %s" % gt_root)
    if not rdnet_root.exists():
        fail("train/val RDNet directory does not exist: %s" % rdnet_root)

    input_map = build_image_map(input_root, "input")
    gt_map = build_image_map(gt_root, "gt")
    rdnet_map = build_image_map(rdnet_root, "rdnet", preferred_filename=rdnet_filename)
    if not input_map:
        fail("no input images found under: %s" % input_root)
    if not gt_map:
        fail("no GT images found under: %s" % gt_root)
    if not rdnet_map:
        fail("no RDNet images found under: %s" % rdnet_root)

    pairs = []
    missing = []
    for stem in sorted(gt_map):
        input_path = input_map.get(stem)
        rdnet_path = rdnet_map.get(stem)
        if input_path is None or rdnet_path is None:
            missing.append(stem)
            continue
        pairs.append(
            {
                "dataset": dataset_name,
                "stem": stem,
                "input": input_path,
                "gt": gt_map[stem],
                "rdnet": rdnet_path,
            }
        )
    if missing and not allow_missing:
        fail(
            "missing matched input/RDNet images for %d GT stem(s) in %s; first missing: %s"
            % (len(missing), dataset_name, ", ".join(missing[:10]))
        )
    if not pairs:
        fail("no matched I/GT/RDNet triplets found for dataset: %s" % dataset_name)
    return pairs


def require_explicit_training_triplet(args):
    return args.train_input_dir is not None or args.train_gt_dir is not None or args.train_rdnet_dir is not None


def build_train_dataset_specs(args):
    if require_explicit_training_triplet(args):
        if not (args.train_input_dir and args.train_gt_dir and args.train_rdnet_dir):
            fail(
                "provide all of --train_input_dir, --train_gt_dir, and --train_rdnet_dir, "
                "or omit them and use auto-discovery through --train_data_root/--train_rdnet_root."
            )
        dataset_name = Path(args.train_input_dir).expanduser().resolve().name
        return [
            {
                "name": dataset_name,
                "input_dir": args.train_input_dir,
                "gt_dir": args.train_gt_dir,
                "rdnet_dir": args.train_rdnet_dir,
            }
        ]

    discovered = discover_paired_dataset_roots(args.train_data_root)
    excludes = set(parse_name_list(args.exclude_train_datasets))
    requested_names = parse_name_list(args.train_dataset_names)
    if not requested_names:
        requested_names = ["fair_train"]

    if requested_names == ["fair_train"]:
        selected_names = [
            name
            for name in sorted(discovered, key=dataset_sort_key)
            if not is_benchmark_eval_dataset(name)
        ]
    elif requested_names == ["all_paired"]:
        selected_names = sorted(discovered, key=dataset_sort_key)
    else:
        selected_names = requested_names

    benchmark_selected = [name for name in selected_names if is_benchmark_eval_dataset(name)]
    if benchmark_selected and not args.allow_benchmark_train_mix:
        fail(
            "refusing to train on benchmark/eval datasets without "
            "--allow_benchmark_train_mix. Remove these from --train_dataset_names "
            "or use fair_train instead: %s"
            % ", ".join(sorted(benchmark_selected, key=dataset_sort_key))
        )

    specs = []
    missing = []
    for dataset_name in selected_names:
        if dataset_name in excludes:
            continue
        dataset_root = discovered.get(dataset_name)
        if dataset_root is None:
            missing.append(dataset_name)
            continue
        specs.append(
            {
                "name": dataset_name,
                "input_dir": dataset_root,
                "gt_dir": dataset_root,
                "rdnet_dir": resolve_rdnet_dir_for_dataset(dataset_name, args.train_rdnet_root),
            }
        )
    if missing:
        fail(
            "requested train dataset(s) were not found under %s: %s"
            % (Path(args.train_data_root).expanduser().resolve(), ", ".join(missing))
        )
    if not specs:
        fail("no training datasets remain after applying selection and exclusions.")
    return specs


def collect_pairs_from_specs(specs, rdnet_filename):
    pairs = []
    for spec in specs:
        dataset_pairs = collect_triplets(
            spec["input_dir"],
            spec["gt_dir"],
            spec["rdnet_dir"],
            rdnet_filename=rdnet_filename,
            dataset_name=spec["name"],
        )
        log_info(
            "[i] collected %d RDNet refiner triplets from %s"
            % (len(dataset_pairs), spec["name"])
        )
        pairs.extend(dataset_pairs)
    return pairs


def load_rgb_tensor(path, size=None):
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class MGRDNetTripletDataset(Dataset):
    def __init__(self, pairs, patch_size=256, augment=False, max_images=None):
        if max_images is not None:
            pairs = pairs[: int(max_images)]
        if not pairs:
            fail("empty RDNet refiner dataset after filtering.")
        self.pairs = pairs
        self.patch_size = int(patch_size) if patch_size else 0
        self.augment = bool(augment)

    def __len__(self):
        return len(self.pairs)

    def _resize_to_min_patch(self, tensors):
        if self.patch_size <= 0:
            return tensors
        height, width = tensors[0].shape[-2:]
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
        if self.patch_size <= 0:
            return tensors
        height, width = tensors[0].shape[-2:]
        if height == self.patch_size and width == self.patch_size:
            return tensors
        top = random.randint(0, height - self.patch_size)
        left = random.randint(0, width - self.patch_size)
        return [tensor[:, top : top + self.patch_size, left : left + self.patch_size] for tensor in tensors]

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
        gt_image = Image.open(pair["gt"]).convert("RGB")
        size = gt_image.size
        gt = torch.from_numpy(np.asarray(gt_image, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
        input_image = load_rgb_tensor(pair["input"], size=size)
        rdnet_image = load_rgb_tensor(pair["rdnet"], size=size)

        tensors = self._resize_to_min_patch([input_image, rdnet_image, gt])
        tensors = self._random_crop(tensors)
        tensors = self._augment(tensors)
        return {
            "dataset": pair["dataset"],
            "stem": pair["stem"],
            "input": tensors[0],
            "rdnet": tensors[1],
            "gt": tensors[2],
        }


def normalize_per_image(mask, eps=1e-6):
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    flat = mask.flatten(2)
    low = flat.amin(dim=2).view(mask.shape[0], mask.shape[1], 1, 1)
    high = flat.amax(dim=2).view(mask.shape[0], mask.shape[1], 1, 1)
    return ((mask - low) / (high - low + eps)).clamp(0.0, 1.0)


def make_input_reflection_mask(input_image, rdnet_image, args):
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


def make_target_reflection_mask(input_image, target_image, args):
    mask = (input_image - target_image).abs().mean(dim=1, keepdim=True)
    mask = normalize_per_image(mask)
    if args.target_mask_gamma > 0 and args.target_mask_gamma != 1.0:
        mask = mask.clamp(min=1e-6).pow(float(args.target_mask_gamma))
    return mask.clamp(0.0, 1.0)


def gradient_loss(pred, target):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return (pred_dx - target_dx).abs().mean() + (pred_dy - target_dy).abs().mean()


def compute_losses(t_out, gt, rdnet, m_star, args):
    abs_err = (t_out - gt).abs()
    loss_l1 = abs_err.mean()
    loss_ref = ((1.0 + args.lambda_ref * m_star) * abs_err).mean()
    loss_bg = ((1.0 - m_star) * (t_out - rdnet).abs()).mean()
    loss_grad = gradient_loss(t_out, gt)
    total = loss_l1 + loss_ref + args.lambda_bg * loss_bg + args.lambda_grad * loss_grad
    return {
        "total": total,
        "l1": loss_l1,
        "ref": loss_ref,
        "bg": loss_bg,
        "grad": loss_grad,
    }


def psnr_torch(pred, target, eps=1e-10):
    mse = torch.mean((pred.clamp(0, 1) - target.clamp(0, 1)) ** 2).item()
    if mse <= eps:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def tensor_to_numpy_image(tensor):
    return tensor.detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def ssim_numpy(pred, target):
    if structural_similarity is None:
        pred_mean = float(pred.mean())
        target_mean = float(target.mean())
        pred_var = float(pred.var())
        target_var = float(target.var())
        cov = float(((pred - pred_mean) * (target - target_mean)).mean())
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        return ((2 * pred_mean * target_mean + c1) * (2 * cov + c2)) / (
            (pred_mean ** 2 + target_mean ** 2 + c1) * (pred_var + target_var + c2)
        )
    height, width = pred.shape[:2]
    win_size = min(7, height, width)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        return float("nan")
    return float(structural_similarity(target, pred, data_range=1.0, channel_axis=-1, win_size=win_size))


def ssq_error(correct, estimate):
    denom = np.sum(estimate ** 2)
    alpha = np.sum(correct * estimate) / denom if denom > 1e-5 else 0.0
    return np.sum((correct - alpha * estimate) ** 2)


def lmse_numpy(pred, target, window_size=20, window_shift=10):
    height, width, channels = target.shape
    window_size = min(window_size, height, width)
    if window_size <= 0:
        return float("nan")
    window_shift = max(1, min(window_shift, window_size))
    ssq = 0.0
    total = 0.0
    for channel in range(channels):
        for row in range(0, height - window_size + 1, window_shift):
            for col in range(0, width - window_size + 1, window_shift):
                correct = target[row : row + window_size, col : col + window_size, channel]
                estimate = pred[row : row + window_size, col : col + window_size, channel]
                ssq += ssq_error(correct, estimate)
                total += np.sum(correct ** 2)
    if total < 1e-12:
        return float("nan")
    return float(ssq / total)


def evaluate(model, loader, args, device):
    if loader is None:
        return {"PSNR": float("nan"), "SSIM": float("nan"), "LMSE": float("nan")}
    model.eval()
    psnr_sum = 0.0
    ssim_sum = 0.0
    lmse_sum = 0.0
    psnr_count = 0.0
    ssim_count = 0.0
    lmse_count = 0.0
    with torch.no_grad():
        for batch in loader:
            input_image = batch["input"].to(device)
            rdnet = batch["rdnet"].to(device)
            gt = batch["gt"].to(device)
            mask = make_input_reflection_mask(input_image, rdnet, args)
            t_out, _, _ = model(input_image, rdnet, mask)
            for index in range(t_out.shape[0]):
                pred_np = tensor_to_numpy_image(t_out[index])
                gt_np = tensor_to_numpy_image(gt[index])
                psnr_value = psnr_torch(t_out[index], gt[index])
                ssim_value = ssim_numpy(pred_np, gt_np)
                lmse_value = lmse_numpy(pred_np, gt_np)
                if math.isfinite(psnr_value):
                    psnr_sum += float(psnr_value)
                    psnr_count += 1.0
                if math.isfinite(ssim_value):
                    ssim_sum += float(ssim_value)
                    ssim_count += 1.0
                if math.isfinite(lmse_value):
                    lmse_sum += float(lmse_value)
                    lmse_count += 1.0
    if dist_ready():
        reduced = torch.tensor(
            [psnr_sum, psnr_count, ssim_sum, ssim_count, lmse_sum, lmse_count],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        psnr_sum, psnr_count, ssim_sum, ssim_count, lmse_sum, lmse_count = reduced.tolist()
    model.train()
    return {
        "PSNR": float(psnr_sum / psnr_count) if psnr_count > 0 else float("nan"),
        "SSIM": float(ssim_sum / ssim_count) if ssim_count > 0 else float("nan"),
        "LMSE": float(lmse_sum / lmse_count) if lmse_count > 0 else float("nan"),
    }


def save_checkpoint(path, model, optimizer, epoch, args, best_psnr):
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = unwrap_model(model).state_dict()
    torch.save(
        {
            "epoch": epoch,
            "best_psnr": best_psnr,
            "state_dict": model_state,
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "model_config": {
                "base_channels": args.base_channels,
                "residual_scale": args.residual_scale,
                "gate_mode": args.gate_mode,
                "gate_init_bias": args.gate_init_bias,
            },
            "args": vars(args),
        },
        path,
    )


def append_train_log(path, row):
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_l1",
        "train_ref",
        "train_bg",
        "train_grad",
        "val_psnr",
        "val_ssim",
        "val_lmse",
        "lr",
    ]
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def split_pairs_for_validation(pairs, val_ratio, seed):
    if val_ratio <= 0 or len(pairs) < 2:
        return list(pairs), []
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair["dataset"]].append(pair)

    rng = random.Random(seed)
    train_pairs = []
    val_pairs = []
    for dataset_name in sorted(grouped, key=dataset_sort_key):
        items = list(grouped[dataset_name])
        rng.shuffle(items)
        if len(items) < 2:
            train_pairs.extend(items)
            continue
        val_count = max(1, int(round(len(items) * val_ratio)))
        val_count = min(val_count, len(items) - 1)
        val_pairs.extend(items[:val_count])
        train_pairs.extend(items[val_count:])
    return train_pairs, val_pairs


def build_train_sampler(pairs, args):
    if args.sample_mode == "by_image":
        return None
    counts = Counter(pair["dataset"] for pair in pairs)
    if not counts:
        return None
    if args.sample_mode == "balanced_dataset":
        total_datasets = float(len(counts))
        dataset_mass = {name: 1.0 / total_datasets for name in counts}
    elif args.sample_mode == "sqrt_balanced":
        raw_mass = {name: math.sqrt(count) for name, count in counts.items()}
        normalizer = sum(raw_mass.values())
        dataset_mass = {name: raw_mass[name] / normalizer for name in raw_mass}
    else:  # pragma: no cover - parse_args constrains the choices.
        return None

    weights = [dataset_mass[pair["dataset"]] / counts[pair["dataset"]] for pair in pairs]
    num_samples = args.epoch_size or len(pairs)
    if args.distributed:
        return DistributedWeightedSampler(
            weights,
            num_samples=num_samples,
            num_replicas=args.world_size,
            rank=args.rank,
            replacement=True,
            seed=args.seed,
        )
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )


class DistributedWeightedSampler(Sampler):
    def __init__(
        self,
        weights,
        num_samples,
        num_replicas=None,
        rank=None,
        replacement=True,
        seed=0,
    ):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        if self.weights.ndim != 1 or self.weights.numel() == 0:
            raise ValueError("weights must be a non-empty 1D sequence.")
        self.num_samples = int(num_samples)
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        self.num_replicas = int(num_replicas if num_replicas is not None else get_world_size())
        self.rank = int(rank if rank is not None else get_rank())
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples_per_rank = int(math.ceil(float(self.num_samples) / float(self.num_replicas)))
        self.total_size = self.num_samples_per_rank * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            self.replacement,
            generator=generator,
        ).tolist()
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self):
        return self.num_samples_per_rank

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


def count_by_dataset(pairs):
    return {name: count for name, count in sorted(Counter(pair["dataset"] for pair in pairs).items(), key=lambda item: dataset_sort_key(item[0]))}


def print_dataset_breakdown(label, pairs):
    counts = count_by_dataset(pairs)
    print("[i] %s images: %d" % (label, len(pairs)))
    for dataset_name, count in counts.items():
        print("    %s: %d" % (dataset_name, count))


def write_dataset_breakdown(path, train_pairs, val_pairs):
    summary = {
        "train_total": len(train_pairs),
        "val_total": len(val_pairs),
        "train_by_dataset": count_by_dataset(train_pairs),
        "val_by_dataset": count_by_dataset(val_pairs),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


def make_loaders(args):
    has_any_val = args.val_input_dir is not None or args.val_gt_dir is not None or args.val_rdnet_dir is not None
    if has_any_val and not (args.val_input_dir and args.val_gt_dir and args.val_rdnet_dir):
        fail("provide all of --val_input_dir, --val_gt_dir, and --val_rdnet_dir, or omit all of them.")

    train_specs = build_train_dataset_specs(args)
    train_pairs = collect_pairs_from_specs(train_specs, rdnet_filename=args.rdnet_filename)
    if not train_pairs:
        fail("no matched I/GT/RDNet triplets found for training.")

    if has_any_val:
        val_pairs = collect_triplets(
            args.val_input_dir,
            args.val_gt_dir,
            args.val_rdnet_dir,
            rdnet_filename=args.rdnet_filename,
            dataset_name=Path(args.val_input_dir).expanduser().resolve().name,
        )
    else:
        train_pairs, val_pairs = split_pairs_for_validation(train_pairs, args.val_ratio, args.seed)

    train_dataset = MGRDNetTripletDataset(
        train_pairs,
        patch_size=args.patch_size,
        augment=True,
        max_images=args.max_train_images,
    )
    val_dataset = None
    if val_pairs:
        val_dataset = MGRDNetTripletDataset(
            val_pairs,
            patch_size=0,
            augment=False,
            max_images=args.max_val_images,
        )

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    train_sampler = build_train_sampler(train_dataset.pairs, args)
    if args.distributed and train_sampler is None:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )
    val_loader = None
    if val_dataset is not None:
        val_sampler = None
        if args.distributed:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=False,
                drop_last=False,
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            sampler=val_sampler,
            num_workers=max(0, min(args.num_workers, 2)),
            pin_memory=use_cuda,
        )
    return train_loader, val_loader, train_dataset.pairs, (val_dataset.pairs if val_dataset is not None else [])


def main():
    args = parse_args()
    if args.save_dir is None:
        fail("provide --save_dir for RDNet + NAFNet refiner checkpoints.")
    if args.epochs <= 0:
        fail("--epochs must be positive.")
    if args.batch_size <= 0:
        fail("--batch_size must be positive.")
    if args.resume_checkpoint is not None and args.init_checkpoint is not None:
        fail("use only one of --resume_checkpoint or --init_checkpoint.")

    resume_state = None
    if args.resume_checkpoint is not None:
        resume_state = torch_load_checkpoint(args.resume_checkpoint)
        resume_config = checkpoint_model_config(resume_state)
        if "base_channels" in resume_config:
            args.base_channels = int(resume_config["base_channels"])
        if "residual_scale" in resume_config:
            args.residual_scale = float(resume_config["residual_scale"])
        if "gate_mode" in resume_config:
            args.gate_mode = str(resume_config["gate_mode"])
        if "gate_init_bias" in resume_config:
            args.gate_init_bias = float(resume_config["gate_init_bias"])

    init_state = None
    if args.init_checkpoint is not None:
        init_state = torch_load_checkpoint(args.init_checkpoint)

    device = init_distributed_mode(args)
    try:
        seed = int(args.seed) + int(get_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = True

        if is_main_process():
            args.save_dir.mkdir(parents=True, exist_ok=True)
            with (args.save_dir / "train_args.json").open("w") as handle:
                json.dump(vars(args), handle, indent=2, default=str)
                handle.write("\n")
        sync_workers()

        train_loader, val_loader, train_pairs, val_pairs = make_loaders(args)
        if is_main_process():
            print_dataset_breakdown("train", train_pairs)
            print_dataset_breakdown("val", val_pairs)
            write_dataset_breakdown(args.save_dir / "dataset_breakdown.json", train_pairs, val_pairs)
            print(
                "[i] training mode: %s | world_size=%d | per_gpu_batch=%d | global_batch=%d"
                % (
                    "ddp" if args.distributed else "single-process",
                    get_world_size(),
                    args.batch_size,
                    args.batch_size * max(1, get_world_size()),
                )
            )

        model = RDNetNAFRefiner(
            base_channels=args.base_channels,
            residual_scale=args.residual_scale,
            gate_mode=args.gate_mode,
            gate_init_bias=args.gate_init_bias,
        )

        best_psnr = float("-inf")
        start_epoch = 1
        if resume_state is not None:
            state_dict = checkpoint_state_dict(resume_state)
            model.load_state_dict(state_dict, strict=True)
            best_psnr = float(resume_state.get("best_psnr", best_psnr))
            start_epoch = int(resume_state.get("epoch", 0)) + 1
            log_info(
                "[i] resumed RDNet + NAFNet refiner from %s at epoch %d"
                % (args.resume_checkpoint, start_epoch - 1)
            )
        elif init_state is not None:
            load_matching_state_dict(
                model,
                checkpoint_state_dict(init_state),
                "RDNet + NAFNet refiner warm start",
            )
            log_info(
                "[i] warm-started learned confidence gating from %s; optimizer and epoch reset"
                % args.init_checkpoint
            )

        model = model.to(device)
        if args.distributed:
            model = DDP(model, device_ids=[device.index], output_device=device.index)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if isinstance(resume_state, dict) and resume_state.get("optimizer_state") is not None:
            optimizer.load_state_dict(resume_state["optimizer_state"])

        if start_epoch > args.epochs:
            fail(
                "resume checkpoint epoch %d is already beyond requested --epochs %d."
                % (start_epoch - 1, args.epochs)
            )

        log_path = args.save_dir / "train_log.csv"
        if is_main_process() and log_path.exists() and resume_state is None:
            log_path.unlink()

        for epoch in range(start_epoch, args.epochs + 1):
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            model.train()
            sums = {"total": 0.0, "l1": 0.0, "ref": 0.0, "bg": 0.0, "grad": 0.0}
            num_batches = 0
            for batch_index, batch in enumerate(train_loader, start=1):
                input_image = batch["input"].to(device, non_blocking=True)
                rdnet = batch["rdnet"].to(device, non_blocking=True)
                gt = batch["gt"].to(device, non_blocking=True)
                mask = make_input_reflection_mask(input_image, rdnet, args)
                m_star = make_target_reflection_mask(input_image, gt, args)

                t_out, _, _ = model(input_image, rdnet, mask)
                losses = compute_losses(t_out, gt, rdnet, m_star, args)
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                num_batches += 1
                for key in sums:
                    sums[key] += float(losses[key].detach().cpu())
                if is_main_process() and (batch_index == 1 or batch_index % args.print_freq == 0):
                    print(
                        "[i] epoch %d/%d batch %d/%d loss %.6f l1 %.6f ref %.6f bg %.6f grad %.6f"
                        % (
                            epoch,
                            args.epochs,
                            batch_index,
                            len(train_loader),
                            float(losses["total"].detach().cpu()),
                            float(losses["l1"].detach().cpu()),
                            float(losses["ref"].detach().cpu()),
                            float(losses["bg"].detach().cpu()),
                            float(losses["grad"].detach().cpu()),
                        )
                    )

            reduced_train = torch.tensor(
                [
                    sums["total"],
                    sums["l1"],
                    sums["ref"],
                    sums["bg"],
                    sums["grad"],
                    float(num_batches),
                ],
                device=device,
                dtype=torch.float64,
            )
            if dist_ready():
                dist.all_reduce(reduced_train, op=dist.ReduceOp.SUM)
            denom = max(1.0, float(reduced_train[5].item()))
            train_means = {
                "total": float(reduced_train[0].item() / denom),
                "l1": float(reduced_train[1].item() / denom),
                "ref": float(reduced_train[2].item() / denom),
                "bg": float(reduced_train[3].item() / denom),
                "grad": float(reduced_train[4].item() / denom),
            }

            val_metrics = evaluate(model, val_loader, args, device)
            if is_main_process():
                save_checkpoint(args.save_dir / "latest.pth", model, optimizer, epoch, args, best_psnr)
                if val_metrics["PSNR"] > best_psnr:
                    best_psnr = val_metrics["PSNR"]
                    save_checkpoint(args.save_dir / "best_psnr.pth", model, optimizer, epoch, args, best_psnr)
                append_train_log(
                    log_path,
                    {
                        "epoch": epoch,
                        "train_loss": "%.8f" % train_means["total"],
                        "train_l1": "%.8f" % train_means["l1"],
                        "train_ref": "%.8f" % train_means["ref"],
                        "train_bg": "%.8f" % train_means["bg"],
                        "train_grad": "%.8f" % train_means["grad"],
                        "val_psnr": "%.8f" % val_metrics["PSNR"],
                        "val_ssim": "%.8f" % val_metrics["SSIM"],
                        "val_lmse": "%.8f" % val_metrics["LMSE"],
                        "lr": "%.10f" % optimizer.param_groups[0]["lr"],
                    },
                )
                print(
                    "[i] epoch %d train %.6f val PSNR %.6f SSIM %.6f LMSE %.6f best %.6f"
                    % (
                        epoch,
                        train_means["total"],
                        val_metrics["PSNR"],
                        val_metrics["SSIM"],
                        val_metrics["LMSE"],
                        best_psnr,
                    )
                )

        if is_main_process():
            if not (args.save_dir / "best_psnr.pth").exists():
                save_checkpoint(args.save_dir / "best_psnr.pth", model, optimizer, args.epochs, args, best_psnr)
            print("[i] RDNet + NAFNet refiner training finished: %s" % args.save_dir)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
