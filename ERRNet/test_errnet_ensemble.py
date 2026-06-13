import argparse
import copy
import csv
import itertools
import json
import math
import os
import shutil
import sys
import tempfile
from os.path import join

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw

import data.reflect_dataset as datasets
import util.index as metric_index
import util.util as util
from models.errnet_model import (
    ERRNetCascadeModel,
    ERRNetModel,
    ERRNetTransformerCascadeModel,
    _torch_load_compat,
    tensor2im,
)
from models.prior_branch import ERRNetPriorBranchModel
from options.errnet.train_options import TrainOptions
from test_errnet import EVAL_DATASETS, TEST_DATASETS


METRIC_KEYS = ["PSNR", "SSIM", "NCC", "LMSE"]
SCORE_DESCRIPTION = (
    "equal mean of candidate-wise min-max normalized PSNR, SSIM, NCC, "
    "and inverted LMSE"
)
ALIGNMENT_DESCRIPTION = "top-left crop to common input/output/target height and width"

DEFAULT_MODEL_SPECS = {
    "baseline": {
        "type": "errnet",
        "checkpoint": "checkpoints/errnet/errnet_060_00463920.pt",
        "hyper": True,
    },
    "improved": {
        "type": "errnet",
        "checkpoint": "checkpoints/errnet_improved_loss_v1/errnet_060_00463920.pt",
        "hyper": True,
    },
    "attn": {
        "type": "errnet",
        "checkpoint": "checkpoints/errnet_attn_rebalanced_v1/errnet_060_00463920.pt",
        "hyper": True,
    },
    "prior": {
        "type": "prior",
        "checkpoint": (
            "checkpoints/errnet_prior_stage2_base_s030_id005_g002/"
            "errnet_prior_best.pt"
        ),
        "hyper": True,
    },
    "transformer": {
        "type": "transformer",
        "checkpoint": (
            "checkpoints/errnet_transformer_cascade_v2/"
            "errnet_transformer_cascade_latest.pt"
        ),
        "hyper": True,
    },
}

MODEL_CLASSES = {
    "errnet": ERRNetModel,
    "cascade": ERRNetCascadeModel,
    "prior": ERRNetPriorBranchModel,
    "transformer": ERRNetTransformerCascadeModel,
}


def _split_csv(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_float_csv(value):
    values = _split_csv(value)
    if values is None:
        return None
    return [float(item) for item in values]


def _load_config(path):
    if path is None:
        return {}
    with open(path, "r") as config_file:
        if path.lower().endswith(".json"):
            config = json.load(config_file)
        else:
            config = yaml.safe_load(config_file)
    return config or {}


def _merge_dict(base, override):
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_ensemble_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Adaptive multi-checkpoint ensemble evaluation for ERRNet.",
        epilog="""Dataset-wise config example:
model_specs:
  baseline:
    type: errnet
    checkpoint: checkpoints/errnet/errnet_latest.pt
    hyper: true
default:
  active_models: [baseline, improved]
  fusion: weighted
  weights: [0.5, 0.5]
datasets:
  ceilnet_table2:
    active_models: [baseline, improved, attn]
    model_tta:
      attn: true

Standard ERRNet options such as --gpu_ids, --nThreads and --no-verbose are
also accepted and forwarded to TrainOptions.
""",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(
            list(EVAL_DATASETS.keys()) + list(TEST_DATASETS.keys()) + ["custom"]
        ),
    )
    parser.add_argument("--data_root", default="./datasets/processed_data")
    parser.add_argument(
        "--input_dir",
        default="./datasets/raw_data/CEILNet/testdata_reflection_real",
    )
    parser.add_argument("--result_dir", default="./results/eval_ensemble")
    parser.add_argument("--save_subdir", default=None)
    parser.add_argument("--max_long_edge", type=int, default=None)
    parser.add_argument("--eval_size", type=int, default=None)
    parser.add_argument("--config", default=None, help="dataset-wise JSON/YAML config")

    parser.add_argument(
        "--models",
        default=None,
        help="active aliases, e.g. baseline,improved,attn,prior,transformer",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="generic checkpoint override in ALIAS=PATH form",
    )
    parser.add_argument(
        "--model_spec",
        action="append",
        default=[],
        help="generic model in ALIAS=TYPE:PATH form",
    )
    parser.add_argument("--baseline_checkpoint", default=None)
    parser.add_argument("--improved_checkpoint", default=None)
    parser.add_argument("--attn_checkpoint", default=None)
    parser.add_argument("--prior_checkpoint", default=None)
    parser.add_argument("--transformer_checkpoint", default=None)

    parser.add_argument(
        "--fusion",
        choices=["weighted", "gated"],
        default=None,
    )
    parser.add_argument("--weights", default=None, help="comma-separated weights")
    parser.add_argument("--base_model", default=None, help="base alias for gated fusion")
    parser.add_argument(
        "--expert_models",
        default=None,
        help="expert aliases for gated fusion; defaults to all non-base models",
    )
    parser.add_argument("--mask_blur", type=int, default=None)
    parser.add_argument("--mask_gamma", type=float, default=None)

    parser.add_argument("--tta", action="store_true", default=None)
    parser.add_argument("--no_tta", action="store_true", default=None)
    parser.add_argument(
        "--tta_models",
        default=None,
        help="comma-separated model aliases that use 4-way flip TTA",
    )

    parser.add_argument("--grid_search", action="store_true")
    parser.add_argument(
        "--grid_values",
        default=None,
        help="candidate scalar weights, e.g. 0,0.25,0.5,0.75,1",
    )
    parser.add_argument(
        "--grid_weights",
        default=None,
        help="explicit vectors separated by ';', e.g. 0.5,0.5;0.75,0.25",
    )
    parser.add_argument("--max_search_combinations", type=int, default=10000)
    parser.add_argument("--keep_cache", action="store_true")

    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def build_runtime_config(args):
    config = _load_config(args.config)
    runtime = _merge_dict(
        config.get("default", {}),
        config.get("datasets", {}).get(args.dataset, {}),
    )

    cli_overrides = {
        "active_models": _split_csv(args.models),
        "fusion": args.fusion,
        "weights": _parse_float_csv(args.weights),
        "base_model": args.base_model,
        "expert_models": _split_csv(args.expert_models),
        "tta_models": _split_csv(args.tta_models),
        "mask_blur": args.mask_blur,
        "mask_gamma": args.mask_gamma,
        "max_long_edge": args.max_long_edge,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            runtime[key] = value

    if args.tta:
        runtime["tta"] = True
    if args.no_tta:
        runtime["tta"] = False

    runtime.setdefault(
        "active_models",
        ["baseline", "improved", "attn", "prior", "transformer"],
    )
    runtime.setdefault("fusion", "weighted")
    runtime.setdefault("base_model", "improved")
    runtime.setdefault("mask_blur", 15)
    runtime.setdefault("mask_gamma", 1.0)
    runtime.setdefault("tta", False)
    runtime.setdefault("tta_models", [])
    runtime.setdefault("model_tta", {})
    runtime["active_models"] = _split_csv(runtime["active_models"])
    runtime["tta_models"] = _split_csv(runtime["tta_models"]) or []
    if runtime.get("expert_models") is not None:
        runtime["expert_models"] = _split_csv(runtime["expert_models"])
    if isinstance(runtime.get("weights"), str):
        runtime["weights"] = _parse_float_csv(runtime["weights"])
    return config, runtime


def build_model_specs(args, config):
    specs = _merge_dict(DEFAULT_MODEL_SPECS, config.get("model_specs", {}))

    explicit_paths = {
        "baseline": args.baseline_checkpoint,
        "improved": args.improved_checkpoint,
        "attn": args.attn_checkpoint,
        "prior": args.prior_checkpoint,
        "transformer": args.transformer_checkpoint,
    }
    for alias, path in explicit_paths.items():
        if path is not None:
            specs.setdefault(alias, {})["checkpoint"] = path

    for value in args.checkpoint:
        if "=" not in value:
            raise ValueError("--checkpoint must use ALIAS=PATH")
        alias, path = value.split("=", 1)
        specs.setdefault(alias.strip(), {})["checkpoint"] = path.strip()

    for value in args.model_spec:
        if "=" not in value or ":" not in value.split("=", 1)[1]:
            raise ValueError("--model_spec must use ALIAS=TYPE:PATH")
        alias, remainder = value.split("=", 1)
        model_type, path = remainder.split(":", 1)
        specs[alias.strip()] = {
            "type": model_type.strip(),
            "checkpoint": path.strip(),
            "hyper": True,
        }
    return specs


def build_dataloader(opt, args, runtime):
    max_long_edge = runtime.get("max_long_edge")
    if args.dataset in EVAL_DATASETS:
        spec = EVAL_DATASETS[args.dataset]
        if max_long_edge is None:
            max_long_edge = spec.get("max_long_edge")
        dataset = datasets.CEILTestDataset(
            join(args.data_root, spec["path"]),
            size=args.eval_size,
            max_long_edge=max_long_edge,
        )
        save_subdir = args.save_subdir or spec["save_subdir"]
        dataset_name = spec["dataset_name"]
        has_gt = True
    else:
        if args.dataset == "custom":
            input_dir = args.input_dir
            default_subdir = "custom"
        else:
            spec = TEST_DATASETS[args.dataset]
            input_dir = (
                args.input_dir
                if spec["path"] is None
                else join(args.input_dir, spec["path"])
            )
            default_subdir = spec["save_subdir"]
        dataset = datasets.RealDataset(
            input_dir,
            size=args.eval_size,
            max_long_edge=max_long_edge,
        )
        save_subdir = args.save_subdir or default_subdir
        dataset_name = args.dataset
        has_gt = False

    loader = datasets.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.nThreads,
        pin_memory=len(opt.gpu_ids) > 0,
    )
    return dataset_name, save_subdir, loader, has_gt


def _image_name(data, index):
    value = data.get("fn", str(index))
    if isinstance(value, (list, tuple)):
        value = value[0]
    return os.path.splitext(os.path.basename(str(value)))[0]


def _save_chw_png(chw, path):
    image = tensor2im(torch.from_numpy(chw).unsqueeze(0))
    Image.fromarray(image.astype(np.uint8)).save(path)


def _metric_image(chw):
    return tensor2im(torch.from_numpy(chw).unsqueeze(0))


def _quality_assess(output, target):
    output_image = _metric_image(output)
    target_image = _metric_image(target)
    height = min(output_image.shape[0], target_image.shape[0])
    width = min(output_image.shape[1], target_image.shape[1])
    return metric_index.quality_assess(
        output_image[:height, :width],
        target_image[:height, :width],
    )


def cache_dataset(loader, cache_dir, output_dir, has_gt):
    records = []
    data_dir = join(cache_dir, "data")
    util.mkdirs(data_dir)

    for index, data in enumerate(loader):
        name = _image_name(data, index)
        key = "%06d_%s" % (index, name)
        image_dir = join(output_dir, name)
        util.mkdirs(image_dir)

        input_chw = data["input"][0].detach().cpu().float().numpy()
        input_path = join(data_dir, key + "_input.npy")
        np.save(input_path, input_chw)
        _save_chw_png(input_chw, join(image_dir, "input.png"))
        _save_chw_png(input_chw, join(image_dir, "m_input.png"))

        target_path = None
        if has_gt:
            target_chw = data["target_t"][0].detach().cpu().float().numpy()
            target_path = join(data_dir, key + "_target.npy")
            np.save(target_path, target_chw)
            _save_chw_png(target_chw, join(image_dir, "gt.png"))
            _save_chw_png(target_chw, join(image_dir, "t_label.png"))

        records.append(
            {
                "index": index,
                "name": name,
                "key": key,
                "image_dir": image_dir,
                "input_path": input_path,
                "target_path": target_path,
                "outputs": {},
            }
        )
    return records


def _apply_checkpoint_config(opt, checkpoint_path, model_type):
    if model_type != "prior":
        return
    state = _torch_load_compat(checkpoint_path, map_location=torch.device("cpu"))
    for key, value in state.get("prior_config", {}).items():
        setattr(opt, key, value)
    opt.prior_detach_mask_features = True
    opt.prior_save_masks = False
    opt.prior_save_coarse = False
    del state


def initialize_model(base_opt, alias, spec):
    model_type = spec.get("type", "errnet")
    if model_type not in MODEL_CLASSES:
        raise ValueError(
            "Unknown model type %r for %s; choose from %s"
            % (model_type, alias, sorted(MODEL_CLASSES))
        )

    checkpoint_path = spec.get("checkpoint")
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("Checkpoint for %s not found: %s" % (alias, checkpoint_path))

    opt = copy.deepcopy(base_opt)
    opt.name = alias
    opt.isTrain = False
    opt.no_log = True
    opt.no_verbose = True
    opt.display_id = 0
    opt.resume = True
    opt.icnn_path = checkpoint_path
    opt.hyper = bool(spec.get("hyper", True))
    for key, value in spec.get("options", {}).items():
        setattr(opt, key, value)
    _apply_checkpoint_config(opt, checkpoint_path, model_type)

    model = MODEL_CLASSES[model_type]()
    model.initialize(opt)
    model._eval()
    return model


def model_uses_tta(alias, spec, runtime):
    if "tta" in spec:
        return bool(spec["tta"])
    model_tta = runtime.get("model_tta", {})
    if alias in model_tta:
        return bool(model_tta[alias])
    if alias in runtime.get("tta_models", []):
        return True
    return bool(runtime.get("tta", False))


def run_model_inference(
    base_opt,
    alias,
    spec,
    runtime,
    loader,
    records,
    cache_dir,
    has_gt,
):
    print("[i] loading model %s from %s" % (alias, spec["checkpoint"]))
    model = initialize_model(base_opt, alias, spec)
    use_tta = model_uses_tta(alias, spec, runtime)
    model_cache_dir = join(cache_dir, alias)
    util.mkdirs(model_cache_dir)
    metric_rows = []

    with torch.no_grad():
        for index, data in enumerate(loader):
            mode = "eval" if has_gt else "test"
            model.set_input(data, mode)
            output = model.forward_tta() if use_tta else model.forward()
            output_chw = output[0].detach().cpu().float().numpy()
            output_path = join(model_cache_dir, records[index]["key"] + ".npy")
            np.save(output_path, output_chw)
            records[index]["outputs"][alias] = output_path
            _save_chw_png(output_chw, join(records[index]["image_dir"], alias + ".png"))

            if has_gt:
                target = np.load(records[index]["target_path"])
                row = {"image": records[index]["name"]}
                row.update(_quality_assess(output_chw, target))
                metric_rows.append(row)
            util.progress_bar(
                index,
                len(loader),
                "%s%s" % (alias, " + TTA" if use_tta else ""),
            )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metric_rows


def _normalize_weights(weights):
    array = np.asarray(weights, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("weights must be a non-empty vector")
    if np.any(array < 0):
        raise ValueError("ensemble weights must be non-negative")
    total = float(array.sum())
    if total <= 0:
        raise ValueError("ensemble weights must have a positive sum")
    return (array / total).tolist()


def resolve_fusion_models(runtime, active_models):
    if runtime["fusion"] == "weighted":
        return list(active_models)

    base_model = runtime["base_model"]
    if base_model not in active_models:
        raise ValueError("gated base model %r is not active" % base_model)
    expert_models = runtime.get("expert_models")
    if expert_models is None:
        expert_models = [name for name in active_models if name != base_model]
    if not expert_models:
        raise ValueError("gated fusion needs at least one expert model")
    unknown = [name for name in expert_models if name not in active_models]
    if unknown:
        raise ValueError("inactive expert models: %s" % ",".join(unknown))
    return list(expert_models)


def resolve_initial_weights(runtime, active_models, fusion_models):
    weights = runtime.get("weights")
    if weights is None:
        return _normalize_weights([1.0] * len(fusion_models))
    if len(weights) == len(fusion_models):
        return _normalize_weights(weights)
    if runtime["fusion"] == "gated" and len(weights) == len(active_models):
        by_name = dict(zip(active_models, weights))
        return _normalize_weights([by_name[name] for name in fusion_models])
    raise ValueError(
        "received %d weights, but fusion uses %d models: %s"
        % (len(weights), len(fusion_models), ",".join(fusion_models))
    )


def _gaussian_blur(mask, kernel_size):
    kernel_size = max(1, int(kernel_size))
    if kernel_size == 1:
        return mask
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = max(kernel_size / 6.0, 1e-3)
    coords = torch.arange(kernel_size, dtype=mask.dtype, device=mask.device)
    coords = coords - (kernel_size - 1) / 2.0
    kernel_1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, kernel_size, kernel_size)
    return F.conv2d(mask, kernel_2d, padding=kernel_size // 2)


def build_residual_mask(input_chw, base_chw, blur_size, gamma):
    if gamma <= 0:
        raise ValueError("mask_gamma must be positive")
    input_tensor = torch.from_numpy(input_chw).unsqueeze(0)
    base_tensor = torch.from_numpy(base_chw).unsqueeze(0)
    residual = (input_tensor - base_tensor).abs().mean(dim=1, keepdim=True)
    residual = _gaussian_blur(residual, blur_size)
    minimum = residual.amin(dim=(2, 3), keepdim=True)
    maximum = residual.amax(dim=(2, 3), keepdim=True)
    mask = (residual - minimum) / (maximum - minimum).clamp_min(1e-8)
    if gamma != 1.0:
        mask = mask.clamp_min(1e-8).pow(gamma)
    return mask[0].numpy()


def weighted_average(outputs, model_names, weights):
    result = np.zeros_like(outputs[model_names[0]], dtype=np.float32)
    for model_name, weight in zip(model_names, weights):
        result += outputs[model_name].astype(np.float32, copy=False) * float(weight)
    return result


def align_for_fusion(input_chw, outputs, target_chw=None):
    arrays = [input_chw] + list(outputs.values())
    if target_chw is not None:
        arrays.append(target_chw)
    height = min(array.shape[-2] for array in arrays)
    width = min(array.shape[-1] for array in arrays)
    aligned_input = input_chw[..., :height, :width]
    aligned_outputs = {
        name: output[..., :height, :width]
        for name, output in outputs.items()
    }
    aligned_target = (
        target_chw[..., :height, :width] if target_chw is not None else None
    )
    return aligned_input, aligned_outputs, aligned_target


def fuse_outputs(input_chw, outputs, runtime, fusion_models, weights):
    if runtime["fusion"] == "weighted":
        return weighted_average(outputs, fusion_models, weights), None

    base = outputs[runtime["base_model"]].astype(np.float32, copy=False)
    expert = weighted_average(outputs, fusion_models, weights)
    mask = build_residual_mask(
        input_chw,
        base,
        runtime["mask_blur"],
        runtime["mask_gamma"],
    )
    final = (1.0 - mask) * base + mask * expert
    return final, mask


def build_search_candidates(args, initial_weights, dimension):
    if not args.grid_search:
        return [initial_weights]

    if args.grid_weights:
        candidates = []
        for vector in args.grid_weights.split(";"):
            values = _parse_float_csv(vector)
            if len(values) != dimension:
                raise ValueError(
                    "grid vector has %d values; expected %d" % (len(values), dimension)
                )
            candidates.append(_normalize_weights(values))
        return _deduplicate_weights(candidates)

    values = _parse_float_csv(args.grid_values or "0,0.25,0.5,0.75,1")
    combination_count = len(values) ** dimension
    if combination_count > args.max_search_combinations:
        raise ValueError(
            "grid has %d raw combinations; increase --max_search_combinations "
            "or use fewer --grid_values" % combination_count
        )
    candidates = [
        list(vector)
        for vector in itertools.product(values, repeat=dimension)
        if math.isclose(sum(vector), 1.0, rel_tol=0.0, abs_tol=1e-6)
    ]
    if not candidates:
        raise ValueError("grid_values produced no non-negative vectors summing to 1")
    return _deduplicate_weights([_normalize_weights(vector) for vector in candidates])


def _deduplicate_weights(candidates):
    unique = []
    seen = set()
    for weights in candidates:
        key = tuple(round(value, 10) for value in weights)
        if key not in seen:
            seen.add(key)
            unique.append(weights)
    return unique


def evaluate_candidates(records, candidates, runtime, active_models, fusion_models):
    totals = [{key: 0.0 for key in METRIC_KEYS} for _ in candidates]
    counts = [{key: 0 for key in METRIC_KEYS} for _ in candidates]

    for record_index, record in enumerate(records):
        input_chw = np.load(record["input_path"])
        target_chw = np.load(record["target_path"])
        outputs = {
            alias: np.load(record["outputs"][alias])
            for alias in active_models
        }
        input_chw, outputs, target_chw = align_for_fusion(
            input_chw,
            outputs,
            target_chw,
        )
        for candidate_index, weights in enumerate(candidates):
            fused, _ = fuse_outputs(
                input_chw,
                outputs,
                runtime,
                fusion_models,
                weights,
            )
            metrics = _quality_assess(fused, target_chw)
            for key in METRIC_KEYS:
                value = float(metrics[key])
                if np.isfinite(value):
                    totals[candidate_index][key] += value
                    counts[candidate_index][key] += 1
        util.progress_bar(record_index, len(records), "ensemble search")

    rows = []
    for weights, metric_totals, metric_counts in zip(candidates, totals, counts):
        row = {
            "weights": ",".join("%.10g" % value for value in weights),
        }
        for key in METRIC_KEYS:
            count = metric_counts[key]
            row[key] = metric_totals[key] / count if count else float("nan")
        rows.append(row)
    _attach_composite_scores(rows)
    return rows


def _attach_composite_scores(rows):
    for row in rows:
        row["score"] = 0.0
    valid_metric_count = 0
    for key in METRIC_KEYS:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        valid_metric_count += 1
        minimum = float(values[finite].min())
        maximum = float(values[finite].max())
        if math.isclose(minimum, maximum):
            normalized = np.ones_like(values)
        else:
            normalized = (values - minimum) / (maximum - minimum)
            if key == "LMSE":
                normalized = 1.0 - normalized
        for index, row in enumerate(rows):
            if finite[index]:
                row["score"] += float(normalized[index])
    if valid_metric_count:
        for row in rows:
            row["score"] /= valid_metric_count


def select_best_search_row(rows):
    return max(
        rows,
        key=lambda row: (
            row["score"],
            row["PSNR"],
            row["SSIM"],
            row["NCC"],
            -row["LMSE"],
        ),
    )


def save_search_results(path, rows, fusion_models):
    fieldnames = ["weights"] + METRIC_KEYS + ["score"]
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("[i] search models: %s" % ",".join(fusion_models))
    print("[i] saved ensemble search results to %s" % path)


def _average_metric_rows(rows):
    averages = {}
    for key in METRIC_KEYS:
        values = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
        if values:
            averages[key] = float(np.mean(values))
    return averages


def _make_comparison(image_dir, active_models, has_gt):
    panels = [("input", join(image_dir, "input.png"))]
    for alias in ("baseline", "improved"):
        if alias in active_models:
            panels.append((alias, join(image_dir, alias + ".png")))
    panels.append(("ensemble", join(image_dir, "ensemble.png")))
    if has_gt:
        panels.append(("gt", join(image_dir, "gt.png")))

    loaded = []
    for title, path in panels:
        image = Image.open(path).convert("RGB")
        image.thumbnail((384, 384), Image.BICUBIC)
        loaded.append((title, image.copy()))
    label_height = 24
    width = sum(image.width for _, image in loaded)
    height = max(image.height for _, image in loaded) + label_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for title, image in loaded:
        canvas.paste(image, (x, label_height))
        draw.text((x + 4, 4), title, fill="black")
        x += image.width
    canvas.save(join(image_dir, "comparison.png"))


def save_final_outputs(
    records,
    weights,
    runtime,
    active_models,
    fusion_models,
    has_gt,
):
    metric_rows = []
    for index, record in enumerate(records):
        input_chw = np.load(record["input_path"])
        outputs = {
            alias: np.load(record["outputs"][alias])
            for alias in active_models
        }
        target = np.load(record["target_path"]) if has_gt else None
        input_chw, outputs, target = align_for_fusion(
            input_chw,
            outputs,
            target,
        )
        fused, mask = fuse_outputs(
            input_chw,
            outputs,
            runtime,
            fusion_models,
            weights,
        )
        _save_chw_png(fused, join(record["image_dir"], "ensemble.png"))
        if mask is not None:
            _save_chw_png(mask, join(record["image_dir"], "residual_mask.png"))

        if has_gt:
            row = {"image": record["name"]}
            row.update(_quality_assess(fused, target))
            metric_rows.append(row)
        _make_comparison(record["image_dir"], active_models, has_gt)
        util.progress_bar(index, len(records), "saving ensemble")
    return metric_rows


def save_empty_metrics(output_dir, dataset_name, image_count):
    with open(join(output_dir, "per_image_metrics.csv"), "w", newline="") as csv_file:
        csv.DictWriter(csv_file, fieldnames=["image"] + METRIC_KEYS).writeheader()
    with open(join(output_dir, "summary.txt"), "w") as summary_file:
        summary_file.write("Dataset: %s\n" % dataset_name)
        summary_file.write("Images: %d\n" % image_count)
        summary_file.write("Ground truth unavailable; metrics were not computed.\n")


def append_summary(
    output_dir,
    runtime,
    active_models,
    fusion_models,
    weights,
    model_metric_rows,
    model_specs,
):
    with open(join(output_dir, "summary.txt"), "a") as summary_file:
        summary_file.write("\nEnsemble\n")
        summary_file.write("Fusion: %s\n" % runtime["fusion"])
        summary_file.write("Spatial alignment: %s\n" % ALIGNMENT_DESCRIPTION)
        summary_file.write("Search score: %s\n" % SCORE_DESCRIPTION)
        summary_file.write("Active models: %s\n" % ",".join(active_models))
        if runtime["fusion"] == "gated":
            summary_file.write("Base model: %s\n" % runtime["base_model"])
        summary_file.write("Weight models: %s\n" % ",".join(fusion_models))
        summary_file.write(
            "Weights: %s\n" % ",".join("%.10g" % value for value in weights)
        )
        summary_file.write(
            "TTA models: %s\n"
            % ",".join(
                alias
                for alias in active_models
                if model_uses_tta(alias, model_specs[alias], runtime)
            )
        )

        if model_metric_rows:
            summary_file.write("\nIndividual model averages\n")
            for alias in active_models:
                averages = _average_metric_rows(model_metric_rows.get(alias, []))
                if not averages:
                    continue
                summary_file.write(
                    "%s %s\n"
                    % (
                        alias,
                        " ".join(
                            "%s %.6f" % (key, averages[key])
                            for key in METRIC_KEYS
                            if key in averages
                        ),
                    )
                )


def main():
    args = parse_ensemble_args()
    config, runtime = build_runtime_config(args)
    model_specs = build_model_specs(args, config)

    option_parser = TrainOptions()
    option_parser.isTrain = False
    base_opt = option_parser.parse()
    base_opt.isTrain = False
    base_opt.no_log = True
    base_opt.display_id = 0
    base_opt.no_verbose = True
    cudnn.benchmark = len(base_opt.gpu_ids) > 0

    active_models = list(runtime["active_models"])
    if len(active_models) < 2:
        raise ValueError("adaptive ensemble needs at least two active models")
    if len(set(active_models)) != len(active_models):
        raise ValueError("active model aliases must be unique")
    missing_specs = [alias for alias in active_models if alias not in model_specs]
    if missing_specs:
        raise ValueError("missing model specs: %s" % ",".join(missing_specs))
    fusion_models = resolve_fusion_models(runtime, active_models)
    initial_weights = resolve_initial_weights(runtime, active_models, fusion_models)

    dataset_name, save_subdir, loader, has_gt = build_dataloader(
        base_opt,
        args,
        runtime,
    )
    output_dir = join(args.result_dir, save_subdir)
    util.mkdirs(output_dir)
    cache_dir = tempfile.mkdtemp(prefix=".ensemble_cache_", dir=output_dir)

    try:
        if args.grid_search and not has_gt:
            raise ValueError("grid search requires a dataset with ground truth")

        records = cache_dataset(loader, cache_dir, output_dir, has_gt)
        if not records:
            raise ValueError("dataset contains no images")
        model_metric_rows = {}
        for alias in active_models:
            model_metric_rows[alias] = run_model_inference(
                base_opt,
                alias,
                model_specs[alias],
                runtime,
                loader,
                records,
                cache_dir,
                has_gt,
            )

        candidates = build_search_candidates(
            args,
            initial_weights,
            len(fusion_models),
        )
        if has_gt:
            search_rows = evaluate_candidates(
                records,
                candidates,
                runtime,
                active_models,
                fusion_models,
            )
            best_row = select_best_search_row(search_rows)
            best_weights = _parse_float_csv(best_row["weights"])
        else:
            search_rows = [
                {
                    "weights": ",".join("%.10g" % value for value in initial_weights),
                    "PSNR": "",
                    "SSIM": "",
                    "NCC": "",
                    "LMSE": "",
                    "score": "",
                }
            ]
            best_row = search_rows[0]
            best_weights = initial_weights

        save_search_results(
            join(output_dir, "ensemble_search_results.csv"),
            search_rows,
            fusion_models,
        )
        best_payload = {
            "dataset": args.dataset,
            "fusion": runtime["fusion"],
            "active_models": active_models,
            "weight_models": fusion_models,
            "weights": {
                alias: weight for alias, weight in zip(fusion_models, best_weights)
            },
            "metrics": {
                key: best_row[key] for key in METRIC_KEYS if best_row[key] != ""
            },
            "composite_score": best_row["score"],
            "score_definition": SCORE_DESCRIPTION,
            "spatial_alignment": ALIGNMENT_DESCRIPTION,
            "checkpoints": {
                alias: model_specs[alias]["checkpoint"] for alias in active_models
            },
        }
        if runtime["fusion"] == "gated":
            best_payload["base_model"] = runtime["base_model"]
            best_payload["mask_blur"] = runtime["mask_blur"]
            best_payload["mask_gamma"] = runtime["mask_gamma"]
        with open(join(output_dir, "best_weights.json"), "w") as json_file:
            json.dump(best_payload, json_file, indent=2, sort_keys=True)

        final_metric_rows = save_final_outputs(
            records,
            best_weights,
            runtime,
            active_models,
            fusion_models,
            has_gt,
        )
        if has_gt:
            averages = util.AverageMeters()
            for row in final_metric_rows:
                averages.update({key: row[key] for key in METRIC_KEYS})
            util.save_eval_metrics(
                final_metric_rows,
                averages,
                output_dir,
                dataset_name,
            )
        else:
            save_empty_metrics(output_dir, dataset_name, len(records))

        append_summary(
            output_dir,
            runtime,
            active_models,
            fusion_models,
            best_weights,
            model_metric_rows,
            model_specs,
        )
        print("[i] best_weights: %s" % best_payload["weights"])
        if has_gt:
            print(
                "[i] ensemble metrics: %s"
                % " ".join(
                    "%s=%.6f" % (key, float(best_row[key])) for key in METRIC_KEYS
                )
            )
    finally:
        if args.keep_cache:
            print("[i] kept floating-point cache at %s" % cache_dir)
        else:
            shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
