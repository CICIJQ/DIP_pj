import argparse
import os
import sys
from os.path import join

import torch
import torch.backends.cudnn as cudnn

import data.prior_dataset as datasets
import util.util as util
from models.prior_branch import ERRNetPriorBranchModel
from options.errnet.train_options import TrainOptions


EVAL_DATASETS = {
    "ceilnet_table2": {
        "dataset_name": "testdata_table2",
        "path": "testdata_CEILNET_table2",
        "save_subdir": "CEILNet_table2",
    },
    "real20": {
        "dataset_name": "testdata_real",
        "path": "real20",
        "save_subdir": "real20",
        "max_long_edge": 512,
    },
    "postcard": {
        "dataset_name": "testdata_postcard",
        "path": "postcard",
        "save_subdir": "SIR2_postcard",
    },
    "objects": {
        "dataset_name": "testdata_objects",
        "path": "objects",
        "save_subdir": "SIR2_objects",
    },
    "wild": {
        "dataset_name": "testdata_wild",
        "path": "wild",
        "save_subdir": "SIR2_wild",
    },
    "sir2_withgt": {
        "dataset_name": "testdata_sir2",
        "path": "sir2_withgt",
        "save_subdir": "sir2_withgt",
    },
}


def parse_cli_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", required=True, choices=sorted(list(EVAL_DATASETS.keys()) + ["custom"]))
    parser.add_argument("--data_root", default="./datasets/processed_data")
    parser.add_argument("--input_dir", default="./datasets/raw_data/CEILNet/testdata_reflection_real")
    parser.add_argument("--result_dir", default="./results/eval_prior")
    parser.add_argument("--save_subdir", default=None)
    parser.add_argument("--max_long_edge", type=int, default=None)
    parser.add_argument("--eval_size", type=int, default=None)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--prior_base", choices=["errnet", "cascade"], default="errnet")
    parser.add_argument("--prior_init_icnn", type=str, default=None)
    parser.add_argument("--prior_head_feats", type=int, default=32)
    parser.add_argument("--prior_refine_feats", type=int, default=64)
    parser.add_argument("--prior_refine_blocks", type=int, default=4)
    parser.add_argument("--prior_delta_scale", type=float, default=1.0)
    parser.add_argument("--prior_gate_threshold", type=float, default=0.0)
    parser.add_argument("--prior_gate_gamma", type=float, default=1.0)
    parser.add_argument("--prior_freeze_base", action="store_true")
    parser.add_argument("--prior_target_source", choices=["diff", "reflection", "hybrid"], default="diff")
    parser.add_argument("--prior_target_norm", choices=["max", "quantile", "meanstd"], default="max")
    parser.add_argument("--prior_target_quantile", type=float, default=0.99)
    parser.add_argument("--prior_target_abs_floor", type=float, default=0.0)
    parser.add_argument("--prior_target_std_scale", type=float, default=3.0)
    parser.add_argument("--prior_target_low", type=float, default=0.05)
    parser.add_argument("--prior_target_high", type=float, default=0.5)
    parser.add_argument("--prior_target_gamma", type=float, default=1.0)
    parser.add_argument("--prior_target_blur", type=int, default=3)
    parser.add_argument("--prior_save_masks", action="store_true")
    parser.add_argument("--prior_save_coarse", action="store_true")
    args, remaining = parser.parse_known_args()
    args.prior_detach_mask_features = True
    sys.argv = [sys.argv[0]] + remaining
    return args


def attach_args(opt, args):
    for key, value in vars(args).items():
        if key not in ("dataset", "data_root", "input_dir", "result_dir", "save_subdir", "max_long_edge", "tta"):
            setattr(opt, key, value)


def build_eval_loader(opt, args):
    spec = EVAL_DATASETS[args.dataset]
    max_long_edge = args.max_long_edge if args.max_long_edge is not None else spec.get("max_long_edge")
    dataset = datasets.PriorCEILTestDataset(
        join(args.data_root, spec["path"]),
        size=args.eval_size,
        max_long_edge=max_long_edge)
    loader = datasets.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.nThreads,
        pin_memory=len(opt.gpu_ids) > 0)
    save_subdir = args.save_subdir or spec["save_subdir"]
    return spec["dataset_name"], save_subdir, loader


def build_custom_loader(opt, args):
    dataset = datasets.PriorRealDataset(args.input_dir, max_long_edge=args.max_long_edge)
    loader = datasets.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.nThreads,
        pin_memory=len(opt.gpu_ids) > 0)
    return args.save_subdir or "custom", loader


def evaluate(model, loader, dataset_name, savedir=None, tta=False):
    avg_meters = util.AverageMeters()
    metric_rows = []
    model._eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            metrics = model.eval(data, savedir=savedir, tta=tta)
            avg_meters.update(metrics)
            if metrics:
                data_name = getattr(model, "data_name", None)
                image_name = data_name[0] if isinstance(data_name, (list, tuple)) else data_name
                row = {"image": os.path.splitext(os.path.basename(image_name or str(i)))[0]}
                row.update(metrics)
                metric_rows.append(row)
            util.progress_bar(i, len(loader), str(avg_meters))
    if savedir is not None and metric_rows:
        util.save_eval_metrics(metric_rows, avg_meters, savedir, dataset_name)
    return avg_meters


def test(model, loader, savedir=None, tta=False):
    model._eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            model.test(data, savedir=savedir, tta=tta)
            util.progress_bar(i, len(loader))


def main():
    args = parse_cli_args()
    option_parser = TrainOptions()
    option_parser.isTrain = False
    opt = option_parser.parse()
    opt.isTrain = False
    opt.no_log = True
    opt.display_id = 0
    attach_args(opt, args)

    cudnn.benchmark = len(opt.gpu_ids) > 0
    model = ERRNetPriorBranchModel()
    model.initialize(opt)

    if args.dataset == "custom":
        save_subdir, loader = build_custom_loader(opt, args)
        test(model, loader, savedir=join(args.result_dir, save_subdir), tta=args.tta)
    else:
        dataset_name, save_subdir, loader = build_eval_loader(opt, args)
        savedir = None if args.no_save else join(args.result_dir, save_subdir)
        metrics = evaluate(
            model,
            loader,
            dataset_name,
            savedir=savedir,
            tta=args.tta)
        print(metrics)


if __name__ == "__main__":
    main()
