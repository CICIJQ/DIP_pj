import argparse
import csv
import os
import sys
import time
from collections import OrderedDict
from os.path import join

import torch
import torch.backends.cudnn as cudnn

import util.util as util
import data.prior_dataset as datasets
from data.image_folder import read_fns
from models.prior_branch import ERRNetPriorBranchModel
from options.errnet.train_options import TrainOptions
from util.visualizer import Visualizer


EVAL_DATASETS = {
    "ceilnet_table2": ("testdata_table2", "testdata_CEILNET_table2", "CEILNet_table2", None),
    "real20": ("testdata_real", "real20", "real20", 512),
    "sir2_withgt": ("testdata_sir2", "sir2_withgt", "sir2_withgt", None),
    "objects": ("testdata_objects", "objects", "SIR2_objects", None),
    "postcard": ("testdata_postcard", "postcard", "SIR2_postcard", None),
    "wild": ("testdata_wild", "wild", "SIR2_wild", None),
}


def parse_prior_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prior_base", choices=["errnet", "cascade"], default="errnet")
    parser.add_argument("--prior_init_icnn", type=str, default=None)
    parser.add_argument("--prior_train_synthesis", choices=["baseline", "realistic"], default="realistic")
    parser.add_argument("--prior_real_ratio", type=float, default=0.3)
    parser.add_argument("--prior_lambda_mask", type=float, default=0.1)
    parser.add_argument("--prior_lambda_gate", type=float, default=0.05)
    parser.add_argument("--prior_lambda_smooth", type=float, default=0.01)
    parser.add_argument("--prior_lambda_sparse", type=float, default=0.0)
    parser.add_argument("--prior_lambda_identity", type=float, default=0.0)
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
    parser.add_argument("--prior_head_feats", type=int, default=32)
    parser.add_argument("--prior_refine_feats", type=int, default=64)
    parser.add_argument("--prior_refine_blocks", type=int, default=4)
    parser.add_argument("--prior_init_type", type=str, default="kaiming")
    parser.add_argument("--prior_joint_mask_features", action="store_true")
    parser.add_argument("--prior_eval_freq", type=int, default=5)
    parser.add_argument("--prior_eval_datasets", type=str, default="real20,ceilnet_table2")
    parser.add_argument("--prior_result_dir", type=str, default="results/eval_prior")
    parser.add_argument("--prior_save_masks", action="store_true")
    parser.add_argument("--prior_save_coarse", action="store_true")
    parser.add_argument("--prior_eval_size", type=int, default=None)
    args, remaining = parser.parse_known_args()
    args.prior_detach_mask_features = not args.prior_joint_mask_features
    sys.argv = [sys.argv[0]] + remaining
    return args


def attach_prior_args(opt, prior_args):
    for key, value in vars(prior_args).items():
        setattr(opt, key, value)


def set_learning_rate(optimizers, lr):
    for optimizer in optimizers:
        print("[i] set learning rate to {}".format(lr))
        util.set_opt_param(optimizer, "lr", lr)


def build_train_loader(opt):
    datadir = "./datasets/processed_data"
    datadir_syn = join(datadir, "VOCdevkit/VOC2012/PNGImages")
    datadir_real = join(datadir, "real_train")

    train_dataset_syn = datasets.PriorCEILDataset(
            datadir_syn,
            read_fns("VOC2012_224_train_png.txt"),
            size=opt.max_dataset_size,
            enable_transforms=True,
            synthesis=opt.prior_train_synthesis,
            low_sigma=opt.low_sigma,
            high_sigma=opt.high_sigma,
            low_gamma=opt.low_gamma,
            high_gamma=opt.high_gamma,
            alpha_min=opt.realistic_alpha_min,
            alpha_max=opt.realistic_alpha_max,
            ghost_prob=opt.realistic_ghost_prob,
            max_ghost_shift=opt.realistic_max_ghost_shift,
            noise_std=opt.realistic_noise_std,
            jpeg_prob=opt.realistic_jpeg_prob)

    if opt.prior_real_ratio > 0 and os.path.isdir(datadir_real):
        train_dataset_real = datasets.PriorCEILTestDataset(
            datadir_real,
            size=opt.max_dataset_size,
            enable_transforms=True)
        real_ratio = min(max(opt.prior_real_ratio, 0.0), 1.0)
        train_dataset = datasets.FusionDataset(
            [train_dataset_syn, train_dataset_real],
            [1.0 - real_ratio, real_ratio])
    else:
        train_dataset = train_dataset_syn

    return datasets.DataLoader(
        train_dataset,
        batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=opt.nThreads,
        pin_memory=len(opt.gpu_ids) > 0)


def build_eval_loader(opt, dataset_key):
    dataset_name, rel_path, save_subdir, max_long_edge = EVAL_DATASETS[dataset_key]
    size = opt.prior_eval_size
    if opt.debug and size is None:
        size = 4
    dataset = datasets.PriorCEILTestDataset(
        join("./datasets/processed_data", rel_path),
        size=size,
        max_long_edge=max_long_edge)
    loader = datasets.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.nThreads,
        pin_memory=len(opt.gpu_ids) > 0)
    return dataset_name, save_subdir, loader


def train_one_epoch(model, loader, opt, writer=None, visualizer=None):
    avg_meters = util.AverageMeters()
    model._train()
    for i, data in enumerate(loader):
        iter_start_time = time.time()
        model.set_input(data, mode="train")
        model.optimize_parameters()
        errors = model.get_current_errors()
        avg_meters.update(errors)
        util.progress_bar(i, len(loader), str(avg_meters))

        if writer is not None:
            util.write_loss(writer, "train", avg_meters, model.iterations)

        if i % opt.print_freq == 0:
            t = time.time() - iter_start_time
            if visualizer is not None:
                visualizer.print_current_errors(model.epoch, i, errors, t)
            else:
                print("(epoch: %d, iters: %d, time: %.3f) %s" % (
                    model.epoch, i, t, " ".join("%s: %.3f" % (k, v) for k, v in errors.items())))

        model.iterations += 1

    loader.reset()
    return avg_meters


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
                metric_row = {"image": os.path.splitext(os.path.basename(image_name or str(i)))[0]}
                metric_row.update(metrics)
                metric_rows.append(metric_row)
            util.progress_bar(i, len(loader), str(avg_meters))
    if savedir is not None and metric_rows:
        util.save_eval_metrics(metric_rows, avg_meters, savedir, dataset_name)
    return avg_meters


def append_eval_history(path, rows):
    exists = os.path.exists(path)
    util.mkdirs(os.path.dirname(path))
    fieldnames = ["epoch", "dataset", "PSNR", "SSIM", "NCC", "LMSE"]
    with open(path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    prior_args = parse_prior_args()
    opt = TrainOptions().parse()
    attach_prior_args(opt, prior_args)
    if opt.name is None:
        opt.name = "errnet_prior_{}".format(opt.prior_base)

    if opt.debug:
        opt.nEpochs = min(opt.nEpochs, 2)
        opt.max_dataset_size = min(opt.max_dataset_size or 8, 8)
        opt.nThreads = 0
        opt.display_id = 0
        opt.print_freq = 1
        opt.prior_eval_freq = 1
        opt.prior_eval_size = opt.prior_eval_size or 2

    cudnn.benchmark = len(opt.gpu_ids) > 0

    writer = None
    visualizer = None
    if not opt.no_log:
        writer = util.get_summary_writer(join("checkpoints", opt.name, "logs"))
        visualizer = Visualizer(opt)
        visualizer.log_name = join(opt.checkpoints_dir, opt.name, "loss_log_prior.txt")

    train_loader = build_train_loader(opt)
    eval_keys = [key.strip() for key in opt.prior_eval_datasets.split(",") if key.strip()]

    model = ERRNetPriorBranchModel()
    model.initialize(opt)

    set_learning_rate(model.optimizers, opt.lr)
    best_score = -1e9
    history_path = join(opt.checkpoints_dir, opt.name, "prior_eval_history.csv")

    while model.epoch < opt.nEpochs:
        if model.epoch == max(1, int(opt.nEpochs * 0.5)):
            set_learning_rate(model.optimizers, opt.lr * 0.5)
        if model.epoch == max(1, int(opt.nEpochs * 0.75)):
            set_learning_rate(model.optimizers, opt.lr * 0.1)

        print("\nEpoch: %d" % model.epoch)
        train_metrics = train_one_epoch(model, train_loader, opt, writer, visualizer)
        print("\nEpoch %d train metrics: %s" % (model.epoch, train_metrics))
        model.epoch += 1

        if model.epoch % opt.save_epoch_freq == 0:
            print("saving the model at epoch %d" % model.epoch)
            model.save()
        print("saving the latest model at the end of epoch %d" % model.epoch)
        model.save(label="latest")

        if opt.prior_eval_freq > 0 and (model.epoch % opt.prior_eval_freq == 0 or model.epoch == opt.nEpochs):
            score_parts = []
            rows = []
            for key in eval_keys:
                dataset_name, save_subdir, loader = build_eval_loader(opt, key)
                savedir = join(opt.prior_result_dir, opt.name, "epoch_%03d" % model.epoch, save_subdir)
                print("[i] evaluating %s..." % key)
                metrics = evaluate(model, loader, dataset_name, savedir=savedir)
                row = {"epoch": model.epoch, "dataset": key}
                for metric_key in ["PSNR", "SSIM", "NCC", "LMSE"]:
                    row[metric_key] = metrics[metric_key] if metric_key in metrics.keys() else ""
                rows.append(row)
                if "PSNR" in metrics.keys():
                    score_parts.append(metrics["PSNR"])
            append_eval_history(history_path, rows)
            if score_parts:
                score = sum(score_parts) / float(len(score_parts))
                if score > best_score:
                    best_score = score
                    print("saving the best prior model at epoch %d, score %.6f" % (model.epoch, score))
                    model.save(label="best")

    print("Prior branch training finished.")


if __name__ == "__main__":
    main()
