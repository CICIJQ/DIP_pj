import os
import time
from os.path import join

import torch
import torch.backends.cudnn as cudnn

import data.reflect_dataset as datasets
import util.util as util
from data.image_folder import read_fns
from nafnet_refinement.common import build_eval_loader
from nafnet_refinement.model import ERRNetNAFRefinerModel
from nafnet_refinement.options import NAFTrainOptions
from util.visualizer import Visualizer


def build_train_loader(opt):
    data_root = opt.naf_data_root
    synthetic_root = join(data_root, "VOCdevkit", "VOC2012", "PNGImages")
    real_root = join(data_root, "real_train")
    synthetic = datasets.CEILDataset(
        synthetic_root,
        read_fns("VOC2012_224_train_png.txt"),
        size=opt.max_dataset_size,
        enable_transforms=True,
        low_sigma=opt.low_sigma,
        high_sigma=opt.high_sigma,
        low_gamma=opt.low_gamma,
        high_gamma=opt.high_gamma,
    )

    if opt.naf_real_ratio > 0 and os.path.isdir(real_root):
        real = datasets.CEILTestDataset(
            real_root,
            size=opt.max_dataset_size if opt.debug else None,
            enable_transforms=True,
        )
        real_ratio = min(max(opt.naf_real_ratio, 0.0), 1.0)
        training_dataset = datasets.FusionDataset(
            [synthetic, real],
            [1.0 - real_ratio, real_ratio],
        )
    else:
        training_dataset = synthetic

    return datasets.DataLoader(
        training_dataset,
        batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=opt.nThreads,
        pin_memory=len(opt.gpu_ids) > 0,
    )


def train_one_epoch(model, loader, opt, writer=None, visualizer=None):
    meters = util.AverageMeters()
    model._train()
    for index, data in enumerate(loader):
        start_time = time.time()
        model.set_input(data, mode="train")
        model.optimize_parameters()
        errors = model.get_current_errors()
        meters.update(errors)
        util.progress_bar(index, len(loader), str(meters))

        if writer is not None:
            util.write_loss(writer, "train", meters, model.iterations)
        if index % opt.print_freq == 0:
            if visualizer is not None:
                visualizer.print_current_errors(
                    model.epoch,
                    index,
                    errors,
                    time.time() - start_time,
                )
            else:
                print(
                    "(epoch: %d, iters: %d) %s"
                    % (
                        model.epoch,
                        index,
                        " ".join(
                            "%s: %.4f" % (key, value)
                            for key, value in errors.items()
                        ),
                    )
                )
        if (
            visualizer is not None
            and opt.display_id != 0
            and index % opt.display_freq == 0
        ):
            visualizer.display_current_results(
                model.get_current_visuals(),
                model.epoch,
                True,
            )
        model.iterations += 1

    loader.reset()
    return meters


def evaluate(model, loader):
    meters = util.AverageMeters()
    model._eval()
    with torch.no_grad():
        for index, data in enumerate(loader):
            model.set_input(data, mode="eval")
            model.forward()
            metrics = model.quality_assess()
            meters.update(metrics)
            util.progress_bar(index, len(loader), str(meters))
    return meters


def set_learning_rate(model, learning_rate):
    for optimizer in model.optimizers:
        util.set_opt_param(optimizer, "lr", learning_rate)
    print("[i] NAF refiner learning rate: %.8f" % learning_rate)


def main():
    opt = NAFTrainOptions().parse()
    opt.lambda_gan = 0
    if opt.name in (None, "errnet_model"):
        opt.name = "errnet_naf_refiner_%s" % opt.naf_coarse_kind

    if opt.debug:
        opt.nEpochs = min(opt.nEpochs, 2)
        opt.max_dataset_size = min(opt.max_dataset_size or 8, 8)
        opt.nThreads = 0
        opt.print_freq = 1
        opt.display_id = 0
        opt.naf_eval_freq = 1
        opt.naf_eval_size = opt.naf_eval_size or 2

    cudnn.benchmark = len(opt.gpu_ids) > 0
    checkpoint_dir = join(opt.checkpoints_dir, opt.name)
    util.mkdirs(checkpoint_dir)

    writer = None
    visualizer = None
    if not opt.no_log:
        writer = util.get_summary_writer(join(checkpoint_dir, "logs"))
        visualizer = Visualizer(opt)
        visualizer.log_name = join(checkpoint_dir, "loss_log_naf_refiner.txt")

    train_loader = build_train_loader(opt)
    eval_keys = [
        key.strip()
        for key in opt.naf_eval_datasets.split(",")
        if key.strip()
    ]
    eval_loaders = {}
    for key in eval_keys:
        size = opt.naf_eval_size
        spec, loader = build_eval_loader(
            opt,
            opt.naf_data_root,
            key,
            size=size,
        )
        eval_loaders[key] = (spec, loader)

    model = ERRNetNAFRefinerModel()
    model.initialize(opt, training=True)
    best_score = float("-inf")
    base_lr = opt.lr

    while model.epoch < opt.nEpochs:
        if model.epoch == max(1, int(opt.nEpochs * 0.5)):
            set_learning_rate(model, base_lr * 0.5)
        if model.epoch == max(1, int(opt.nEpochs * 0.75)):
            set_learning_rate(model, base_lr * 0.1)

        print("\nEpoch: %d" % model.epoch)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            opt,
            writer=writer,
            visualizer=visualizer,
        )
        print("\nEpoch %d train metrics: %s" % (model.epoch, train_metrics))
        model.epoch += 1

        model.save(join(checkpoint_dir, "naf_refiner_latest.pt"))
        if model.epoch % opt.save_epoch_freq == 0:
            model.save(
                join(
                    checkpoint_dir,
                    "naf_refiner_%03d_%08d.pt"
                    % (model.epoch, model.iterations),
                )
            )

        should_eval = (
            opt.naf_eval_freq > 0
            and (
                model.epoch % opt.naf_eval_freq == 0
                or model.epoch == opt.nEpochs
            )
        )
        if should_eval:
            psnrs = []
            for key, (spec, loader) in eval_loaders.items():
                print("[i] evaluating %s..." % key)
                metrics = evaluate(model, loader)
                print("[i] %s: %s" % (key, metrics))
                if "PSNR" in metrics.keys():
                    psnrs.append(metrics["PSNR"])
            if psnrs:
                score = sum(psnrs) / len(psnrs)
                if score > best_score:
                    best_score = score
                    print("[i] saving best NAF refiner, PSNR score %.6f" % score)
                    model.save(join(checkpoint_dir, "naf_refiner_best.pt"))

    print("NAF refiner training finished.")


if __name__ == "__main__":
    main()
