import copy
import os
from os.path import join

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw

import util.index as metric_index
import util.util as util
from models.errnet_model import ERRNetModel, tensor2im
from nafnet_refinement.common import (
    EVAL_DATASETS,
    build_eval_loader,
    image_name,
)
from nafnet_refinement.model import ERRNetNAFRefinerModel
from nafnet_refinement.options import NAFTestOptions


def _initialize_reference(base_opt, checkpoint, alias):
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError("%s checkpoint not found: %s" % (alias, checkpoint))
    opt = copy.deepcopy(base_opt)
    opt.isTrain = False
    opt.resume = True
    opt.icnn_path = checkpoint
    opt.inet = "errnet"
    opt.hyper = True
    opt.name = "naf_reference_%s" % alias
    opt.no_log = True
    opt.no_verbose = True
    model = ERRNetModel()
    model.initialize(opt)
    model._eval()
    return model


def _save_tensor(tensor, path):
    Image.fromarray(tensor2im(tensor).astype(np.uint8)).save(path)


def _assess(output_tensor, target_tensor):
    output = tensor2im(output_tensor)
    target = tensor2im(target_tensor)
    height = min(output.shape[0], target.shape[0])
    width = min(output.shape[1], target.shape[1])
    return metric_index.quality_assess(
        output[:height, :width],
        target[:height, :width],
    )


def run_reference(
    base_opt,
    loader,
    checkpoint,
    alias,
    output_dir,
    use_tta=False,
):
    model = _initialize_reference(base_opt, checkpoint, alias)
    rows = []
    with torch.no_grad():
        for index, data in enumerate(loader):
            model.set_input(data, mode="eval")
            output = model.forward_tta() if use_tta else model.forward()
            name = image_name(data, index)
            image_dir = join(output_dir, name)
            util.mkdirs(image_dir)
            _save_tensor(output, join(image_dir, alias + ".png"))
            row = {"image": name}
            row.update(_assess(output, model.target_t))
            rows.append(row)
            util.progress_bar(
                index,
                len(loader),
                "%s%s" % (alias, " + TTA" if use_tta else ""),
            )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def make_comparison(image_dir):
    panel_specs = [
        ("input", "input.png"),
        ("baseline", "baseline.png"),
        ("improved loss", "improved.png"),
        ("naf refiner", "naf_refiner.png"),
        ("ground truth", "gt.png"),
    ]
    panels = []
    for title, filename in panel_specs:
        image = Image.open(join(image_dir, filename)).convert("RGB")
        image.thumbnail((384, 384), Image.BICUBIC)
        panels.append((title, image.copy()))

    label_height = 24
    canvas = Image.new(
        "RGB",
        (
            sum(image.width for _, image in panels),
            max(image.height for _, image in panels) + label_height,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    offset = 0
    for title, image in panels:
        draw.text((offset + 4, 4), title, fill="black")
        canvas.paste(image, (offset, label_height))
        offset += image.width
    canvas.save(join(image_dir, "comparison.png"))


def average_rows(rows):
    averages = {}
    for key in ("PSNR", "SSIM", "NCC", "LMSE"):
        averages[key] = float(np.mean([float(row[key]) for row in rows]))
    return averages


def evaluate_naf(model, loader, output_dir, use_tta=False):
    rows = []
    model._eval()
    with torch.no_grad():
        for index, data in enumerate(loader):
            model.set_input(data, mode="eval")
            output = model.forward_tta() if use_tta else model.forward()
            metrics = model.quality_assess()
            name = image_name(data, index)
            image_dir = join(output_dir, name)
            util.mkdirs(image_dir)
            _save_tensor(model.aligned_input, join(image_dir, "input.png"))
            _save_tensor(model.aligned_input, join(image_dir, "m_input.png"))
            _save_tensor(model.coarse_i, join(image_dir, "coarse.png"))
            _save_tensor(output, join(image_dir, "naf_refiner.png"))
            _save_tensor(model.target_t, join(image_dir, "gt.png"))
            _save_tensor(model.target_t, join(image_dir, "t_label.png"))
            make_comparison(image_dir)
            row = {"image": name}
            row.update(metrics)
            rows.append(row)
            util.progress_bar(
                index,
                len(loader),
                "naf refiner%s" % (" + TTA" if use_tta else ""),
            )
    return rows


def save_metrics(rows, output_dir, dataset_name):
    meters = util.AverageMeters()
    for row in rows:
        meters.update(
            {
                key: row[key]
                for key in ("PSNR", "SSIM", "NCC", "LMSE")
            }
        )
    util.save_eval_metrics(rows, meters, output_dir, dataset_name)


def append_reference_summary(
    output_dir,
    model,
    baseline_rows,
    improved_rows,
    test_args,
):
    with open(join(output_dir, "summary.txt"), "a") as summary:
        summary.write("\nNAF refinement\n")
        summary.write("Checkpoint: %s\n" % test_args.naf_checkpoint)
        summary.write("Coarse kind: %s\n" % model.coarse_kind)
        summary.write("Coarse checkpoint: %s\n" % model.coarse_checkpoint)
        summary.write("TTA: %s\n" % bool(test_args.tta))
        for alias, rows in (
            ("baseline", baseline_rows),
            ("improved", improved_rows),
        ):
            averages = average_rows(rows)
            summary.write(
                "%s %s\n"
                % (
                    alias,
                    " ".join(
                        "%s %.6f" % (key, averages[key])
                        for key in ("PSNR", "SSIM", "NCC", "LMSE")
                    ),
                )
            )


def evaluate_dataset(base_opt, test_args, dataset_key):
    spec, loader = build_eval_loader(
        base_opt,
        test_args.data_root,
        dataset_key,
        size=test_args.eval_size,
        max_long_edge=test_args.max_long_edge,
    )
    save_subdir = (
        test_args.save_subdir
        if test_args.save_subdir and test_args.dataset != "all"
        else spec["save_subdir"]
    )
    output_dir = join(test_args.result_dir, save_subdir)
    util.mkdirs(output_dir)

    print("[i] %s: baseline reference" % dataset_key)
    baseline_rows = run_reference(
        base_opt,
        loader,
        test_args.baseline_checkpoint,
        "baseline",
        output_dir,
        use_tta=test_args.reference_tta,
    )
    print("[i] %s: improved-loss reference" % dataset_key)
    improved_rows = run_reference(
        base_opt,
        loader,
        test_args.improved_checkpoint,
        "improved",
        output_dir,
        use_tta=test_args.reference_tta,
    )

    naf_opt = copy.deepcopy(base_opt)
    naf_opt.naf_checkpoint = test_args.naf_checkpoint
    naf_opt.naf_coarse_kind = test_args.naf_coarse_kind
    naf_opt.naf_coarse_checkpoint = test_args.naf_coarse_checkpoint
    naf_opt.naf_coarse_hyper = True
    model = ERRNetNAFRefinerModel()
    model.initialize(naf_opt, training=False)

    print("[i] %s: NAF refinement" % dataset_key)
    rows = evaluate_naf(model, loader, output_dir, use_tta=test_args.tta)
    save_metrics(rows, output_dir, spec["dataset_name"])
    append_reference_summary(
        output_dir,
        model,
        baseline_rows,
        improved_rows,
        test_args,
    )
    print("[i] %s NAF metrics: %s" % (dataset_key, average_rows(rows)))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    opt = NAFTestOptions().parse()
    opt.isTrain = False
    opt.no_log = True
    opt.no_verbose = True
    opt.display_id = 0
    if not os.path.isfile(opt.naf_checkpoint):
        raise FileNotFoundError(
            "NAF checkpoint not found: %s" % opt.naf_checkpoint
        )

    cudnn.benchmark = len(opt.gpu_ids) > 0

    dataset_keys = (
        sorted(EVAL_DATASETS)
        if opt.dataset == "all"
        else [opt.dataset]
    )
    for dataset_key in dataset_keys:
        evaluate_dataset(opt, opt, dataset_key)


if __name__ == "__main__":
    main()
