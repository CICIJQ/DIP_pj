import csv
import os
from os.path import join

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
from PIL import Image

import data.sirs_dataset as datasets
import util.index as index
import util.util as util
from engine import Engine
from options.net_options.train_options import TrainOptions
from models.dsrnet_model_sirs import tensor2im


DATASETS = {
    "real20": ("real20", "testdata_real"),
    "ceilnet_table2": ("testdata_CEILNET_table2", "testdata_table2"),
    "sir2_withgt": ("sir2_withgt", "testdata_sir2"),
    "objects": ("objects", "testdata_objects"),
    "postcard": ("postcard", "testdata_postcard"),
    "wild": ("wild", "testdata_wild"),
}


class EvalErrnetProcessedOptions(TrainOptions):
    def initialize(self):
        super().initialize()
        self.parser.add_argument(
            "--eval_datasets",
            type=str,
            default="real20,ceilnet_table2,sir2_withgt,objects,postcard,wild",
            help="Comma-separated dataset keys from: %s" % ",".join(DATASETS.keys()),
        )
        self.parser.add_argument(
            "--result_dir",
            type=str,
            default="./results/dsrnet_s_epoch14_errnet_processed",
            help="Directory for per-dataset summaries.",
        )
        self.parser.add_argument(
            "--max_eval_size",
            type=int,
            default=-1,
            help="Limit images per dataset for smoke tests. Use -1 for full evaluation.",
        )
        self.parser.add_argument(
            "--save_images",
            action="store_true",
            help="Save visual outputs. Disabled by default to keep full evaluation compact.",
        )
        self.parser.add_argument(
            "--max_long_edge",
            type=int,
            default=-1,
            help="Resize images whose long edge exceeds this value before evaluation. Use -1 for original size.",
        )
        self.parser.add_argument(
            "--split_intro_devices",
            type=str,
            default="",
            help="Experimental inference-only model split, e.g. '0,1' puts VGG+intro on cuda:0 and the remaining network on cuda:1.",
        )
        self.parser.add_argument(
            "--amp",
            action="store_true",
            help="Use CUDA automatic mixed precision during inference to reduce memory.",
        )
        self.parser.add_argument(
            "--tile_size",
            type=int,
            default=0,
            help="Use full-resolution tiled inference with this tile size. 0 disables tiling.",
        )
        self.parser.add_argument(
            "--tile_overlap",
            type=int,
            default=64,
            help="Overlap for tiled inference.",
        )


class ResizedDSRTestDataset(datasets.DSRTestDataset):
    def __init__(self, *args, max_long_edge=-1, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_long_edge = max_long_edge

    def resize_max_long_edge(self, x1, x2):
        if self.max_long_edge <= 0:
            return x1, x2

        w, h = x1.size
        long_edge = max(w, h)
        if long_edge <= self.max_long_edge:
            return x1, x2

        scale = float(self.max_long_edge) / float(long_edge)
        new_w = max(32, int(round(w * scale)))
        new_h = max(32, int(round(h * scale)))
        return x1.resize((new_w, new_h), Image.BICUBIC), x2.resize((new_w, new_h), Image.BICUBIC)

    def __getitem__(self, index):
        fn = self.fns[index]

        t_img = Image.open(join(self.datadir, "transmission_layer", fn)).convert("RGB")
        m_img = Image.open(join(self.datadir, "blended", fn)).convert("RGB")
        t_img, m_img = self.resize_max_long_edge(t_img, m_img)

        if self.if_align:
            t_img, m_img = self.align(t_img, m_img)

        if self.enable_transforms:
            t_img, m_img = datasets.paired_data_transforms(t_img, m_img, self.unaligned_transforms)

        B = TF.to_tensor(t_img)
        M = TF.to_tensor(m_img)

        dic = {"input": M, "target_t": B, "fn": fn, "real": True, "target_r": M - B}
        if self.flag is not None:
            dic.update(self.flag)
        return dic


def list_filenames(datadir):
    blended_dir = join(datadir, "blended")
    return sorted(
        fn
        for fn in os.listdir(blended_dir)
        if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )


def average(metrics):
    keys = ["PSNR", "SSIM", "NCC", "LMSE"]
    return {key: sum(row[key] for row in metrics) / len(metrics) for key in keys}


def write_summary(savedir, dataset_name, image_count, avg, opt):
    summary_path = join(savedir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Images: {image_count}\n\n")
        f.write("Averages\n")
        f.write(f"PSNR {avg['PSNR']:.6f}\n")
        f.write(f"SSIM {avg['SSIM']:.6f}\n")
        f.write(f"NCC {avg['NCC']:.6f}\n")
        f.write(f"LMSE {avg['LMSE']:.6f}\n\n")
        f.write("DSRNet\n")
        f.write(f"Checkpoint: {opt.weight_path}\n")
        f.write(f"Architecture: {opt.inet}\n")
        f.write(f"Hypercolumn: {bool(opt.hyper)}\n")
        f.write(f"Align 32x: {bool(opt.if_align)}\n")
        f.write(f"Max long edge: {opt.max_long_edge}\n")
        f.write(f"Split intro devices: {opt.split_intro_devices}\n")
        f.write(f"AMP: {bool(opt.amp)}\n")
        f.write(f"Tile size: {opt.tile_size}\n")
        f.write(f"Tile overlap: {opt.tile_overlap}\n")
        f.write(f"Save images: {bool(opt.save_images)}\n")
    return summary_path


def tile_starts(size, tile_size, overlap):
    if tile_size <= 0 or size <= tile_size:
        return [0]
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("tile_overlap must be smaller than tile_size")
    starts = list(range(0, size - tile_size + 1, stride))
    last = size - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def infer_tiled(model, data, opt):
    input_tensor = data["input"]
    target_t = data["target_t"]
    _, _, height, width = input_tensor.shape
    tile_size = opt.tile_size
    overlap = opt.tile_overlap

    output_sum = torch.zeros_like(input_tensor)
    weight_sum = torch.zeros((1, 1, height, width), dtype=input_tensor.dtype)

    y_starts = tile_starts(height, tile_size, overlap)
    x_starts = tile_starts(width, tile_size, overlap)

    for y0 in y_starts:
        y1 = min(y0 + tile_size, height)
        for x0 in x_starts:
            x1 = min(x0 + tile_size, width)
            tile_input = input_tensor[:, :, y0:y1, x0:x1]
            tile_target = target_t[:, :, y0:y1, x0:x1]
            tile_data = {
                "input": tile_input,
                "target_t": tile_target,
                "target_r": tile_input - tile_target,
                "fn": data["fn"],
                "real": True,
            }

            model.set_input(tile_data, "eval")
            model.forward()
            output_tile = model.output_t.detach().float().cpu()

            output_sum[:, :, y0:y1, x0:x1] += output_tile
            weight_sum[:, :, y0:y1, x0:x1] += 1

            del output_tile
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return output_sum / weight_sum.clamp_min(1e-6)


def evaluate_tiled(model, data, opt, savedir=None, suffix=None):
    model._eval()
    output_t_tensor = infer_tiled(model, data, opt)

    output_t = tensor2im(output_t_tensor)
    target = tensor2im(data["target_t"])
    metrics = index.quality_assess(output_t, target)

    if savedir is not None:
        name = os.path.splitext(os.path.basename(data["fn"][0]))[0]
        out_dir = join(savedir, suffix, name)
        os.makedirs(out_dir, exist_ok=True)
        Image.fromarray(output_t.astype("uint8")).save(join(out_dir, "%s_t.png" % opt.name))
        Image.fromarray(target.astype("uint8")).save(join(out_dir, "t_label.png"))
        Image.fromarray(tensor2im(data["input"]).astype("uint8")).save(join(out_dir, "m_input.png"))

    return metrics


def evaluate_dataset(engine, dataset_key, datadir, dataset_name, opt):
    fns = list_filenames(datadir)
    size = None if opt.max_eval_size < 0 else opt.max_eval_size
    if size is not None:
        fns = fns[:size]

    dataset = ResizedDSRTestDataset(datadir, fns=fns, if_align=opt.if_align, max_long_edge=opt.max_long_edge)
    dataloader = datasets.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.nThreads,
        pin_memory=len(opt.gpu_ids) > 0,
    )

    savedir = join(opt.result_dir, dataset_key)
    os.makedirs(savedir, exist_ok=True)

    image_savedir = join(savedir, "images") if opt.save_images else None
    rows = []

    model = engine.model
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(opt.amp and len(opt.gpu_ids) > 0)):
                if opt.tile_size > 0:
                    metrics = evaluate_tiled(model, data, opt, savedir=image_savedir, suffix=dataset_key)
                else:
                    metrics = model.eval(
                        data,
                        savedir=image_savedir,
                        suffix=dataset_key,
                    )
            row = {"fn": data["fn"][0]}
            row.update({key: float(metrics[key]) for key in ["PSNR", "SSIM", "NCC", "LMSE"]})
            rows.append(row)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            util.progress_bar(
                i,
                len(dataloader),
                " | ".join("%s: %.4f" % (key, average(rows)[key]) for key in ["PSNR", "SSIM", "NCC", "LMSE"]),
            )

    per_image_path = join(savedir, "per_image_metrics.csv")
    with open(per_image_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fn", "PSNR", "SSIM", "NCC", "LMSE"])
        writer.writeheader()
        writer.writerows(rows)

    avg = average(rows)
    summary_path = write_summary(savedir, dataset_name, len(rows), avg, opt)
    print("[i] saved per-image metrics to %s" % per_image_path)
    print("[i] saved metric summary to %s" % summary_path)
    print("[i] %s DSRNet metrics: %s" % (dataset_key, avg))
    return dataset_key, dataset_name, len(rows), avg


def main():
    opt = EvalErrnetProcessedOptions().parse()
    opt.isTrain = False
    opt.no_log = True
    opt.display_id = 0
    opt.verbose = False
    cudnn.benchmark = True

    selected = [name.strip() for name in opt.eval_datasets.split(",") if name.strip()]
    unknown = [name for name in selected if name not in DATASETS]
    if unknown:
        raise ValueError("Unknown dataset keys: %s" % ", ".join(unknown))

    engine = Engine(opt)
    os.makedirs(opt.result_dir, exist_ok=True)

    all_rows = []
    for dataset_key in selected:
        rel_dir, dataset_name = DATASETS[dataset_key]
        datadir = join(opt.base_dir, rel_dir)
        if not os.path.isdir(join(datadir, "blended")) or not os.path.isdir(join(datadir, "transmission_layer")):
            raise FileNotFoundError("Missing blended/transmission_layer under %s" % datadir)
        all_rows.append(evaluate_dataset(engine, dataset_key, datadir, dataset_name, opt))

    if all_rows:
        keys = ["PSNR", "SSIM", "NCC", "LMSE"]
        mean = {key: sum(row[3][key] for row in all_rows) / len(all_rows) for key in keys}
        summary_path = join(opt.result_dir, "summary_all.txt")
        with open(summary_path, "w") as f:
            f.write("| Dataset | Images | PSNR | SSIM | NCC | LMSE |\n")
            f.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
            for dataset_key, _, image_count, avg in all_rows:
                f.write(
                    "| %s | %d | %.6f | %.6f | %.6f | %.6f |\n"
                    % (dataset_key, image_count, avg["PSNR"], avg["SSIM"], avg["NCC"], avg["LMSE"])
                )
            f.write(
                "| 平均 | - | %.6f | %.6f | %.6f | %.6f |\n"
                % (mean["PSNR"], mean["SSIM"], mean["NCC"], mean["LMSE"])
            )
        print("[i] saved aggregate summary to %s" % summary_path)


if __name__ == "__main__":
    main()
