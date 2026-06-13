import os
from os.path import join

import data.sirs_dataset as datasets
import util.util as util
from data.image_folder import read_fns
from engine import Engine
from options.net_options.train_options import TrainOptions
from tools import mutils


DATASET_DIRS = {
    "real20": "real20",
    "ceilnet_table2": "testdata_CEILNET_table2",
    "sir2_withgt": "sir2_withgt",
    "objects": "objects",
    "postcard": "postcard",
    "wild": "wild",
}


def build_loader(dataset, opt, batch_size=1, shuffle=False):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "pin_memory": len(opt.gpu_ids) > 0,
        "num_workers": opt.nThreads,
    }
    if opt.nThreads > 0:
        kwargs["prefetch_factor"] = 4
    return datasets.DataLoader(dataset, **kwargs)


def build_eval_loaders(base_dir, opt):
    names = [name.strip() for name in opt.errnet_eval_datasets.split(",") if name.strip()]
    loaders = []
    for name in names:
        if name not in DATASET_DIRS:
            raise ValueError("unknown eval dataset '%s', choices: %s" % (name, ",".join(DATASET_DIRS)))
        dataset_dir = join(base_dir, DATASET_DIRS[name])
        dataset = datasets.DSRTestDataset(dataset_dir, if_align=opt.if_align)
        loaders.append((name, build_loader(dataset, opt, batch_size=1, shuffle=False)))
    return loaders


def set_learning_rate(engine, lr):
    for optimizer in engine.model.optimizers:
        print("[i] set learning rate to {}".format(lr))
        util.set_opt_param(optimizer, "lr", lr)


opt_parser = TrainOptions()
opt_parser.initialize()
opt_parser.parser.add_argument(
    "--errnet_real_ratio",
    type=float,
    default=0.3,
    help="sampling ratio for ERRNet real_train pairs in the synthetic/real fusion dataset",
)
opt_parser.parser.add_argument(
    "--errnet_eval_datasets",
    type=str,
    default="real20,ceilnet_table2,sir2_withgt,objects,postcard,wild",
    help="comma-separated processed_data validation datasets",
)
opt_parser.parser.add_argument(
    "--errnet_eval_save_size",
    type=int,
    default=5,
    help="number of visual samples to save per eval dataset",
)
opt = opt_parser.parse()

if opt.errnet_real_ratio < 0 or opt.errnet_real_ratio > 1:
    raise ValueError("--errnet_real_ratio must be in [0, 1]")

print(opt)

base_dir = os.path.abspath(opt.base_dir)
syn_dir = join(base_dir, "VOCdevkit", "VOC2012", "PNGImages")
real_dir = join(base_dir, "real_train")

if not os.path.isdir(syn_dir):
    raise FileNotFoundError("missing synthetic image dir: %s" % syn_dir)
if not os.path.isdir(join(real_dir, "blended")):
    raise FileNotFoundError("missing real_train blended dir: %s" % join(real_dir, "blended"))
if not os.path.isdir(join(real_dir, "transmission_layer")):
    raise FileNotFoundError("missing real_train transmission dir: %s" % join(real_dir, "transmission_layer"))

train_dataset_syn = datasets.DSRDataset(
    syn_dir,
    read_fns("data/VOC2012_224_train_png.txt"),
    size=opt.max_dataset_size,
    enable_transforms=True,
)
train_dataset_real = datasets.DSRTestDataset(real_dir, enable_transforms=True, if_align=opt.if_align)
train_dataset = datasets.FusionDataset(
    [train_dataset_syn, train_dataset_real],
    [1.0 - opt.errnet_real_ratio, opt.errnet_real_ratio],
    size=opt.num_train if opt.num_train > 0 else None,
)
train_loader = build_loader(train_dataset, opt, batch_size=opt.batchSize, shuffle=not opt.serial_batches)
eval_loaders = build_eval_loaders(base_dir, opt)

engine = Engine(opt)
result_dir = os.path.join("./checkpoints", opt.name, "results", mutils.get_formatted_time())

set_learning_rate(engine, opt.lr)

if opt.resume or opt.debug_eval:
    save_dir = os.path.join(result_dir, "%03d" % engine.epoch)
    os.makedirs(save_dir, exist_ok=True)
    engine.save_model()
    for dataset_name, loader in eval_loaders:
        engine.eval(
            loader,
            dataset_name="eval_%s" % dataset_name,
            savedir=os.path.join(save_dir, dataset_name),
            max_save_size=opt.errnet_eval_save_size,
        )

while engine.epoch < opt.nEpochs:
    print("random_seed: ", opt.seed)
    engine.train(train_loader)

    if opt.eval_freq > 0 and engine.epoch % opt.eval_freq == 0:
        save_dir = os.path.join(result_dir, "%03d" % engine.epoch)
        os.makedirs(save_dir, exist_ok=True)
        for dataset_name, loader in eval_loaders:
            engine.eval(
                loader,
                dataset_name="eval_%s" % dataset_name,
                savedir=os.path.join(save_dir, dataset_name),
                max_save_size=opt.errnet_eval_save_size,
            )
