import os
import sys
from os.path import join

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util


opt = TrainOptions().parse()
opt.inet = 'errnet_transformer'
opt.model = 'errnet_transformer_cascade_model'

if opt.name in (None, 'errnet_model', 'errnet_cascade_model'):
    opt.name = 'errnet_transformer_cascade_v2'

if opt.coarse_icnn_path is None:
    default_coarse = join(ROOT_DIR, 'checkpoints', 'errnet', 'errnet_latest.pt')
    if os.path.exists(default_coarse) and not opt.resume:
        opt.coarse_icnn_path = default_coarse

cudnn.benchmark = True

opt.display_freq = 10
opt.print_freq = 100

if opt.debug:
    opt.display_freq = 20
    opt.print_freq = 20
    opt.nEpochs = 2
    opt.max_dataset_size = 100
    opt.no_log = False
    opt.nThreads = 0
    opt.decay_iter = 0
    opt.serial_batches = True
    opt.no_flip = True

datadir = join(ROOT_DIR, 'datasets', 'processed_data')
datadir_syn = join(datadir, 'VOCdevkit', 'VOC2012', 'PNGImages')
datadir_real = join(datadir, 'real_train')

train_dataset = datasets.CEILDataset(
    datadir_syn,
    read_fns(join(ROOT_DIR, 'VOC2012_224_train_png.txt')),
    size=opt.max_dataset_size,
    enable_transforms=True,
    low_sigma=opt.low_sigma,
    high_sigma=opt.high_sigma,
    low_gamma=opt.low_gamma,
    high_gamma=opt.high_gamma)

train_dataset_real = datasets.CEILTestDataset(datadir_real, enable_transforms=True)
train_dataset_fusion = datasets.FusionDataset([train_dataset, train_dataset_real], [0.7, 0.3])

train_dataloader_fusion = datasets.DataLoader(
    train_dataset_fusion,
    batch_size=opt.batchSize,
    shuffle=not opt.serial_batches,
    num_workers=opt.nThreads,
    pin_memory=True)

eval_dataset_ceilnet = datasets.CEILTestDataset(join(datadir, 'testdata_CEILNET_table2'))
eval_dataset_real = datasets.CEILTestDataset(join(datadir, 'real20'), size=20, max_long_edge=512)

eval_dataloader_ceilnet = datasets.DataLoader(
    eval_dataset_ceilnet,
    batch_size=1,
    shuffle=False,
    num_workers=opt.nThreads,
    pin_memory=True)
eval_dataloader_real = datasets.DataLoader(
    eval_dataset_real,
    batch_size=1,
    shuffle=False,
    num_workers=opt.nThreads,
    pin_memory=True)

engine = Engine(opt)
assert engine.model is not None


def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)


save_root = join(ROOT_DIR, 'results', 'eval_transformer_cascade_v2')

engine.model.opt.lambda_gan = 0
set_learning_rate(1e-4)

while engine.epoch < opt.nEpochs:
    if engine.epoch == 20:
        engine.model.opt.lambda_gan = 0.01
    if engine.epoch == 30:
        set_learning_rate(5e-5)
    if engine.epoch == 40:
        set_learning_rate(1e-5)
    if engine.epoch == 45:
        ratio = [0.5, 0.5]
        print('[i] adjust fusion ratio to {}'.format(ratio))
        train_dataset_fusion.fusion_ratios = ratio
        set_learning_rate(5e-5)
    if engine.epoch == 50:
        set_learning_rate(1e-5)

    engine.train(train_dataloader_fusion)

    if engine.epoch % 5 == 0:
        engine.eval(
            eval_dataloader_ceilnet,
            dataset_name='transformer_cascade_v2_table2',
            savedir=join(save_root, 'table2'))
        engine.eval(
            eval_dataloader_real,
            dataset_name='transformer_cascade_v2_real20',
            savedir=join(save_root, 'real20'))
