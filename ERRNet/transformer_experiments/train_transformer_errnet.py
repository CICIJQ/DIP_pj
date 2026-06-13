import os
import sys
from os.path import join

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util

opt = TrainOptions().parse()
opt.inet = 'errnet_transformer'
opt.model = 'errnet_model'

if opt.name in (None, 'errnet_model', 'errnet_cascade_model'):
    opt.name = 'errnet_transformer'

cudnn.benchmark = True

opt.display_freq = 10
opt.print_freq = 100

if opt.debug:
    opt.display_freq = 20
    opt.print_freq = 20
    opt.nEpochs = 10
    opt.max_dataset_size = 100
    opt.no_log = False
    opt.nThreads = 0
    opt.decay_iter = 0
    opt.serial_batches = True
    opt.no_flip = True

# processed datasets prepared by datasets/prepare_train_data.py and datasets/prepare_test_data.py
root = os.path.dirname(os.path.dirname(__file__))
datadir = join(root, 'datasets/processed_data')

datadir_syn = join(datadir, 'VOCdevkit/VOC2012/PNGImages')
datadir_real = join(datadir, 'real_train')

train_dataset = datasets.CEILDataset(
    datadir_syn, read_fns('VOC2012_224_train_png.txt'), size=opt.max_dataset_size, enable_transforms=True,
    low_sigma=opt.low_sigma, high_sigma=opt.high_sigma,
    low_gamma=opt.low_gamma, high_gamma=opt.high_gamma)

train_dataset_real = datasets.CEILTestDataset(datadir_real, enable_transforms=True)
train_dataset_fusion = datasets.FusionDataset([train_dataset, train_dataset_real], [0.7, 0.3])

train_dataloader_fusion = datasets.DataLoader(
    train_dataset_fusion, batch_size=opt.batchSize, shuffle=not opt.serial_batches,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataset_ceilnet = datasets.CEILTestDataset(join(datadir, 'testdata_CEILNET_table2'))
eval_dataset_real = datasets.CEILTestDataset(join(datadir, 'real20'), size=20, max_long_edge=512)

eval_dataloader_ceilnet = datasets.DataLoader(
    eval_dataset_ceilnet, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)
eval_dataloader_real = datasets.DataLoader(
    eval_dataset_real, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

engine = Engine(opt)
assert engine.model is not None

engine.model.opt.lambda_gan = 0
set_lr = 1e-4
for optimizer in engine.model.optimizers:
    util.set_opt_param(optimizer, 'lr', set_lr)

save_root = join(root, 'results', 'eval_transformer')

while engine.epoch < 60:
    if engine.epoch == 2:
        set_lr = 5e-5
        for optimizer in engine.model.optimizers:
            util.set_opt_param(optimizer, 'lr', set_lr)

    engine.train(train_dataloader_fusion)

    if engine.epoch % 2 == 0:
        engine.eval(eval_dataloader_ceilnet, dataset_name='transformer_testdata_table2', savedir=join(save_root, 'table2'))
        engine.eval(eval_dataloader_real, dataset_name='transformer_testdata_real20', savedir=join(save_root, 'real20'))
