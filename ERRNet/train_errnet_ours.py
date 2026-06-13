from os.path import join

import torch.backends.cudnn as cudnn

import data.reflect_dataset as datasets
import util.util as util
from data.image_folder import read_fns
from engine import Engine
from options.errnet.train_options import TrainOptions


def set_learning_rate(model, lr):
    for optimizer in model.optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)


def build_eval_loader(datadir, opt, dataset_name, size=None, max_long_edge=None):
    dataset = datasets.CEILTestDataset(
        join(datadir, dataset_name),
        size=size,
        max_long_edge=max_long_edge)
    return datasets.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.nThreads,
        pin_memory=True)


def main():
    opt = TrainOptions().parse()
    opt.model = 'errnet_cascade_model'
    opt.inet = 'errnet_cascade'
    if opt.name in (None, 'errnet_model', 'errnet_cascade_model'):
        opt.name = 'errnet_ours_realistic_cascade'
    opt.nEpochs = opt.ours_stage1_epochs + opt.ours_stage2_epochs

    cudnn.benchmark = len(opt.gpu_ids) > 0
    opt.display_freq = 10

    if opt.debug:
        opt.display_id = 1
        opt.display_freq = 20
        opt.print_freq = 20
        opt.nEpochs = 2
        opt.ours_stage1_epochs = 1
        opt.ours_stage2_epochs = 1
        opt.max_dataset_size = 2
        opt.no_log = True
        opt.nThreads = 0
        opt.serial_batches = True
        opt.no_flip = True

    datadir = './datasets/processed_data'
    datadir_syn = join(datadir, 'VOCdevkit/VOC2012/PNGImages')
    datadir_real = join(datadir, 'real_train')

    train_dataset_syn = datasets.RealisticCEILDataset(
        datadir_syn,
        read_fns('VOC2012_224_train_png.txt'),
        size=opt.max_dataset_size,
        enable_transforms=True,
        low_sigma=opt.low_sigma,
        high_sigma=opt.high_sigma,
        alpha_min=opt.realistic_alpha_min,
        alpha_max=opt.realistic_alpha_max,
        ghost_prob=opt.realistic_ghost_prob,
        max_ghost_shift=opt.realistic_max_ghost_shift,
        noise_std=opt.realistic_noise_std,
        jpeg_prob=opt.realistic_jpeg_prob)

    train_dataset_real = datasets.CEILTestDataset(
        datadir_real,
        size=opt.max_dataset_size if opt.debug else None,
        enable_transforms=True)
    real_ratio = min(max(opt.ours_real_ratio, 0.0), 1.0)
    train_dataset_stage2 = datasets.FusionDataset(
        [train_dataset_syn, train_dataset_real],
        [1.0 - real_ratio, real_ratio])

    train_loader_stage1 = datasets.DataLoader(
        train_dataset_syn,
        batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=opt.nThreads,
        pin_memory=True)
    train_loader_stage2 = datasets.DataLoader(
        train_dataset_stage2,
        batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=opt.nThreads,
        pin_memory=True)

    debug_eval_size = 2 if opt.debug else None
    eval_loader_ceilnet = build_eval_loader(datadir, opt, 'testdata_CEILNET_table2', size=debug_eval_size)
    eval_loader_real20 = build_eval_loader(datadir, opt, 'real20', size=debug_eval_size or 20, max_long_edge=512)
    eval_loader_sir2 = build_eval_loader(datadir, opt, 'sir2_withgt', size=debug_eval_size)

    engine = Engine(opt)
    assert engine.model is not None

    set_learning_rate(engine.model, opt.lr)
    engine.model.opt.lambda_gan = 0

    while engine.epoch < opt.nEpochs:
        if engine.epoch < opt.ours_stage1_epochs:
            stage = 'stage1 synthetic realistic'
            train_loader = train_loader_stage1
            if engine.epoch == 20:
                engine.model.opt.lambda_gan = opt.lambda_gan
        else:
            stage = 'stage2 synthetic+real finetune'
            train_loader = train_loader_stage2
            if engine.epoch == opt.ours_stage1_epochs:
                engine.model.opt.lambda_gan = 0
                set_learning_rate(engine.model, opt.lr * 0.2)
            if engine.epoch == opt.ours_stage1_epochs + max(1, opt.ours_stage2_epochs // 2):
                set_learning_rate(engine.model, opt.lr * 0.05)

        print('[i] {}'.format(stage))
        engine.train(train_loader)

        if engine.epoch % 5 == 0 or engine.epoch == opt.nEpochs:
            engine.eval(eval_loader_ceilnet, dataset_name='testdata_table2')
            engine.eval(eval_loader_real20, dataset_name='testdata_real20')
            engine.eval(eval_loader_sir2, dataset_name='testdata_sir2')

    print('Improved training finished.')


if __name__ == '__main__':
    main()
