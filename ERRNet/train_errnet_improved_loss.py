from collections import OrderedDict
from os.path import join
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import util.util as util
from util.visualizer import Visualizer
from options.errnet.train_options import TrainOptions
from models.errnet_model import ERRNetModel
from models.improved_losses import init_improved_loss
import data.reflect_dataset as datasets
from data.image_folder import read_fns


def set_learning_rate(optimizers, lr):
    for optimizer in optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)


def train_one_epoch(model, dataloader, opt):
    avg_meters = util.AverageMeters()
    model._train()
    for i, data in enumerate(dataloader):
        iter_start_time = time.time()
        model.set_input(data, mode='train')
        model.forward()

        loss_pixel = model.loss_dic['t_pixel'].get_loss(model.output_i, model.target_t)
        loss_vgg = model.loss_dic['t_vgg'].get_loss(model.output_i, model.target_t)
        loss_ssim = model.loss_dic['t_ssim'].get_loss(model.output_i, model.target_t)

        loss_G = loss_pixel + opt.lambda_vgg * loss_vgg + opt.lambda_ssim * loss_ssim

        model.optimizer_G.zero_grad()
        loss_G.backward()
        model.optimizer_G.step()

        avg_meters.update({'Loss': loss_G.item(), 'IPixel': loss_pixel.item(), 'VGG': loss_vgg.item(), 'SSIM': loss_ssim.item()})
        util.progress_bar(i, len(dataloader), str(avg_meters))

        if i % opt.print_freq == 0:
            errors = OrderedDict([
                ('Loss', loss_G.item()),
                ('IPixel', loss_pixel.item()),
                ('VGG', loss_vgg.item()),
                ('SSIM', loss_ssim.item())
            ])
            t = time.time() - iter_start_time
            model.visualizer.print_current_errors(model.epoch, i, errors, t)
            sys.stdout.flush()

        if not opt.no_log and i % opt.display_freq == 0:
            util.write_loss(model.writer, 'train', avg_meters, model.iterations)
            if i % opt.update_html_freq == 0:
                model.visualizer.display_current_results(model.get_current_visuals(), model.epoch, True)

        model.iterations += 1

    return avg_meters


def evaluate(model, dataloader, dataset_name):
    avg_meters = util.AverageMeters()
    model._eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            index = model.eval(data, savedir=None)
            avg_meters.update(index)
            util.progress_bar(i, len(dataloader), str(avg_meters))
    return avg_meters


def main():
    opt = TrainOptions().parse()

    # independent improved-loss training branch
    opt.lambda_ssim = 0.1
    opt.lambda_grad = 1.0
    opt.lambda_l1 = 1.0
    opt.lambda_gan = 0

    cudnn.benchmark = True

    opt.display_freq = 10
    opt.print_freq = 100

    datadir = './datasets/processed_data'
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

    model = ERRNetModel()
    model.initialize(opt)
    model.loss_dic = init_improved_loss(opt, model.Tensor)

    if not opt.no_log:
        model.writer = util.get_summary_writer(join('checkpoints', opt.name, 'logs'))
        model.visualizer = Visualizer(opt)
        # use a dedicated improved-model loss log file, separate from baseline
        log_name = join(opt.checkpoints_dir, opt.name, 'loss_log_improved.txt')
        model.visualizer.log_name = log_name
        with open(log_name, 'a') as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    if opt.resume:
        if opt.resume_epoch is not None:
            model.load(model, opt.resume_epoch)

    set_learning_rate(model.optimizers, opt.lr)

    while model.epoch < opt.nEpochs:
        if model.epoch == 30:
            set_learning_rate(model.optimizers, 5e-5)
        if model.epoch == 45:
            set_learning_rate(model.optimizers, 1e-5)

        train_metrics = train_one_epoch(model, train_dataloader_fusion, opt)
        print('\nEpoch %d train metrics: %s' % (model.epoch, train_metrics))

        if model.epoch % 5 == 0:
            print('[i] evaluating CEILNet table2...')
            evaluate(model, eval_dataloader_ceilnet, 'testdata_table2')
            print('[i] evaluating real20...')
            evaluate(model, eval_dataloader_real, 'testdata_real20')

        model.epoch += 1

        if model.epoch % opt.save_epoch_freq == 0:
            print('saving the model at epoch %d' % model.epoch)
            model.save()

        print('saving the latest model at the end of epoch %d' % model.epoch)
        model.save(label='latest')

    print('Training finished.')


if __name__ == '__main__':
    main()
