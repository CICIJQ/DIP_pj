import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.losses import ContentLoss, GradientLoss, MultipleLoss, VGGLoss, CXLoss, DiscLoss, DiscLossR, DiscLossRa


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.register_buffer('window', create_window(window_size, self.channel))

    def forward(self, img1, img2):
        if img1.size() != img2.size():
            raise ValueError('Input tensors must have the same size')

        channel = img1.size(1)
        if channel != self.channel:
            window = create_window(self.window_size, channel).to(img1.device)
        else:
            window = self.window
            if window.device != img1.device:
                window = window.to(img1.device)

        return 1 - _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class SafeVGGLoss(VGGLoss):
    def forward(self, x, y):
        if self.normalize is not None and self.normalize.weight.device != x.device:
            self.normalize = self.normalize.to(x.device)
        if next(self.vgg.parameters()).device != x.device:
            self.vgg = self.vgg.to(x.device)
        return super(SafeVGGLoss, self).forward(x, y)

class SafeCXLoss(CXLoss):
    def forward(self, x, y):
        if self.normalize is not None and self.normalize.weight.device != x.device:
            self.normalize = self.normalize.to(x.device)
        if next(self.vgg.parameters()).device != x.device:
            self.vgg = self.vgg.to(x.device)
        return super(SafeCXLoss, self).forward(x, y)

class ImprovedPixelLoss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_grad=1.0, lambda_ssim=0.1):
        super(ImprovedPixelLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_grad = lambda_grad
        self.lambda_ssim = lambda_ssim
        self.l1 = nn.L1Loss()
        self.grad = GradientLoss()
        self.ssim = SSIMLoss()

    def forward(self, predict, target):
        loss = self.lambda_l1 * self.l1(predict, target)
        if self.lambda_grad != 0:
            loss = loss + self.lambda_grad * self.grad(predict, target)
        if self.lambda_ssim != 0:
            loss = loss + self.lambda_ssim * self.ssim(predict, target)
        return loss


def init_improved_loss(opt, tensor):
    loss_dic = {}

    pixel_loss = ContentLoss()
    pixel_loss.initialize(ImprovedPixelLoss(
        lambda_l1=getattr(opt, 'lambda_l1', 1.0),
        lambda_grad=getattr(opt, 'lambda_grad', 1.0),
        lambda_ssim=getattr(opt, 'lambda_ssim', 0.1)))
    loss_dic['t_pixel'] = pixel_loss
    loss_dic['r_pixel'] = pixel_loss

    ssim_loss = ContentLoss()
    ssim_loss.initialize(SSIMLoss())
    loss_dic['t_ssim'] = ssim_loss

    vgg_loss = ContentLoss()
    vgg_loss.initialize(SafeVGGLoss(weights=[1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10/1.5], indices=[2, 7, 12, 21, 30]))
    loss_dic['t_vgg'] = vgg_loss

    if getattr(opt, 'lambda_gan', 0) > 0:
        if opt.gan_type == 'sgan' or opt.gan_type == 'gan':
            disc_loss = DiscLoss()
        elif opt.gan_type == 'rsgan':
            disc_loss = DiscLossR()
        elif opt.gan_type == 'rasgan':
            disc_loss = DiscLossRa()
        else:
            raise ValueError("GAN [%s] not recognized." % opt.gan_type)

        disc_loss.initialize(opt, tensor)
        loss_dic['gan'] = disc_loss

    return loss_dic
