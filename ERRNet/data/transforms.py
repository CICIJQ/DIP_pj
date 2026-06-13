from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps, ImageEnhance
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import scipy.stats as st
import cv2
import numbers
import types
import collections
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import util.util as util
from scipy.signal import convolve2d


# utility
def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def arrshow(arr):
    Image.fromarray(arr.astype(np.uint8)).show()


def get_transform(opt):
    transform_list = []
    osizes = util.parse_args(opt.loadSize)
    fineSize = util.parse_args(opt.fineSize)
    if opt.resize_or_crop == 'resize_and_crop':    
        transform_list.append(
            transforms.RandomChoice([
                transforms.Resize([osize, osize], Image.BICUBIC) for osize in osizes
            ]))
        transform_list.append(transforms.RandomCrop(fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    return transforms.Compose(transform_list)


to_norm_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

to_tensor = transforms.ToTensor()


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    h = math.ceil(h / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)


# functional 
def gaussian_blur(img, kernel_size, sigma):
    from scipy.ndimage.filters import gaussian_filter
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    img = np.asarray(img)
    # the 3rd dimension (i.e. inter-band) would be filtered which is unwanted for our purpose
    # new = gaussian_filter(img, sigma=sigma, truncate=truncate)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    elif isinstance(kernel_size, collections.Sequence):
        assert len(kernel_size) == 2        
    new = cv2.GaussianBlur(img, kernel_size, sigma)  # apply gaussian filter band by band    
    return Image.fromarray(new)


# transforms
class GaussianBlur(object):
    def __init__(self, kernel_size=11, sigma=3):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        return gaussian_blur(img, self.kernel_size, self.sigma)


class ReflectionSythesis_1(object):
    """Reflection image data synthesis for weakly-supervised learning 
    of ICCV 2017 paper *"A Generic Deep Architecture for Single Image Reflection Removal and Image Smoothing"*    
    """
    def __init__(self, kernel_sizes=None, low_sigma=2, high_sigma=5, low_gamma=1.3, high_gamma=1.3):
        self.kernel_sizes = kernel_sizes or [11]
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma
        self.low_gamma = low_gamma
        self.high_gamma = high_gamma
        print('[i] reflection sythesis model: {}'.format({
            'kernel_sizes': kernel_sizes, 'low_sigma': low_sigma, 'high_sigma': high_sigma,
            'low_gamma': low_gamma, 'high_gamma': high_gamma}))

    def __call__(self, B, R):
        if not _is_pil_image(B):
            raise TypeError('B should be PIL Image. Got {}'.format(type(B)))
        if not _is_pil_image(R):
            raise TypeError('R should be PIL Image. Got {}'.format(type(R)))
        
        B_ = np.asarray(B, np.float32) / 255.
        R_ = np.asarray(R, np.float32) / 255.

        kernel_size = np.random.choice(self.kernel_sizes)
        sigma = np.random.uniform(self.low_sigma, self.high_sigma)
        gamma = np.random.uniform(self.low_gamma, self.high_gamma)
        R_blur = R_
        kernel = cv2.getGaussianKernel(11, sigma)
        kernel2d = np.dot(kernel, kernel.T)

        for i in range(3):
            R_blur[...,i] = convolve2d(R_blur[...,i], kernel2d, mode='same')

        M_ = B_ + R_blur
        
        if np.max(M_) > 1:
            m = M_[M_ > 1]
            m = (np.mean(m) - 1) * gamma
            R_blur = np.clip(R_blur - m, 0, 1)
            M_ = np.clip(R_blur + B_, 0, 1)
        
        return B_, R_blur, M_


class Sobel(object):
    def __call__(self, img):
        if not _is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        gray_img = np.array(img.convert('L'))
        x = cv2.Sobel(gray_img,cv2.CV_16S,1,0)
        y = cv2.Sobel(gray_img,cv2.CV_16S,0,1)
        
        absX = cv2.convertScaleAbs(x)   
        absY = cv2.convertScaleAbs(y)
        
        dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
        return Image.fromarray(dst)


class ReflectionSythesis_2(object):
    """Reflection image data synthesis for weakly-supervised learning 
    of CVPR 2018 paper *"Single Image Reflection Separation with Perceptual Losses"*
    """
    def __init__(self, kernel_sizes=None):
        self.kernel_sizes = kernel_sizes or np.linspace(1,5,80)
    
    @staticmethod
    def gkern(kernlen=100, nsig=1):
        """Returns a 2D Gaussian kernel array."""
        interval = (2*nsig+1.)/(kernlen)
        x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw/kernel_raw.sum()
        kernel = kernel/kernel.max()
        return kernel

    def __call__(self, t, r):        
        t = np.float32(t) / 255.
        r = np.float32(r) / 255.
        ori_t = t
        # create a vignetting mask
        g_mask=self.gkern(560,3)
        g_mask=np.dstack((g_mask,g_mask,g_mask))
        sigma=self.kernel_sizes[np.random.randint(0, len(self.kernel_sizes))]

        t=np.power(t,2.2)
        r=np.power(r,2.2)
        
        sz=int(2*np.ceil(2*sigma)+1)
        
        r_blur=cv2.GaussianBlur(r,(sz,sz),sigma,sigma,0)
        blend=r_blur+t
        
        att=1.08+np.random.random()/10.0
        
        for i in range(3):
            maski=blend[:,:,i]>1
            mean_i=max(1.,np.sum(blend[:,:,i]*maski)/(maski.sum()+1e-6))
            r_blur[:,:,i]=r_blur[:,:,i]-(mean_i-1)*att
        r_blur[r_blur>=1]=1
        r_blur[r_blur<=0]=0

        h,w=r_blur.shape[0:2]
        neww=np.random.randint(0, 560-w-10)
        newh=np.random.randint(0, 560-h-10)
        alpha1=g_mask[newh:newh+h,neww:neww+w,:]
        alpha2 = 1-np.random.random()/5.0
        r_blur_mask=np.multiply(r_blur,alpha1)
        blend=r_blur_mask+t*alpha2
        
        t=np.power(t,1/2.2)
        r_blur_mask=np.power(r_blur_mask,1/2.2)
        blend=np.power(blend,1/2.2)
        blend[blend>=1]=1
        blend[blend<=0]=0
        
        return np.float32(ori_t), np.float32(r_blur_mask), np.float32(blend)


class RealisticReflectionSynthesis(object):
    """Online synthesis with spatially varying reflection strength.

    This keeps the original ERRNet training interface but makes the synthetic
    mixture closer to hand-held glass photos: blurred/ghosted reflection,
    smooth alpha masks, gamma-domain blending, mild color shift and sensor
    noise.
    """
    def __init__(
            self,
            low_sigma=1.0,
            high_sigma=6.0,
            alpha_min=0.55,
            alpha_max=0.9,
            ghost_prob=0.6,
            max_ghost_shift=8,
            noise_std=0.01,
            jpeg_prob=0.25):
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.ghost_prob = ghost_prob
        self.max_ghost_shift = max_ghost_shift
        self.noise_std = noise_std
        self.jpeg_prob = jpeg_prob
        print('[i] realistic reflection synthesis model: {}'.format({
            'low_sigma': low_sigma,
            'high_sigma': high_sigma,
            'alpha_min': alpha_min,
            'alpha_max': alpha_max,
            'ghost_prob': ghost_prob,
            'max_ghost_shift': max_ghost_shift,
            'noise_std': noise_std,
            'jpeg_prob': jpeg_prob}))

    @staticmethod
    def _smooth_mask(h, w, low, high):
        small_h = max(2, int(np.ceil(h / 32.0)))
        small_w = max(2, int(np.ceil(w / 32.0)))
        mask = np.random.uniform(low, high, (small_h, small_w, 1)).astype(np.float32)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC)
        if mask.ndim == 2:
            mask = mask[..., None]
        sigma = max(h, w) / 12.0
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
        if mask.ndim == 2:
            mask = mask[..., None]
        return np.clip(mask, low, high).astype(np.float32)

    @staticmethod
    def _shift(img, dx, dy):
        h, w = img.shape[:2]
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    @staticmethod
    def _jpeg_roundtrip(img):
        quality = int(np.random.randint(70, 96))
        encoded = cv2.imencode('.jpg', cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1]
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return decoded

    def __call__(self, B, R):
        if not _is_pil_image(B):
            raise TypeError('B should be PIL Image. Got {}'.format(type(B)))
        if not _is_pil_image(R):
            raise TypeError('R should be PIL Image. Got {}'.format(type(R)))

        B_ = np.asarray(B, np.float32) / 255.0
        R_ = np.asarray(R, np.float32) / 255.0
        h, w = B_.shape[:2]

        sigma = np.random.uniform(self.low_sigma, self.high_sigma)
        kernel_size = int(2 * np.ceil(2 * sigma) + 1)
        R_blur = cv2.GaussianBlur(R_, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)

        if np.random.random() < self.ghost_prob:
            dx = np.random.randint(-self.max_ghost_shift, self.max_ghost_shift + 1)
            dy = np.random.randint(-self.max_ghost_shift, self.max_ghost_shift + 1)
            ghost = self._shift(R_blur, dx, dy)
            R_blur = np.clip(0.75 * R_blur + 0.25 * ghost, 0, 1)

        color_gain = np.random.uniform(0.85, 1.15, (1, 1, 3)).astype(np.float32)
        R_blur = np.clip(R_blur * color_gain, 0, 1)

        alpha_t = self._smooth_mask(h, w, self.alpha_min, self.alpha_max)
        alpha_r = self._smooth_mask(h, w, 0.12, 0.45)

        gamma = np.random.uniform(1.8, 2.4)
        B_lin = np.power(np.clip(B_, 0, 1), gamma)
        R_lin = np.power(np.clip(R_blur, 0, 1), gamma)
        M_lin = alpha_t * B_lin + alpha_r * R_lin
        M_ = np.power(np.clip(M_lin, 0, 1), 1.0 / gamma)

        exposure = np.random.uniform(0.95, 1.08)
        M_ = np.clip(M_ * exposure, 0, 1)

        if self.noise_std > 0:
            M_ = np.clip(M_ + np.random.normal(0, self.noise_std, M_.shape).astype(np.float32), 0, 1)

        if np.random.random() < self.jpeg_prob:
            M_ = self._jpeg_roundtrip(M_)

        R_target = np.clip(alpha_r * R_blur, 0, 1).astype(np.float32)
        return B_.astype(np.float32), R_target, M_.astype(np.float32)


# Examples
if __name__ == '__main__':
    """cv2 imread"""
    # img = cv2.imread('testdata_reflection_real/19-input.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img2 = cv2.GaussianBlur(img, (11,11), 3)    

    """Sobel Operator"""
    # img = np.array(Image.open('datasets/VOC224/train/B/2007_000250.png').convert('L'))


    """Reflection Sythesis"""
    b = Image.open('datasets/VOCsmall/train/B/2008_000148.png')
    r = Image.open('datasets/VOCsmall/train/B/2007_000243.png')
    G = ReflectionSythesis_1()
    m, r = G(b, r)
    r.show()
    
    # img2 = gaussian_blur(img, 11, 3)
    # img2 = GaussianBlur(1, 1)(img)
    # print(np.sum(np.array(img2) - np.array(img)))
    # img2.show()
