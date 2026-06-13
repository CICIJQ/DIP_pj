from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)        
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')        
        self.parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training (Note: in train_errnet.py, we mannually tune the training protocol, but you can also use following setting by modifying the code in errnet_model.py)
        self.parser.add_argument('--nEpochs', '-n', type=int, default=60, help='# of epochs to run')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        self.parser.add_argument('--wd', type=float, default=0, help='weight decay for adam')

        self.parser.add_argument('--low_sigma', type=float, default=2, help='min sigma in synthetic dataset')
        self.parser.add_argument('--high_sigma', type=float, default=5, help='max sigma in synthetic dataset')
        self.parser.add_argument('--low_gamma', type=float, default=1.3, help='max gamma in synthetic dataset')
        self.parser.add_argument('--high_gamma', type=float, default=1.3, help='max gamma in synthetic dataset')
        
        # data augmentation
        self.parser.add_argument('--batchSize', '-b', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=str, default='224,336,448', help='scale images to multiple size')
        self.parser.add_argument('--fineSize', type=str, default='224,224', help='then crop to this size')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')

        # for discriminator
        self.parser.add_argument('--which_model_D', type=str, default='disc_vgg', choices=['disc_vgg', 'disc_patch'])
        self.parser.add_argument('--gan_type', type=str, default='rasgan', help='gan/sgan : Vanilla GAN; rasgan : relativistic gan')
        
        # loss weight
        self.parser.add_argument('--unaligned_loss', type=str, default='vgg', help='learning rate policy: vgg|mse|ctx|ctx_vgg')
        self.parser.add_argument('--vgg_layer', type=int, default=31, help='vgg layer of unaligned loss')
        
        self.parser.add_argument('--lambda_gan', type=float, default=0.01, help='weight for gan loss')
        self.parser.add_argument('--lambda_vgg', type=float, default=0.1, help='weight for vgg loss')
        self.parser.add_argument('--lambda_coarse', type=float, default=0.2, help='weight for the coarse output supervision in cascade models')

        # options used by train_errnet_ours.py; harmless for baseline scripts
        self.parser.add_argument('--ours_stage1_epochs', type=int, default=40, help='synthetic-only warmup epochs for the improved method')
        self.parser.add_argument('--ours_stage2_epochs', type=int, default=20, help='synthetic+real finetuning epochs for the improved method')
        self.parser.add_argument('--ours_real_ratio', type=float, default=0.6, help='real-image sampling ratio in stage 2 of the improved method')
        self.parser.add_argument('--coarse_icnn_path', type=str, default=None, help='optional baseline ERRNet checkpoint used to initialize the cascade coarse net')
        self.parser.add_argument('--realistic_alpha_min', type=float, default=0.55, help='minimum transmission alpha for realistic synthesis')
        self.parser.add_argument('--realistic_alpha_max', type=float, default=0.9, help='maximum transmission alpha for realistic synthesis')
        self.parser.add_argument('--realistic_ghost_prob', type=float, default=0.6, help='probability of adding ghosted reflection')
        self.parser.add_argument('--realistic_max_ghost_shift', type=int, default=8, help='maximum ghost shift in pixels')
        self.parser.add_argument('--realistic_noise_std', type=float, default=0.01, help='Gaussian noise std for realistic synthesis')
        self.parser.add_argument('--realistic_jpeg_prob', type=float, default=0.25, help='probability of JPEG degradation for realistic synthesis')
        
        self.isTrain = True
