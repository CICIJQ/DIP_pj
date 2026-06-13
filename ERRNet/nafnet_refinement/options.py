from options.errnet.train_options import TrainOptions


class NAFTrainOptions(TrainOptions):
    def initialize(self):
        super(NAFTrainOptions, self).initialize()
        parser = self.parser
        parser.add_argument(
            "--naf_coarse_kind",
            choices=["baseline", "improved"],
            default="improved",
            help="frozen ERRNet coarse-stage variant",
        )
        parser.add_argument("--naf_coarse_checkpoint", default=None)
        parser.add_argument("--naf_checkpoint", default=None)
        parser.add_argument("--naf_resume_optimizer", action="store_true")
        parser.add_argument(
            "--naf_no_coarse_hyper",
            dest="naf_coarse_hyper",
            action="store_false",
            help="disable VGG hypercolumns in the frozen coarse ERRNet",
        )
        parser.set_defaults(naf_coarse_hyper=True)
        parser.add_argument("--naf_width", type=int, default=32)
        parser.add_argument("--naf_middle_blk_num", type=int, default=4)
        parser.add_argument("--naf_enc_blk_nums", default="1,1,2")
        parser.add_argument("--naf_dec_blk_nums", default="1,1,1")
        parser.add_argument("--naf_delta_scale", type=float, default=1.0)
        parser.add_argument("--naf_lambda_l1", type=float, default=1.0)
        parser.add_argument("--naf_lambda_ssim", type=float, default=0.2)
        parser.add_argument("--naf_lambda_gradient", type=float, default=0.1)
        parser.add_argument("--naf_lambda_vgg", type=float, default=0.05)
        parser.add_argument("--naf_grad_clip", type=float, default=1.0)
        parser.add_argument("--naf_real_ratio", type=float, default=0.3)
        parser.add_argument(
            "--naf_data_root",
            default="./datasets/processed_data",
        )
        parser.add_argument("--naf_eval_freq", type=int, default=5)
        parser.add_argument(
            "--naf_eval_datasets",
            default="real20,ceilnet_table2",
        )
        parser.add_argument("--naf_eval_size", type=int, default=None)


class NAFTestOptions(TrainOptions):
    def initialize(self):
        super(NAFTestOptions, self).initialize()
        parser = self.parser
        parser.add_argument(
            "--dataset",
            default="all",
            choices=[
                "all",
                "ceilnet_table2",
                "objects",
                "postcard",
                "real20",
                "sir2_withgt",
                "wild",
            ],
        )
        parser.add_argument(
            "--data_root",
            default="./datasets/processed_data",
        )
        parser.add_argument(
            "--result_dir",
            default="./results/eval_naf_refiner",
        )
        parser.add_argument("--save_subdir", default=None)
        parser.add_argument("--max_long_edge", type=int, default=None)
        parser.add_argument("--eval_size", type=int, default=None)
        parser.add_argument(
            "--naf_checkpoint",
            default=(
                "checkpoints/errnet_naf_refiner_improved/"
                "naf_refiner_best.pt"
            ),
        )
        parser.add_argument(
            "--naf_coarse_kind",
            choices=["baseline", "improved"],
            default=None,
        )
        parser.add_argument("--naf_coarse_checkpoint", default=None)
        parser.add_argument(
            "--baseline_checkpoint",
            default="checkpoints/errnet/errnet_060_00463920.pt",
        )
        parser.add_argument(
            "--improved_checkpoint",
            default=(
                "checkpoints/errnet_improved_loss_v1/"
                "errnet_060_00463920.pt"
            ),
        )
        parser.add_argument("--tta", action="store_true")
        parser.add_argument("--reference_tta", action="store_true")
        self.isTrain = False
