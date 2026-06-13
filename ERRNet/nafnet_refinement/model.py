import copy
import os
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

import util.index as metric_index
from models.arch.nafnet_refiner import NAFNetRefiner
from models.errnet_model import ERRNetModel, _torch_load_compat, tensor2im
from models.improved_losses import SSIMLoss, SafeVGGLoss
from models.losses import GradientLoss


DEFAULT_COARSE_CHECKPOINTS = {
    "baseline": "checkpoints/errnet/errnet_060_00463920.pt",
    "improved": "checkpoints/errnet_improved_loss_v1/errnet_060_00463920.pt",
}


def parse_int_list(value):
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def align_tensors(*tensors):
    present = [tensor for tensor in tensors if isinstance(tensor, torch.Tensor)]
    height = min(tensor.shape[-2] for tensor in present)
    width = min(tensor.shape[-1] for tensor in present)
    return tuple(
        tensor[..., :height, :width] if isinstance(tensor, torch.Tensor) else tensor
        for tensor in tensors
    )


class ERRNetNAFRefinerModel(object):
    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.input = None
        self.target_t = None
        self.coarse_i = None
        self.output_i = None
        self.data_name = None

    def initialize(self, opt, training=True):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device(
            "cuda:%d" % self.gpu_ids[0] if self.gpu_ids else "cpu"
        )
        self.training_mode = training

        checkpoint_state = None
        naf_checkpoint = getattr(opt, "naf_checkpoint", None)
        if naf_checkpoint:
            checkpoint_state = _torch_load_compat(
                naf_checkpoint,
                map_location=torch.device("cpu"),
            )
        checkpoint_config = (
            checkpoint_state.get("config", {})
            if isinstance(checkpoint_state, dict)
            else {}
        )

        self.coarse_kind = getattr(opt, "naf_coarse_kind", None)
        if not self.coarse_kind:
            self.coarse_kind = checkpoint_config.get("coarse_kind", "improved")
        coarse_checkpoint = getattr(opt, "naf_coarse_checkpoint", None)
        if not coarse_checkpoint:
            coarse_checkpoint = checkpoint_config.get("coarse_checkpoint")
        if not coarse_checkpoint:
            coarse_checkpoint = DEFAULT_COARSE_CHECKPOINTS[self.coarse_kind]
        if not os.path.isfile(coarse_checkpoint):
            raise FileNotFoundError(
                "NAF coarse checkpoint not found: %s" % coarse_checkpoint
            )
        self.coarse_checkpoint = coarse_checkpoint
        self.coarse_hyper = bool(
            checkpoint_config.get(
                "coarse_hyper",
                getattr(opt, "naf_coarse_hyper", True),
            )
        )

        width = int(checkpoint_config.get("width", getattr(opt, "naf_width", 32)))
        middle_blk_num = int(
            checkpoint_config.get(
                "middle_blk_num",
                getattr(opt, "naf_middle_blk_num", 4),
            )
        )
        enc_blk_nums = parse_int_list(
            checkpoint_config.get(
                "enc_blk_nums",
                getattr(opt, "naf_enc_blk_nums", "1,1,2"),
            )
        )
        dec_blk_nums = parse_int_list(
            checkpoint_config.get(
                "dec_blk_nums",
                getattr(opt, "naf_dec_blk_nums", "1,1,1"),
            )
        )
        delta_scale = float(
            checkpoint_config.get(
                "delta_scale",
                getattr(opt, "naf_delta_scale", 1.0),
            )
        )
        self.config = {
            "coarse_kind": self.coarse_kind,
            "coarse_checkpoint": self.coarse_checkpoint,
            "coarse_hyper": self.coarse_hyper,
            "width": width,
            "middle_blk_num": middle_blk_num,
            "enc_blk_nums": list(enc_blk_nums),
            "dec_blk_nums": list(dec_blk_nums),
            "delta_scale": delta_scale,
        }

        coarse_opt = copy.deepcopy(opt)
        coarse_opt.isTrain = False
        coarse_opt.resume = True
        coarse_opt.icnn_path = coarse_checkpoint
        coarse_opt.hyper = self.coarse_hyper
        coarse_opt.inet = "errnet"
        coarse_opt.name = "naf_coarse_%s" % self.coarse_kind
        coarse_opt.no_log = True
        coarse_opt.no_verbose = True
        self.coarse_model = ERRNetModel()
        self.coarse_model.initialize(coarse_opt)
        self.coarse_model._eval()
        for parameter in self.coarse_model.net_i.parameters():
            parameter.requires_grad = False
        if self.coarse_model.vgg is not None:
            self.coarse_model.vgg.eval()
            for parameter in self.coarse_model.vgg.parameters():
                parameter.requires_grad = False

        self.refiner = NAFNetRefiner(
            width=width,
            middle_blk_num=middle_blk_num,
            enc_blk_nums=enc_blk_nums,
            dec_blk_nums=dec_blk_nums,
            delta_scale=delta_scale,
        ).to(self.device)

        if checkpoint_state is not None:
            state = checkpoint_state.get("naf_refiner", checkpoint_state)
            self.refiner.load_state_dict(state, strict=True)
            self.epoch = int(checkpoint_state.get("epoch", 0))
            self.iterations = int(checkpoint_state.get("iterations", 0))
            print(
                "[i] loaded NAF refiner from %s, epoch %d, iteration %d"
                % (naf_checkpoint, self.epoch, self.iterations)
            )

        self.optimizer = None
        self.optimizers = []
        if training:
            self.optimizer = torch.optim.AdamW(
                self.refiner.parameters(),
                lr=opt.lr,
                betas=(0.9, 0.999),
                weight_decay=opt.wd,
            )
            self.optimizers = [self.optimizer]
            if (
                checkpoint_state is not None
                and getattr(opt, "naf_resume_optimizer", False)
                and "optimizer" in checkpoint_state
            ):
                self.optimizer.load_state_dict(checkpoint_state["optimizer"])

            self.l1_loss = nn.L1Loss()
            self.ssim_loss = SSIMLoss().to(self.device)
            self.gradient_loss = GradientLoss().to(self.device)
            self.vgg_loss = SafeVGGLoss(
                vgg=self.coarse_model.vgg,
                weights=[1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10 / 1.5],
                indices=[2, 7, 12, 21, 30],
            ).to(self.device)

    def set_input(self, data, mode="train"):
        input_image = data["input"]
        target = data.get("target_t")
        if self.gpu_ids:
            input_image = input_image.to(self.device, non_blocking=True)
            if isinstance(target, torch.Tensor):
                target = target.to(self.device, non_blocking=True)
        self.input = input_image
        self.target_t = target if isinstance(target, torch.Tensor) else None
        self.data_name = data.get("fn")

    def _coarse_forward(self, input_image):
        with torch.no_grad():
            coarse = self.coarse_model._forward_input(input_image)
        input_image, coarse = align_tensors(input_image, coarse)
        return input_image, coarse

    def _forward_pair(self, input_image):
        input_image, coarse = self._coarse_forward(input_image)
        output = self.refiner(input_image, coarse)
        return input_image, coarse, output

    def forward(self):
        aligned_input, coarse, output = self._forward_pair(self.input)
        self.aligned_input = aligned_input
        self.coarse_i = coarse
        self.output_i = output
        if self.target_t is not None:
            self.aligned_input, self.coarse_i, self.output_i, self.target_t = (
                align_tensors(
                    self.aligned_input,
                    self.coarse_i,
                    self.output_i,
                    self.target_t,
                )
            )
        return self.output_i

    def forward_tta(self):
        inputs = []
        coarses = []
        outputs = []
        for dims in (None, (3,), (2,), (2, 3)):
            augmented = self.input if dims is None else torch.flip(self.input, dims=dims)
            aligned_input, coarse, output = self._forward_pair(augmented)
            if dims is not None:
                aligned_input = torch.flip(aligned_input, dims=dims)
                coarse = torch.flip(coarse, dims=dims)
                output = torch.flip(output, dims=dims)
            inputs.append(aligned_input)
            coarses.append(coarse)
            outputs.append(output)

        height = min(output.shape[-2] for output in outputs)
        width = min(output.shape[-1] for output in outputs)
        self.aligned_input = torch.stack(
            [item[..., :height, :width] for item in inputs],
            dim=0,
        ).mean(dim=0)
        self.coarse_i = torch.stack(
            [item[..., :height, :width] for item in coarses],
            dim=0,
        ).mean(dim=0)
        self.output_i = torch.stack(
            [item[..., :height, :width] for item in outputs],
            dim=0,
        ).mean(dim=0)
        if self.target_t is not None:
            self.aligned_input, self.coarse_i, self.output_i, self.target_t = (
                align_tensors(
                    self.aligned_input,
                    self.coarse_i,
                    self.output_i,
                    self.target_t,
                )
            )
        return self.output_i

    def optimize_parameters(self):
        self._train()
        self.forward()
        self.loss_l1 = self.l1_loss(self.output_i, self.target_t)
        self.loss_ssim = self.ssim_loss(self.output_i, self.target_t)
        self.loss_gradient = self.gradient_loss(self.output_i, self.target_t)
        self.loss_vgg = self.vgg_loss(self.output_i, self.target_t)
        self.loss_total = (
            self.loss_l1 * self.opt.naf_lambda_l1
            + self.loss_ssim * self.opt.naf_lambda_ssim
            + self.loss_gradient * self.opt.naf_lambda_gradient
            + self.loss_vgg * self.opt.naf_lambda_vgg
        )
        self.optimizer.zero_grad()
        self.loss_total.backward()
        if getattr(self.opt, "naf_grad_clip", 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.refiner.parameters(),
                self.opt.naf_grad_clip,
            )
        self.optimizer.step()

    def get_current_errors(self):
        return OrderedDict(
            [
                ("Loss", self.loss_total.item()),
                ("L1", self.loss_l1.item()),
                ("SSIM", self.loss_ssim.item()),
                ("Grad", self.loss_gradient.item()),
                ("VGG", self.loss_vgg.item()),
            ]
        )

    def get_current_visuals(self):
        visuals = OrderedDict()
        visuals["input"] = tensor2im(self.aligned_input).astype(np.uint8)
        visuals["coarse"] = tensor2im(self.coarse_i).astype(np.uint8)
        visuals["naf_refiner"] = tensor2im(self.output_i).astype(np.uint8)
        if self.target_t is not None:
            visuals["target"] = tensor2im(self.target_t).astype(np.uint8)
        return visuals

    def quality_assess(self):
        output = tensor2im(self.output_i)
        target = tensor2im(self.target_t)
        height = min(output.shape[0], target.shape[0])
        width = min(output.shape[1], target.shape[1])
        return metric_index.quality_assess(
            output[:height, :width],
            target[:height, :width],
        )

    def _train(self):
        self.coarse_model._eval()
        self.refiner.train()

    def _eval(self):
        self.coarse_model._eval()
        self.refiner.eval()

    def state_dict(self):
        state = {
            "naf_refiner": self.refiner.state_dict(),
            "epoch": self.epoch,
            "iterations": self.iterations,
            "config": self.config,
        }
        if self.optimizer is not None:
            state["optimizer"] = self.optimizer.state_dict()
        return state

    def save(self, path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), path)
