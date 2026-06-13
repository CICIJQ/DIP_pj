import itertools
import os
from collections import OrderedDict
from os.path import join

import numpy as np
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F

import models.networks as networks
import util.index as index
import util.util as util
from models.errnet_model import ERRNetModel, _torch_load_compat, tensor2im


class FixedHighFrequency(nn.Module):
    def __init__(self):
        super(FixedHighFrequency, self).__init__()
        laplacian = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32).view(1, 1, 3, 3)
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32).view(1, 1, 3, 3) / 4.0
        sobel_y = sobel_x.transpose(2, 3)
        self.register_buffer("laplacian", laplacian)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _gray(self, x):
        if x.size(1) == 1:
            return x
        weights = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (x * weights).sum(dim=1, keepdim=True)

    def forward(self, x):
        gray = self._gray(x)
        lap = F.conv2d(gray, self.laplacian.to(dtype=x.dtype), padding=1).abs()
        gx = F.conv2d(gray, self.sobel_x.to(dtype=x.dtype), padding=1)
        gy = F.conv2d(gray, self.sobel_y.to(dtype=x.dtype), padding=1)
        grad = torch.sqrt(gx * gx + gy * gy + 1e-8)
        return lap, grad


class ReflectionPriorHead(nn.Module):
    def __init__(self, in_channels=16, channels=32):
        super(ReflectionPriorHead, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class RefineBlock(nn.Module):
    def __init__(self, channels):
        super(RefineBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x) * 0.1


class PriorGatedRefiner(nn.Module):
    def __init__(self, in_channels=12, channels=64, blocks=4):
        super(PriorGatedRefiner, self).__init__()
        body = [
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.ReLU(True),
        ]
        body.extend([RefineBlock(channels) for _ in range(blocks)])
        self.body = nn.Sequential(*body)
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        return self.tail(self.body(x))


def _tv_loss(mask):
    loss = 0
    if mask.size(2) > 1:
        loss = loss + (mask[:, :, 1:, :] - mask[:, :, :-1, :]).abs().mean()
    if mask.size(3) > 1:
        loss = loss + (mask[:, :, :, 1:] - mask[:, :, :, :-1]).abs().mean()
    return loss


def _safe_bce(pred, target):
    pred = pred.clamp(1e-4, 1.0 - 1e-4)
    return F.binary_cross_entropy(pred, target)


class ERRNetPriorBranchModel(ERRNetModel):
    def name(self):
        return "errnet_prior"

    def initialize(self, opt):
        self.prior_base = getattr(opt, "prior_base", "errnet")
        self.prior_freeze_base = getattr(opt, "prior_freeze_base", False)
        if self.prior_base == "cascade":
            opt.inet = "errnet_cascade"
        else:
            opt.inet = "errnet"

        original_resume = opt.resume
        original_icnn_path = opt.icnn_path
        original_no_verbose = opt.no_verbose
        opt.resume = False
        opt.icnn_path = None
        opt.no_verbose = True
        super(ERRNetPriorBranchModel, self).initialize(opt)
        opt.resume = original_resume
        opt.icnn_path = original_icnn_path
        opt.no_verbose = original_no_verbose

        self.high_frequency = FixedHighFrequency().to(self.device)
        self.prior_head = ReflectionPriorHead(
            in_channels=14,
            channels=getattr(opt, "prior_head_feats", 32)).to(self.device)
        self.refine_net = PriorGatedRefiner(
            in_channels=12,
            channels=getattr(opt, "prior_refine_feats", 64),
            blocks=getattr(opt, "prior_refine_blocks", 4)).to(self.device)
        networks.init_weights(self.prior_head, init_type=getattr(opt, "prior_init_type", "kaiming"))
        networks.init_weights(self.refine_net.body, init_type=getattr(opt, "prior_init_type", "kaiming"))

        self.prior_mask = None
        self.prior_gate = None
        self.prior_target = None
        self.target_r = None
        self.coarse_i = None
        self.prior_delta_i = None
        self.loss_prior_mask = None
        self.loss_prior_gate = None
        self.loss_prior_smooth = None
        self.loss_prior_sparse = None
        self.loss_prior_identity = None
        self.loss_coarse_pixel = None

        if self.isTrain:
            if self.prior_freeze_base:
                for param in self.net_i.parameters():
                    param.requires_grad = False
                params = itertools.chain(self.prior_head.parameters(), self.refine_net.parameters())
            else:
                params = itertools.chain(
                    self.net_i.parameters(),
                    self.prior_head.parameters(),
                    self.refine_net.parameters())
            self.optimizer_G = torch.optim.Adam(
                params, lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)

            optimizers = []
            if getattr(opt, "lambda_gan", 0) > 0:
                optimizers.append(self.optimizer_D)
            optimizers.append(self.optimizer_G)
            self._init_optimizer(optimizers)

        if original_resume or (not self.isTrain and original_icnn_path is not None):
            self.load_prior(original_icnn_path, resume_epoch=getattr(opt, "resume_epoch", None))
        else:
            init_path = getattr(opt, "prior_init_icnn", None)
            if init_path:
                self.load_base_icnn(init_path)

        if opt.no_verbose is False:
            self.print_network()

    def set_input(self, data, mode="train"):
        super(ERRNetPriorBranchModel, self).set_input(data, mode)
        target_r = None
        if mode.lower() in ("train", "eval") and isinstance(data, dict):
            target_r = data.get("target_r", None)
        if isinstance(target_r, torch.Tensor):
            if len(self.gpu_ids) > 0:
                target_r = target_r.to(device=self.gpu_ids[0])
            self.target_r = target_r
        else:
            self.target_r = None

    def print_network(self):
        print("--------------------- Prior Branch Model ---------------------")
        print("##################### Base Net #####################")
        networks.print_network(self.net_i)
        print("##################### Prior Head #####################")
        networks.print_network(self.prior_head)
        print("##################### Gated Refiner #####################")
        networks.print_network(self.refine_net)
        if self.isTrain and getattr(self.opt, "lambda_gan", 0) > 0:
            print("##################### NetD #####################")
            networks.print_network(self.netD)

    def _eval(self):
        self.net_i.eval()
        self.prior_head.eval()
        self.refine_net.eval()

    def _train(self):
        if self.prior_freeze_base:
            self.net_i.eval()
        else:
            self.net_i.train()
        self.prior_head.train()
        self.refine_net.train()

    def _make_prior_features(self, input_tensor, coarse):
        residual = input_tensor - coarse
        abs_residual = residual.abs()
        lap, grad = self.high_frequency(input_tensor)
        head_coarse = coarse
        head_residual = residual
        head_abs_residual = abs_residual
        if getattr(self.opt, "prior_detach_mask_features", True):
            head_coarse = head_coarse.detach()
            head_residual = head_residual.detach()
            head_abs_residual = head_abs_residual.detach()
        prior_features = torch.cat([
            input_tensor,
            head_coarse,
            head_residual,
            head_abs_residual,
            lap,
            grad,
        ], dim=1)
        refine_features = torch.cat([
            input_tensor,
            coarse,
            residual,
            lap,
            grad,
        ], dim=1)
        return prior_features, refine_features, lap, grad

    def _forward_full(self, input_tensor):
        coarse = self._forward_input(input_tensor)
        if coarse.shape[2:] != input_tensor.shape[2:]:
            coarse = F.interpolate(
                coarse,
                size=input_tensor.shape[2:],
                mode="bilinear",
                align_corners=False)
        prior_features, refine_features, _, _ = self._make_prior_features(input_tensor, coarse)
        mask = self.prior_head(prior_features)
        delta = self.refine_net(torch.cat([refine_features, mask], dim=1))
        delta_scale = getattr(self.opt, "prior_delta_scale", 1.0)
        gate = self._calibrate_gate(mask)
        gated_delta = gate * delta * delta_scale
        output = coarse + gated_delta
        return output, mask, gate, coarse, gated_delta

    def _calibrate_gate(self, mask):
        threshold = getattr(self.opt, "prior_gate_threshold", 0.0)
        if threshold > 0:
            threshold = min(max(threshold, 0.0), 0.99)
            gate = ((mask - threshold) / (1.0 - threshold)).clamp(0.0, 1.0)
        else:
            gate = mask
        gamma = getattr(self.opt, "prior_gate_gamma", 1.0)
        if gamma != 1.0:
            gate = gate.clamp_min(1e-6).pow(gamma)
        return gate

    def forward(self):
        output_i, prior_mask, prior_gate, coarse_i, prior_delta_i = self._forward_full(self.input)
        self.output_i = output_i
        self.prior_mask = prior_mask
        self.prior_gate = prior_gate
        self.coarse_i = coarse_i
        self.prior_delta_i = prior_delta_i
        return output_i

    def forward_tta(self):
        outputs = []
        masks = []
        gates = []
        coarses = []
        deltas = []
        for dims in (None, (3,), (2,), (2, 3)):
            input_i = self.input if dims is None else torch.flip(self.input, dims=dims)
            output_i, mask_i, gate_i, coarse_i, delta_i = self._forward_full(input_i)
            if dims is not None:
                output_i = torch.flip(output_i, dims=dims)
                mask_i = torch.flip(mask_i, dims=dims)
                gate_i = torch.flip(gate_i, dims=dims)
                coarse_i = torch.flip(coarse_i, dims=dims)
                delta_i = torch.flip(delta_i, dims=dims)
            outputs.append(output_i)
            masks.append(mask_i)
            gates.append(gate_i)
            coarses.append(coarse_i)
            deltas.append(delta_i)
        self.output_i = torch.stack(outputs, dim=0).mean(dim=0)
        self.prior_mask = torch.stack(masks, dim=0).mean(dim=0)
        self.prior_gate = torch.stack(gates, dim=0).mean(dim=0)
        self.coarse_i = torch.stack(coarses, dim=0).mean(dim=0)
        self.prior_delta_i = torch.stack(deltas, dim=0).mean(dim=0)
        return self.output_i

    def build_prior_target(self):
        source = getattr(self.opt, "prior_target_source", "diff")
        use_reflection = (
            source in ("reflection", "hybrid")
            and getattr(self, "issyn", False)
            and isinstance(getattr(self, "target_r", None), torch.Tensor)
            and self.target_r.shape == self.input.shape)
        diff = (self.input - self.target_t).abs().max(dim=1, keepdim=True)[0]
        if use_reflection:
            reflection = self.target_r.abs().max(dim=1, keepdim=True)[0]
            if source == "reflection":
                diff = reflection
            else:
                diff = torch.max(diff, reflection)
        blur = int(getattr(self.opt, "prior_target_blur", 3))
        if blur > 1:
            if blur % 2 == 0:
                blur += 1
            diff = F.avg_pool2d(diff, kernel_size=blur, stride=1, padding=blur // 2)

        abs_floor = getattr(self.opt, "prior_target_abs_floor", 0.0)
        if abs_floor > 0:
            diff = (diff - abs_floor).clamp_min(0.0)

        norm = getattr(self.opt, "prior_target_norm", "max")
        if norm == "quantile":
            quantile = min(max(getattr(self.opt, "prior_target_quantile", 0.99), 0.5), 1.0)
            flat = diff.flatten(2)
            max_value = torch.quantile(flat, quantile, dim=2, keepdim=True).view(
                diff.size(0), diff.size(1), 1, 1)
            max_value = max_value.clamp_min(1e-6)
        elif norm == "meanstd":
            mean = diff.mean(dim=(2, 3), keepdim=True)
            std = diff.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            max_value = mean + std * getattr(self.opt, "prior_target_std_scale", 3.0)
            max_value = max_value.clamp_min(1e-6)
        else:
            max_value = diff.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        target = (diff / max_value).clamp(0.0, 1.0)
        low = getattr(self.opt, "prior_target_low", 0.05)
        high = max(getattr(self.opt, "prior_target_high", 0.5), low + 1e-4)
        target = ((target - low) / (high - low)).clamp(0.0, 1.0)
        gamma = getattr(self.opt, "prior_target_gamma", 1.0)
        if gamma != 1.0:
            target = target.pow(gamma)
        return target.detach()

    def backward_G(self):
        for p in self.netD.parameters():
            p.requires_grad = False

        self.loss_G = 0
        self.loss_CX = None
        self.loss_icnn_pixel = None
        self.loss_icnn_vgg = None
        self.loss_G_GAN = None
        self.loss_prior_mask = None
        self.loss_prior_gate = None
        self.loss_prior_smooth = None
        self.loss_prior_sparse = None
        self.loss_prior_identity = None
        self.loss_coarse_pixel = None

        if getattr(self.opt, "lambda_gan", 0) > 0:
            self.loss_G_GAN = self.loss_dic["gan"].get_g_loss(
                self.netD, self.input, self.output_i, self.target_t)
            self.loss_G += self.loss_G_GAN * self.opt.lambda_gan

        if self.aligned:
            self.loss_icnn_pixel = self.loss_dic["t_pixel"].get_loss(
                self.output_i, self.target_t)
            self.loss_icnn_vgg = self.loss_dic["t_vgg"].get_loss(
                self.output_i, self.target_t)
            self.loss_G += self.loss_icnn_pixel + self.loss_icnn_vgg * self.opt.lambda_vgg

            if getattr(self.opt, "lambda_coarse", 0) > 0:
                self.loss_coarse_pixel = self.loss_dic["t_pixel"].get_loss(
                    self.coarse_i, self.target_t)
                self.loss_G += self.loss_coarse_pixel * self.opt.lambda_coarse

            self.prior_target = self.build_prior_target()
            self.loss_prior_mask = (
                _safe_bce(self.prior_mask, self.prior_target)
                + F.l1_loss(self.prior_mask, self.prior_target))
            self.loss_G += self.loss_prior_mask * getattr(self.opt, "prior_lambda_mask", 0.1)

            self.loss_prior_gate = (
                (1.0 - self.prior_target) * (self.output_i - self.input).abs()).mean()
            self.loss_G += self.loss_prior_gate * getattr(self.opt, "prior_lambda_gate", 0.05)

            self.loss_prior_smooth = _tv_loss(self.prior_mask)
            self.loss_G += self.loss_prior_smooth * getattr(self.opt, "prior_lambda_smooth", 0.01)

            self.loss_prior_sparse = self.prior_mask.mean()
            self.loss_G += self.loss_prior_sparse * getattr(self.opt, "prior_lambda_sparse", 0.0)

            self.loss_prior_identity = (
                (1.0 - self.prior_target) * (self.output_i - self.coarse_i).abs()).mean()
            self.loss_G += self.loss_prior_identity * getattr(self.opt, "prior_lambda_identity", 0.0)
        else:
            self.loss_CX = self.loss_dic["t_cx"].get_loss(self.output_i, self.target_t)
            self.loss_G += self.loss_CX

        self.loss_G.backward()

    def get_current_errors(self):
        ret_errors = super(ERRNetPriorBranchModel, self).get_current_errors()
        if self.loss_coarse_pixel is not None:
            ret_errors["Coarse"] = self.loss_coarse_pixel.item()
        if self.loss_prior_mask is not None:
            ret_errors["Prior"] = self.loss_prior_mask.item()
        if self.loss_prior_gate is not None:
            ret_errors["Gate"] = self.loss_prior_gate.item()
        if self.loss_prior_smooth is not None:
            ret_errors["Smooth"] = self.loss_prior_smooth.item()
        if self.loss_prior_sparse is not None:
            ret_errors["Sparse"] = self.loss_prior_sparse.item()
        if self.loss_prior_identity is not None:
            ret_errors["Identity"] = self.loss_prior_identity.item()
        return ret_errors

    def get_current_visuals(self):
        ret_visuals = super(ERRNetPriorBranchModel, self).get_current_visuals()
        ret_visuals["coarse"] = tensor2im(self.coarse_i).astype(np.uint8)
        ret_visuals["prior_mask"] = tensor2im(self.prior_mask).astype(np.uint8)
        if self.prior_gate is not None:
            ret_visuals["prior_gate"] = tensor2im(self.prior_gate).astype(np.uint8)
        if self.prior_target is not None:
            ret_visuals["prior_target"] = tensor2im(self.prior_target).astype(np.uint8)
        return ret_visuals

    def eval(self, data, savedir=None, suffix=None, pieapp=None, tta=False):
        self._eval()
        self.set_input(data, "eval")

        with torch.no_grad():
            output_tensor = self.forward_tta() if tta else self.forward()
            output_i = tensor2im(output_tensor)
            target = tensor2im(self.target_t)

            if self.aligned:
                h = min(output_i.shape[0], target.shape[0])
                w = min(output_i.shape[1], target.shape[1])
                res = index.quality_assess(output_i[:h, :w], target[:h, :w])
            else:
                res = {}

            if savedir is not None:
                name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
                out_dir = join(savedir, name)
                util.mkdirs(out_dir)
                out_name = "{}_{}.png".format(self.opt.name, suffix) if suffix else "{}.png".format(self.opt.name)
                Image.fromarray(output_i.astype(np.uint8)).save(join(out_dir, out_name))
                Image.fromarray(target.astype(np.uint8)).save(join(out_dir, "t_label.png"))
                Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(out_dir, "m_input.png"))
                self._save_prior_aux(out_dir)

            return res

    def test(self, data, savedir=None, tta=False):
        self._eval()
        self.set_input(data, "test")

        if self.data_name is not None and savedir is not None:
            name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
            out_dir = join(savedir, name)
            util.mkdirs(out_dir)
            if os.path.exists(join(out_dir, "{}.png".format(self.opt.name))):
                return

        with torch.no_grad():
            output_i = self.forward_tta() if tta else self.forward()
            output_i = tensor2im(output_i)
            if self.data_name is not None and savedir is not None:
                name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
                out_dir = join(savedir, name)
                Image.fromarray(output_i.astype(np.uint8)).save(join(out_dir, "{}.png".format(self.opt.name)))
                Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(out_dir, "m_input.png"))
                self._save_prior_aux(out_dir)

    def _save_prior_aux(self, out_dir):
        if self.prior_mask is not None and getattr(self.opt, "prior_save_masks", True):
            Image.fromarray(tensor2im(self.prior_mask).astype(np.uint8)).save(join(out_dir, "prior_mask.png"))
        if self.prior_gate is not None and getattr(self.opt, "prior_save_masks", True):
            Image.fromarray(tensor2im(self.prior_gate).astype(np.uint8)).save(join(out_dir, "prior_gate.png"))
        if self.coarse_i is not None and getattr(self.opt, "prior_save_coarse", True):
            Image.fromarray(tensor2im(self.coarse_i).astype(np.uint8)).save(join(out_dir, "coarse.png"))
        if getattr(self, "target_t", None) is not None and isinstance(self.target_t, torch.Tensor):
            target = self.build_prior_target()
            Image.fromarray(tensor2im(target).astype(np.uint8)).save(join(out_dir, "prior_target.png"))

    def load_base_icnn(self, path):
        state_dict = _torch_load_compat(path, map_location=torch.device("cpu"))
        base_state = state_dict["icnn"] if isinstance(state_dict, dict) and "icnn" in state_dict else state_dict
        if self.prior_base == "cascade" and hasattr(self.net_i, "coarse_net"):
            missing, unexpected = self.net_i.coarse_net.load_state_dict(base_state, strict=False)
            target = "cascade coarse net"
        else:
            missing, unexpected = self.net_i.load_state_dict(base_state, strict=False)
            target = "base net"
        print("[i] initialized %s from %s" % (target, path))
        if missing:
            print("[i] missing keys: %d" % len(missing))
        if unexpected:
            print("[i] unexpected keys: %d" % len(unexpected))

    def load_prior(self, path=None, resume_epoch=None):
        if path is None:
            path = util.get_model_list(self.save_dir, self.name(), epoch=resume_epoch)
        state_dict = _torch_load_compat(path, map_location=torch.device("cpu"))
        self.net_i.load_state_dict(state_dict["icnn"], strict=False)
        self.prior_head.load_state_dict(state_dict["prior_head"], strict=False)
        self.refine_net.load_state_dict(state_dict["refine_net"], strict=False)
        self.epoch = state_dict.get("epoch", 0)
        self.iterations = state_dict.get("iterations", 0)
        if self.isTrain:
            if "opt_g" in state_dict:
                self.optimizer_G.load_state_dict(state_dict["opt_g"])
            if getattr(self.opt, "lambda_gan", 0) > 0 and "netD" in state_dict:
                self.netD.load_state_dict(state_dict["netD"])
                self.optimizer_D.load_state_dict(state_dict["opt_d"])
        print("Resume prior branch from %s, epoch %d, iteration %d" % (
            path, self.epoch, self.iterations))
        return state_dict

    def state_dict(self):
        state_dict = {
            "icnn": self.net_i.state_dict(),
            "prior_head": self.prior_head.state_dict(),
            "refine_net": self.refine_net.state_dict(),
            "opt_g": self.optimizer_G.state_dict(),
            "epoch": self.epoch,
            "iterations": self.iterations,
            "prior_config": {
                "prior_base": self.prior_base,
                "prior_delta_scale": getattr(self.opt, "prior_delta_scale", 1.0),
                "prior_gate_threshold": getattr(self.opt, "prior_gate_threshold", 0.0),
                "prior_gate_gamma": getattr(self.opt, "prior_gate_gamma", 1.0),
                "prior_freeze_base": self.prior_freeze_base,
                "prior_head_feats": getattr(self.opt, "prior_head_feats", 32),
                "prior_refine_feats": getattr(self.opt, "prior_refine_feats", 64),
                "prior_refine_blocks": getattr(self.opt, "prior_refine_blocks", 4),
                "prior_target_source": getattr(self.opt, "prior_target_source", "diff"),
                "prior_target_norm": getattr(self.opt, "prior_target_norm", "max"),
                "prior_target_quantile": getattr(self.opt, "prior_target_quantile", 0.99),
                "prior_target_abs_floor": getattr(self.opt, "prior_target_abs_floor", 0.0),
            },
        }
        if getattr(self.opt, "lambda_gan", 0) > 0:
            state_dict.update({
                "opt_d": self.optimizer_D.state_dict(),
                "netD": self.netD.state_dict(),
            })
        return state_dict
