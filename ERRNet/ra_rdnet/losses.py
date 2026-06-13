import torch
import torch.nn.functional as F


def gradient_l1(pred, target):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def mask_weighted_l1(pred, target, mask, eps=1e-6):
    error = (pred - target).abs()
    weighted = error * mask
    return weighted.sum() / (mask.sum() * pred.shape[1] + eps)


def refiner_loss(
    refined,
    gt,
    rdnet,
    mask,
    lambda_l1=1.0,
    lambda_mask=1.0,
    lambda_preserve=0.2,
    lambda_gradient=0.1,
):
    l1 = F.l1_loss(refined, gt)
    mask_l1 = mask_weighted_l1(refined, gt, mask)
    preserve = ((1.0 - mask) * (refined - rdnet).abs()).mean()
    gradient = gradient_l1(refined, gt)
    total = (
        float(lambda_l1) * l1
        + float(lambda_mask) * mask_l1
        + float(lambda_preserve) * preserve
        + float(lambda_gradient) * gradient
    )
    return {
        "total": total,
        "l1": l1.detach(),
        "mask_l1": mask_l1.detach(),
        "preserve": preserve.detach(),
        "gradient": gradient.detach(),
    }
