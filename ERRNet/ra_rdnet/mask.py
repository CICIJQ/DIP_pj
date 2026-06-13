import torch
import torch.nn.functional as F


def rgb_to_luma(image):
    weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def robust_normalize(value, low_q=0.05, high_q=0.95, eps=1e-6):
    if value.ndim == 3:
        value = value.unsqueeze(0)
    batches = []
    for item in value:
        flat = item.reshape(-1)
        low = torch.quantile(flat, low_q)
        high = torch.quantile(flat, high_q)
        batches.append(((item - low) / (high - low + eps)).clamp(0.0, 1.0))
    return torch.stack(batches, dim=0)


def gaussian_kernel1d(radius, sigma, device, dtype):
    if radius <= 0:
        return None
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def gaussian_blur(image, radius=5, sigma=None):
    if radius <= 0:
        return image
    if sigma is None:
        sigma = max(float(radius) / 2.0, 1.0)
    kernel = gaussian_kernel1d(radius, sigma, image.device, image.dtype)
    channels = image.shape[1]
    kernel_x = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kernel_y = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    image = F.pad(image, (radius, radius, 0, 0), mode="reflect")
    image = F.conv2d(image, kernel_x, groups=channels)
    image = F.pad(image, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(image, kernel_y, groups=channels)


def sobel_magnitude(value):
    if value.ndim == 3:
        value = value.unsqueeze(0)
    kernel_x = value.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    kernel_y = value.new_tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    ).view(1, 1, 3, 3)
    padded = F.pad(value, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(padded, kernel_x)
    gy = F.conv2d(padded, kernel_y)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def estimate_reflection_mask(
    input_image,
    rdnet_image,
    sensitivity=1.25,
    gamma=0.70,
    blur_radius=5,
    diff_weight=0.55,
    bright_weight=0.30,
    edge_weight=0.15,
    floor=0.0,
):
    """Estimate a soft reflection confidence map from input and RDNet output.

    The map is high where RDNet changed the input strongly, where the input has
    bright residual energy relative to RDNet, and where the residual contains
    sharp edges.  It is intentionally heuristic so it can run without training.
    """

    if input_image.ndim == 3:
        input_image = input_image.unsqueeze(0)
    if rdnet_image.ndim == 3:
        rdnet_image = rdnet_image.unsqueeze(0)
    input_image = input_image.clamp(0.0, 1.0)
    rdnet_image = rdnet_image.clamp(0.0, 1.0)

    diff = (input_image - rdnet_image).abs().mean(dim=1, keepdim=True)
    luma_input = rgb_to_luma(input_image)
    luma_rdnet = rgb_to_luma(rdnet_image)
    bright_residual = (luma_input - luma_rdnet).clamp(min=0.0)
    edge_residual = sobel_magnitude(diff)

    diff_n = robust_normalize(diff)
    bright_n = robust_normalize(bright_residual)
    edge_n = robust_normalize(edge_residual)
    mask = diff_weight * diff_n + bright_weight * bright_n + edge_weight * edge_n
    mask = robust_normalize(mask)
    mask = gaussian_blur(mask, radius=int(blur_radius))
    mask = (mask * float(sensitivity)).clamp(0.0, 1.0)
    if gamma > 0:
        mask = mask.clamp(min=1e-6).pow(float(gamma))
    if floor > 0:
        mask = torch.maximum(mask, mask.new_full(mask.shape, float(floor)))
    return mask.clamp(0.0, 1.0)
