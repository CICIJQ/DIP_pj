import torch


def apply_reflection_correction_amplification(
    input_image,
    rdnet_image,
    mask,
    strength=0.65,
    max_extra_delta=0.35,
):
    """Amplify RDNet's correction in high-confidence reflection regions.

    If RDNet moved the input from I to T_rd, RCA extrapolates a little farther
    along the same correction direction:

        T_rca = T_rd + strength * M * (T_rd - I)

    `max_extra_delta` limits the extra per-channel correction so aggressive
    settings remain bounded.
    """

    if input_image.ndim == 3:
        input_image = input_image.unsqueeze(0)
    if rdnet_image.ndim == 3:
        rdnet_image = rdnet_image.unsqueeze(0)
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)

    direction = rdnet_image - input_image
    extra = float(strength) * mask * direction
    if max_extra_delta is not None and max_extra_delta > 0:
        extra = extra.clamp(-float(max_extra_delta), float(max_extra_delta))
    return (rdnet_image + extra).clamp(0.0, 1.0)
