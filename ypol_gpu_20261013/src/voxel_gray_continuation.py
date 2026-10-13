"""Gray-only continuation and physical-length 3D Gaussian filtering."""
import math
import statistics

import torch
import torch.nn.functional as F


def correlated_random_theta(shape, *, pitch_nm, layer_height_nm, block_nm, seed, noise=.35):
    """Random physical blocks expanded to independent fine-grid parameters."""
    if not math.isfinite(block_nm) or block_nm <= 0:
        raise ValueError("Random block size must be positive and finite")
    if not math.isfinite(noise) or noise < 0:
        raise ValueError("Random noise must be finite and nonnegative")
    repeats = []
    for size, spacing in zip(shape, (layer_height_nm, pitch_nm, pitch_nm)):
        ratio = block_nm / spacing
        count = round(ratio)
        if count < 1 or not math.isclose(count, ratio, abs_tol=1e-9) or size % count:
            raise ValueError("Random block size must exactly tile the design grid in all axes")
        repeats.append(count)
    coarse_shape = tuple(size // repeat for size, repeat in zip(shape, repeats))
    generator = torch.Generator().manual_seed(seed)
    theta = (.5 + noise * (2 * torch.rand(coarse_shape, generator=generator, dtype=torch.float64) - 1)).clamp(0, 1)
    for axis, repeat in enumerate(repeats):
        theta = theta.repeat_interleave(repeat, dim=axis)
    return theta.contiguous()


def gaussian_3d(theta, sigma_xy, sigma_z, *, z_boundary="replicate",
                fixed_mask=None, fixed_values=None):
    """Normalized Gaussian; sigma arguments are nm sigma divided by voxel pitch.

    domain_normalized truncates at the design's top/bottom and renormalizes.
    Air and SiO2 outside that domain stay separate fixed optical materials,
    not fictitious TiO2 densities. Optional internal fixed voxels participate
    in the filter but are restored exactly and have no design gradient.
    """
    if z_boundary not in ("replicate", "domain_normalized"):
        raise ValueError("Unknown z boundary policy")
    if fixed_mask is not None:
        if fixed_values is None or fixed_mask.shape != theta.shape:
            raise ValueError("Fixed regions require a matching mask and density values")
        values = torch.as_tensor(fixed_values, dtype=theta.dtype, device=theta.device).detach()
        source = torch.where(fixed_mask, values, theta)
    else:
        source = theta
    out = source[None, None]
    for axis, sigma in ((2, sigma_xy), (1, sigma_xy), (0, sigma_z)):
        if sigma <= 0:
            continue
        radius = math.ceil(3*sigma)
        if radius >= theta.shape[axis]:
            raise ValueError("Gaussian radius must be smaller than the design dimension")
        t = torch.arange(-radius, radius+1, dtype=theta.dtype, device=theta.device)
        kernel = torch.exp(-.5*(t/sigma).square())
        kernel = kernel/kernel.sum()
        shape = [1, 1, 1, 1, 1]
        shape[axis+2] = kernel.numel()
        padding = [0]*6
        padding[2*(2-axis):2*(2-axis)+2] = [radius, radius]
        if axis == 0 and z_boundary == "domain_normalized":
            denominator = F.conv3d(F.pad(torch.ones_like(out[:, :, :, :1, :1]), tuple(padding)),
                                   kernel.reshape(shape))
            out = F.conv3d(F.pad(out, tuple(padding)), kernel.reshape(shape)) / denominator
        else:
            out = F.conv3d(F.pad(out, tuple(padding), mode="replicate" if axis == 0 else "circular"),
                           kernel.reshape(shape))
    filtered = out[0, 0]
    return torch.where(fixed_mask, values, filtered) if fixed_mask is not None else filtered


def gray_plateau_decision(records, beta, step, *, max_beta=64., window=10, check_every=5,
                          absolute_tolerance=.005, relative_tolerance=.005):
    if step == 0 or step % check_every:
        return None
    event = dict(step=step, old_beta=beta, applied_beta=beta, action="hold", policy="gray_plateau_1p5")
    if beta >= max_beta:
        return dict(event, reason="beta_cap")
    segment = []
    current_mode = records[-1].get("objective_mode") if records else None
    for row in reversed(records):
        if (not math.isclose(row["beta"], beta, abs_tol=1e-12, rel_tol=0)
                or row.get("objective_mode") != current_mode):
            break
        segment.append(row)
    segment.reverse()
    if len(segment) < 2*window:
        return dict(event, reason="insufficient_same_beta_history")
    values = [row["fom"] for row in segment[-2*window:]]
    if not all(math.isfinite(v) for v in values):
        return dict(event, reason="nonfinite_gray_fom")
    before, after = statistics.mean(values[:window]), statistics.mean(values[window:])
    gain = after-before
    tolerance = max(absolute_tolerance, relative_tolerance*abs(before))
    event.update(before=before, after=after, gain=gain, tolerance=tolerance, window=window)
    if gain > tolerance:
        return dict(event, reason="gray_improving")
    if gain < -tolerance:
        return dict(event, reason="gray_deteriorating_not_plateau")
    return dict(event, action="increase", reason="gray_fom_plateau",
                applied_beta=min(max_beta, 1.5*beta), requested_multiplier=1.5)
