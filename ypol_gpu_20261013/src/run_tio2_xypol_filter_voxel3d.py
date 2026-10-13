"""TiO2 X/Y polarizer topology optimization using voxel3d's RCWA forward.

This adapts the voxel3d stack forward from hyoseokp/information-optimal-color-router
to a polarization-filter objective instead of the original color-router objective.
The design is a periodic 4 x 4 um TiO2/air voxel stack on SiO2.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import generate_binary_structure, label
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
VOXEL_REPO = ROOT / "third_party" / "information-optimal-color-router-latest"
VOXEL_SRC = VOXEL_REPO / "src"
if str(VOXEL_SRC) not in sys.path:
    sys.path.insert(0, str(VOXEL_SRC))

from cr_itd_v2.forward import DEFAULT_R1A_PUPIL, DEFAULT_R1A_STACK, PupilRay  # noqa: E402
from voxel3d import eig as eigmod, materials_siefke  # noqa: E402
from voxel3d.common import gaussian_filter_periodic, project  # noqa: E402
from voxel3d.forward import ENERGY_GUARD_EPS, FastPupilForward as UpstreamForward  # noqa: E402
from voxel_pixel_fourier import pixel_convolution
from voxel_support import supported_and_vented, closed_air
from voxel_target_validation import target_reached, assess_validation
from voxel3d_run_plots import save_history_plots, save_snapshot_plot, save_evolution_overview
from voxel_adaptive_beta import AdaptiveBeta, BetaRules
from voxel_live_report import atomic_json, refresh_live_plots, write_live_status
from voxel_report_guard import report_task
from voxel_polarizer_objective import softmin_difference, softmin_diagnostics
from voxel_gray_continuation import gaussian_3d, gray_plateau_decision, correlated_random_theta
from voxel_memory_forward import EigenReplayMixin, LayerCheckpointMixin, spectral_recompute
from voxel_objective_schedule import active_objective_args, phase_changed, gradient_outlier
from voxel_dashboard import read_json
from voxel_goal_policy import recycle_due
from voxel_binary_ste import binary_forward_ste, gray_penalty_weight
from voxel_snapshot_storage import save_step_arrays
from portable_report import save_history_plots, refresh_live_plots, write_live_status, write_summary


class FastPupilForward(EigenReplayMixin, LayerCheckpointMixin, UpstreamForward):
    """Retain voxel3d's S-matrix/autograd solver with exact binary pixel geometry."""

    def _econv(self, densities):
        rho = densities.to(self.cdt)
        eps_map = (self.eps_bg[None, :, None, None]
                   + rho[:, None] * (self.eps_design - self.eps_bg)[None, :, None, None])
        matrix = pixel_convolution(eps_map, *self.orders).flatten(0, 1)
        return matrix, torch.linalg.inv(matrix)


def dilate(x: torch.Tensor, reach: int) -> torch.Tensor:
    for _ in range(reach):
        padded = F.pad(x[None, None], (1, 1, 1, 1), mode="circular")
        x = F.max_pool2d(padded, 3, 1)[0, 0]
    return x


def gate_grey(rhos: list[torch.Tensor], reach: int, support_mode="local_grounded") -> torch.Tensor:
    """Gate from substrate upward; RCWA layer 0 faces the incident light."""
    if support_mode == "posts_vented":
        return torch.stack(rhos)
    out: list[torch.Tensor] = [rhos[-1]]
    for i in range(len(rhos) - 2, -1, -1):
        support = (2.0 * dilate(out[-1], reach)).clamp(0, 1)
        out.append(rhos[i] * support)
    return torch.stack(out[::-1])


def gate_binary(rhos: list[torch.Tensor], reach: int, threshold: float, support_mode="local_grounded") -> torch.Tensor:
    hard = [(r.detach() >= threshold).to(r.dtype) for r in rhos]
    if support_mode == "posts_vented":
        projected = supported_and_vented(torch.stack(hard).cpu().numpy())
        return torch.as_tensor(projected, dtype=hard[0].dtype, device=hard[0].device)
    out: list[torch.Tensor] = [hard[-1]]
    for i in range(len(hard) - 2, -1, -1):
        out.append(hard[i] * (dilate(out[-1], reach) >= 0.5).to(hard[i].dtype))
    stacked = torch.stack(out[::-1])
    # Face contacts only: diagonal/edge contacts are not printable bonds.
    # This projection is applied before the EM solve, never after acceptance.
    labels, _ = label(stacked.detach().cpu().numpy() > 0.5, generate_binary_structure(3, 1))
    anchors = np.unique(labels[-1])
    keep = np.isin(labels, anchors[anchors > 0])
    return stacked * torch.as_tensor(keep, dtype=stacked.dtype, device=stacked.device)


def solid_connectivity(stack_arr: np.ndarray) -> dict:
    occupied = np.asarray(stack_arr) > 0.5
    labels, count = label(occupied, generate_binary_structure(3, 1))
    anchors = np.unique(labels[-1])
    floating = occupied & ~np.isin(labels, anchors[anchors > 0])
    return {"solid_voxels": int(occupied.sum()), "solid_components": count,
            "grounded_components": int((anchors > 0).sum()),
            "floating_voxels": int(floating.sum()),
            "connectivity_convention": "finite aperture, face contacts, substrate at last layer"}


def unsupported_voxels(stackb: torch.Tensor, reach: int) -> int:
    n = 0
    for i in range(stackb.shape[0] - 1):
        support = torch.zeros_like(stackb[i])
        for dy in range(-reach, reach + 1):
            for dx in range(-reach, reach + 1):
                support = torch.maximum(support, torch.roll(stackb[i + 1], (dy, dx), (0, 1)))
        n += int(((stackb[i] > 0.5) & (support < 0.5)).sum().item())
    return n


def polarized_tabs(
    fwd: FastPupilForward,
    densities: torch.Tensor,
    *,
    tags: tuple[str, ...],
    grad: list[bool] | None,
) -> torch.Tensor:
    """Return detector tabs without voxel3d's incoherent X/Y averaging.

    Output shape is [density, well, wavelength, pol], where pol 0 is X and
    pol 1 is Y. Summing over the four wells gives total transmitted power.
    """
    if getattr(fwd, "spectral_children", None):
        return spectral_recompute(fwd.spectral_children, densities, tags, grad, polarized_tabs)
    if densities.ndim != 4 or densities.shape[1] != fwd.n_layers or tuple(densities.shape[2:]) != fwd.shape:
        raise ValueError("densities must be [D, n_layers, *density_shape]")
    if densities.dtype not in (torch.float32, torch.float64):
        raise ValueError("densities must be float32/64")

    n_d = densities.shape[0]
    if len(tags) != n_d or len(set(tags)) != n_d:
        raise ValueError("one distinct tag per density")
    if grad is not None and len(grad) != n_d:
        raise ValueError("one grad flag per density")

    e_all, einv_all = fwd._econv(densities.reshape(-1, *fwd.shape))
    n_solves = len(fwd.solve_spec)
    pairs = [(d, b) for d in range(n_d) for b in range(n_solves)]
    nudges = [float(fwd.stack.rayleigh_frequency_nudge)] * len(pairs)
    wells, trans = fwd._run(pairs, nudges, e_all, einv_all, tags=tags, tally={}, grad=grad)
    bad = (~torch.isfinite(trans.detach())) | (trans.detach() > 1.0 + ENERGY_GUARD_EPS)
    if bool(bad.any()):
        wells = fwd._guarded(pairs, wells, trans, bad, e_all, einv_all, grad=grad)

    n_wl = len(fwd.wavelengths_nm)
    n_rays = len(fwd.rays)
    weights = fwd.ray_weight.to(wells.dtype)
    out = []
    for d in range(n_d):
        wd = wells[d * n_solves : (d + 1) * n_solves].reshape(n_wl, n_rays, 4, 2)
        tabs = (wd * weights[None, :, None, None]).sum(1).permute(1, 0, 2)
        if not bool(torch.isfinite(tabs.detach()).all()):
            raise RuntimeError("polarized response contains non-finite values")
        out.append(tabs.to(torch.float64))
    return torch.stack(out)


def build_stack(args):
    materials_siefke.install()
    pitch_nm = args.period_um * 1000.0 / args.grid
    layer_h = args.layer_height_nm / 1000.0
    total_h = args.layers * layer_h
    stack = dataclasses.replace(
        DEFAULT_R1A_STACK,
        period_um=(args.period_um, args.period_um),
        superstrate_material="air",
        substrate_material="SiO2_nk",
        design_material=materials_siefke.NAME,
        design_fill_material="air",
        design_height_um=layer_h,
        layer_heights_um=(layer_h,) * args.layers if args.layers > 1 else (),
        layer_gap_um=0.0,
        detector_z_um=total_h + args.detector_gap_um,
        density_shape=(args.grid, args.grid),
        fourier_orders=(args.orders, args.orders),
        detector_samples=(args.detector_samples, args.detector_samples),
        rayleigh_frequency_nudge=args.rayleigh_nudge,
    )
    pupil = dataclasses.replace(
        DEFAULT_R1A_PUPIL,
        numerical_aperture=0.01,
        chief_ray_angle_deg=0.0,
        chief_ray_azimuth_deg=0.0,
        radial_order=1,
        azimuth_count=1,
        reference_index=1.0,
    )
    rays = (PupilRay(theta_reference_rad=0.0, phi_rad=0.0, sx_reference=0.0, sy_reference=0.0, weight=1.0),)
    return stack, pupil, rays, pitch_nm


def filter_density(theta, sigma_px, sigma_z_px=0., z_boundary="replicate"):
    if sigma_z_px > 0 or z_boundary != "replicate":
        return gaussian_3d(theta, sigma_px, sigma_z_px, z_boundary=z_boundary)
    return torch.stack([gaussian_filter_periodic(layer, sigma_px) if sigma_px > 0 else layer
                        for layer in theta])


def make_rhos(theta: torch.Tensor, beta: float, sigma_px: float, eta: float, sigma_z_px=0.,
              z_boundary="replicate") -> list[torch.Tensor]:
    smooth = filter_density(theta, sigma_px, sigma_z_px, z_boundary)
    return list(project(smooth, beta, eta).clamp(0, 1).unbind(0))


def evaluation_binary(filtered, rhos, args, threshold):
    if args.binary_evaluation == "filtered_threshold":
        return (filtered.detach() >= threshold).to(filtered.dtype)
    return gate_binary(rhos, args.reach, threshold, args.support_mode)


def probe_beta(theta, beta, sigma_px, args, fwd):
    """Evaluate both objectives at fixed theta without modifying Adam or geometry state."""
    result = {}
    with torch.no_grad():
        rhos = make_rhos(theta, beta, sigma_px, args.eta)
        for name, density in (("gray", gate_grey(rhos, args.reach, args.support_mode)),
                              ("binary", gate_binary(rhos, args.reach, args.hard_threshold, args.support_mode))):
            tabs = polarized_tabs(fwd, density[None], tags=("beta_probe_"+name,), grad=[False])[0]
            fom, _ = objective_from_tabs(tabs, **objective_kwargs(args))
            metrics = metrics_from_tabs(tabs, args.pass_pol)
            result[name] = dict(fom=float(fom.detach()), **{key: metrics[key] for key in
                                ("mean_tx", "mean_ty", "min_delta", "min_tx", "max_ty")})
    return result


def metrics_from_tabs(tabs: torch.Tensor, pass_pol="x") -> dict[str, float | list[dict[str, float]]]:
    total = tabs.sum(0)  # [wavelength, pol]
    tx = total[:, 0].detach().cpu().numpy()
    ty = total[:, 1].detach().cpu().numpy()
    passed, blocked = (tx, ty) if pass_pol == "x" else (ty, tx)
    delta = passed - blocked
    contrast = delta / np.maximum(tx + ty, 1e-12)
    er_db = 10.0 * np.log10(np.maximum(passed, 1e-15) / np.maximum(blocked, 1e-15))
    return {
        "min_tx": float(tx.min()),
        "mean_tx": float(tx.mean()),
        "max_ty": float(ty.max()),
        "mean_ty": float(ty.mean()),
        "min_ty": float(ty.min()),
        "max_tx": float(tx.max()),
        "min_pass": float(passed.min()),
        "max_block": float(blocked.max()),
        "min_delta": float(delta.min()),
        "mean_delta": float(delta.mean()),
        "min_contrast": float(contrast.min()),
        "mean_contrast": float(contrast.mean()),
        "min_er_db": float(er_db.min()),
        "mean_er_db": float(er_db.mean()),
    }


def objective_from_tabs(
    tabs: torch.Tensor,
    *,
    w_delta: float,
    w_tx: float,
    w_ty: float,
    target_tx: float,
    target_ty: float,
    w_tx_floor: float,
    w_ty_ceiling: float,
    worst_weight: float,
    objective_mode: str = "legacy",
    softmin_tau: float = .03,
    worst_tx_penalty: float = 2.,
    worst_ty_penalty: float = 2.,
    pass_pol: str = "x",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total = tabs.sum(0)
    tx = total[:, 0]
    ty = total[:, 1]
    if pass_pol not in ("x", "y"):
        raise ValueError("pass_pol must be x or y")
    # Legacy optimizer names below mean pass/block roles, not output labels.
    # The EM forward and every stored spectrum retain physical [X, Y] order.
    if pass_pol == "y":
        tx, ty = ty, tx
    if objective_mode == "softmin":
        return softmin_difference(tx, ty, tau=softmin_tau, target_tx=target_tx, target_ty=target_ty,
                                  tx_penalty=worst_tx_penalty, ty_penalty=worst_ty_penalty)
    if objective_mode != "legacy":
        raise ValueError("Unknown objective mode")
    delta = tx - ty
    fom = w_delta * delta.mean() + w_tx * tx.mean() - w_ty * ty.mean()
    if worst_weight:
        fom = fom + worst_weight * torch.amin(delta)
    if w_tx_floor:
        fom = fom - w_tx_floor * torch.relu(target_tx - tx).square().mean()
    if w_ty_ceiling:
        fom = fom - w_ty_ceiling * torch.relu(ty - target_ty).square().mean()
    return fom, {"tx": tx, "ty": ty, "delta": delta}


def objective_kwargs(args):
    keys = ("w_delta", "w_tx", "w_ty", "target_tx", "target_ty", "w_tx_floor", "w_ty_ceiling",
            "worst_weight", "objective_mode", "softmin_tau", "worst_tx_penalty", "worst_ty_penalty")
    return dict({key: getattr(args, key) for key in keys}, pass_pol=args.pass_pol)


def save_mask_images(out: Path, stack_arr: np.ndarray, prefix: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    selected = np.unique(np.linspace(0, stack_arr.shape[0]-1, min(16, stack_arr.shape[0])).astype(int))
    layers = len(selected)
    cols = min(4, layers)
    rows = math.ceil(layers / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows), squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i < layers:
            li = selected[i]
            ax.imshow(stack_arr[li].T, cmap="gray_r", vmin=0, vmax=1, origin="lower")
            ax.set_title(f"layer {li + 1} / {stack_arr.shape[0]}")
    if layers < stack_arr.shape[0]:
        fig.suptitle("Selected layer slices; full-resolution arrays saved in NPZ")
    fig.tight_layout()
    fig.savefig(out / f"{prefix}_layers.png", dpi=180)
    plt.close(fig)


def legacy_x_summary(out: Path, wavelengths: list[float], records: list[dict], best: dict, args: argparse.Namespace) -> None:
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "step",
            "beta",
            "fom",
            "mean_tx",
            "min_tx",
            "mean_ty",
            "max_ty",
            "mean_delta",
            "min_delta",
            "mean_er_db",
            "min_er_db",
            "grayness",
            "gradient_norm",
            "hard_weight",
            "optimization_fom",
            "bin_threshold",
            "bin_fom",
            "bin_mean_tx",
            "bin_min_tx",
            "bin_mean_ty",
            "bin_max_ty",
            "bin_mean_delta",
            "bin_min_delta",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in fields})

    md = [
        "# TiO2 voxel3d X/Y polarizer run",
        "",
        f"- period: {args.period_um:.3f} um",
        f"- grid: {args.grid} x {args.grid} ({args.period_um * 1000 / args.grid:.1f} nm pixel)",
        f"- layers: {args.layers} x {args.layer_height_nm:.1f} nm = {args.layers * args.layer_height_nm / 1000:.3f} um",
        f"- wavelengths: {', '.join(f'{w:.0f}' for w in wavelengths)} nm",
        f"- Fourier orders: +/-{args.orders}",
        f"- objective mode: {args.objective_mode}; maximize X total transmission and suppress Y total transmission",
        "- evaluation: periodic RCWA total transmission, not finite-aperture FDTD or measured reflection",
        "- layer ordering: index 0 at top, last layer on substrate; support gate runs bottom to top",
        f"- support projection: {getattr(args, 'support_mode', 'local_grounded')}; applied before every binary forward solve",
        "- posts_vented adds substrate posts and opens enclosed air; local_grounded deletes unsupported components",
        "- this projection does not certify minimum neck/gap or full process printability",
        ("- objective changed in a separate branch; global step numbers retained, Adam moments reset; see objective_branch.json"
         if (out / "objective_branch.json").exists() else
         "- step 0 is the saved random seed density" if args.init_mode == "random" and not args.init_state and not args.init_binary
         else "- step 0 is this continuation's starting density, not a reconstruction of the original seed"),
        "- state.pt contains post-update theta; snapshot arrays and best_state.pt match evaluated pre-update geometry",
        "",
        "## Best binary candidate",
        "",
    ]
    if best:
        md.extend(
            [
                f"- step: {best['step']}",
                f"- threshold: {best['threshold']}",
                f"- F: {best['fom']:.6f}",
                f"- mean T_X: {best['metrics']['mean_tx']:.6f}",
                f"- min T_X: {best['metrics']['min_tx']:.6f}",
                f"- mean T_Y: {best['metrics']['mean_ty']:.6f}",
                f"- max T_Y: {best['metrics']['max_ty']:.6f}",
                f"- mean delta: {best['metrics']['mean_delta']:.6f}",
                f"- min delta: {best['metrics']['min_delta']:.6f}",
                f"- mean ER: {best['metrics']['mean_er_db']:.3f} dB",
                f"- min ER: {best['metrics']['min_er_db']:.3f} dB",
                "",
                "| wavelength | T_X | T_Y | delta | contrast | ER dB |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in best["spectrum"]:
            md.append(
                f"| {row['wavelength_nm']:.0f} | {row['tx']:.4f} | {row['ty']:.4f} | "
                f"{row['delta']:.4f} | {row['contrast']:.4f} | {row['er_db']:.2f} |"
            )
    (out / "SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def spectrum_rows(wavelengths: list[float], tabs: torch.Tensor, pass_pol="x") -> list[dict[str, float]]:
    total = tabs.sum(0)
    tx = total[:, 0].detach().cpu().numpy()
    ty = total[:, 1].detach().cpu().numpy()
    rows = []
    for w, x, y in zip(wavelengths, tx, ty):
        passed, blocked = (x, y) if pass_pol == "x" else (y, x)
        rows.append(
            {
                "wavelength_nm": float(w),
                "tx": float(x),
                "ty": float(y),
                "delta": float(passed - blocked),
                "contrast": float((passed - blocked) / max(x + y, 1e-12)),
                "er_db": float(10.0 * math.log10(max(passed, 1e-15) / max(blocked, 1e-15))),
            }
        )
    return rows


def validate_binary_candidate(out, item, step, args, device):
    """Freeze the mask; never modify geometry or thresholds during validation."""
    wavelengths = [float(w) for w in args.validation_wavelengths.split(",")]
    orders = [int(o) for o in args.validation_orders.split(",")]
    folder = Path(out) / "validation" / f"step_{step:04d}_t{item['threshold']:g}"
    folder.mkdir(parents=True, exist_ok=True)
    mask = item["stack"]
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("Acceptance requires an exactly binary structure")
    np.savez_compressed(folder / "frozen_binary.npz", stack=mask, step=step, threshold=item["threshold"])
    results = []
    for order in orders:
        options = argparse.Namespace(**dict(vars(args), orders=order,
                                            detector_samples=max(args.detector_samples, 4*order+4)))
        stack, pupil, rays, _ = build_stack(options)
        dtype = torch.complex128 if args.dtype == "complex128" else torch.complex64
        fwd = FastPupilForward(stack=stack, pupil_spec=pupil, wavelengths_nm=wavelengths,
                               rays=rays, device=device, chunk=1, dtype=dtype, verbose_guard=False)
        with torch.no_grad():
            density = torch.as_tensor(mask, dtype=torch.float64, device=device)
            tabs = polarized_tabs(fwd, density[None], tags=("frozen_binary",), grad=[False])[0]
        results.append(dict(order=order, spectrum=spectrum_rows(wavelengths, tabs, args.pass_pol),
                            metrics=metrics_from_tabs(tabs, args.pass_pol)))
        (folder / "spectra.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[validation] step={step} order={order} " + json.dumps(results[-1]["metrics"]), flush=True)
        del fwd, density, tabs
        if device.type == "cuda":
            torch.cuda.empty_cache()
    assessment = assess_validation(results, wavelengths, orders, args.target_tx, args.target_ty,
                                   args.validation_order_tolerance, pass_pol=args.pass_pol)
    connectivity = solid_connectivity(mask)
    assessment.update(step=step, threshold=item["threshold"], solid_connectivity=connectivity,
                      closed_air_voxels=int(closed_air(mask).sum()),
                      binary_sha256=hashlib.sha256(mask.tobytes()).hexdigest(), results=results)
    assessment["accepted"] &= connectivity["floating_voxels"] == 0 and assessment["closed_air_voxels"] == 0
    (folder / "assessment.json").write_text(json.dumps(assessment, indent=2), encoding="utf-8")
    return assessment


def persist_best(out, item, theta, beta, step, wavelengths):
    best = {k: v for k, v in item.items() if k not in ("stack", "tabs")}
    best["step"] = step
    np.savez_compressed(out / "best_binary.npz", stack=item["stack"], tabs=item["tabs"],
                        wavelengths_nm=np.array(wavelengths), threshold=item["threshold"], step=step)
    torch.save(dict(theta=theta.detach().cpu(), beta=beta, step=step, threshold=item["threshold"]), out / "best_state.pt")
    (out / "best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    report_task(out, step, "best_mask_images", save_mask_images, out, item["stack"], "best_binary")
    return best


def best_projected_density(out, best, args, sigma_px):
    snapshot = out / 'snapshots' / f"step_{best['step']:04d}.npz"
    if snapshot.exists():
        with np.load(snapshot) as saved:
            return saved['density'], float(saved['beta'])
    state = torch.load(out / 'best_state.pt', map_location='cpu', weights_only=False)
    if state['step'] != best['step']:
        raise ValueError('Best state and best metrics refer to different candidates')
    with torch.no_grad():
        density = torch.stack(make_rhos(state['theta'], state['beta'], sigma_px, args.eta,
            args.sigma_z_nm / args.layer_height_nm, args.z_filter_boundary)).numpy()
    return density, float(state['beta'])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pass-pol", choices=("x", "y"), default="y")
    ap.add_argument("--init-optimizer", action="store_true", help="Use rotated Adam moments from the exported seed")
    ap.add_argument("--continuation-offset", type=int, default=0)
    ap.add_argument("--wavelengths", default="600,650,700")
    ap.add_argument("--period-um", type=float, default=4.0)
    ap.add_argument("--grid", type=int, default=27)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--layer-height-nm", type=float, default=150.0)
    ap.add_argument("--detector-gap-um", type=float, default=0.35)
    ap.add_argument("--orders", type=int, default=2)
    ap.add_argument("--detector-samples", type=int, default=16)
    ap.add_argument("--reach", type=int, default=1)
    ap.add_argument("--support-mode", choices=("local_grounded", "posts_vented"), default="posts_vented")
    ap.add_argument("--sigma-nm", type=float, default=75.0)
    ap.add_argument("--sigma-z-nm", type=float, default=0.)
    ap.add_argument("--z-filter-boundary", choices=("replicate", "domain_normalized"), default="replicate")
    ap.add_argument("--binary-evaluation", choices=("supported_projection", "filtered_threshold"), default="supported_projection")
    ap.add_argument("--gray-plateau-beta", action="store_true")
    ap.add_argument("--binary-forward-ste", action="store_true",
                    help="Exact filtered-threshold optical forward; surrogate projection/filter backward")
    ap.add_argument("--gray-penalty-max", type=float, default=0.)
    ap.add_argument("--gray-penalty-start", type=int, default=5)
    ap.add_argument("--gray-penalty-ramp", type=int, default=20)
    ap.add_argument("--memory-efficient", action="store_true")
    ap.add_argument("--array-save-every", type=int, default=1)
    ap.add_argument("--compact-snapshots", action="store_true",
                    help="Retain hard masks every evaluation; full densities only at configured milestones")
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--beta0", type=float, default=2.0)
    ap.add_argument("--betamax", type=float, default=96.0)
    ap.add_argument("--beta-step", type=float, default=1.35)
    ap.add_argument("--beta-every", type=int, default=8)
    ap.add_argument("--adaptive-beta", action="store_true")
    ap.add_argument("--binary-first-step", type=int, default=None)
    ap.add_argument("--allow-beta-policy-migration", action="store_true",
                    help="Record an explicit beta-policy-only change on checkpoint resume")
    ap.add_argument("--live-report", action="store_true")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--max-updates", type=int, default=None,
                    help="Evaluate step 0 through this step, without an unevaluated extra update")
    ap.add_argument("--lr", type=float, default=0.04)
    ap.add_argument("--lr-final", type=float, default=None)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--noise", type=float, default=0.35)
    ap.add_argument("--init-mode", choices=("random", "x_stripes", "y_stripes", "checker"), default="random")
    ap.add_argument("--init-correlation-nm", type=float, default=0.,
                    help="Random block size for initialization only; fine voxels remain independent")
    ap.add_argument("--stripe-period-px", type=int, default=4)
    ap.add_argument("--stripe-fill-px", type=int, default=1)
    ap.add_argument("--init-state", default="")
    ap.add_argument("--init-binary", default="")
    ap.add_argument("--init-flip-layers", action="store_true")
    ap.add_argument("--init-binary-contrast", type=float, default=0.08)
    ap.add_argument("--init-beta-override", type=float, default=None)
    ap.add_argument("--reference-binary", default="")
    ap.add_argument("--objective-mode", choices=("legacy", "softmin", "legacy_then_softmin"), default="legacy")
    ap.add_argument("--softmin-start-step", type=int, default=203)
    ap.add_argument("--gradient-spike-limit", type=float, default=0., help="Pause before update on norm outlier; 0 disables")
    ap.add_argument("--softmin-tau", type=float, default=.03)
    ap.add_argument("--worst-tx-penalty", type=float, default=2.)
    ap.add_argument("--worst-ty-penalty", type=float, default=2.)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--plot-every", type=int, default=5)
    ap.add_argument("--snapshot-every", type=int, default=10)
    ap.add_argument("--hard-start-step", type=int, default=-1)
    ap.add_argument("--hard-full-step", type=int, default=80)
    ap.add_argument("--hard-threshold", type=float, default=0.5)
    ap.add_argument("--thresholds", default="0.35,0.4,0.45,0.5")
    ap.add_argument("--w-delta", type=float, default=2.0)
    ap.add_argument("--w-tx", type=float, default=1.0)
    ap.add_argument("--w-ty", type=float, default=1.0)
    ap.add_argument("--target-tx", type=float, default=0.70)
    ap.add_argument("--target-ty", type=float, default=0.25)
    ap.add_argument("--w-tx-floor", type=float, default=4.0)
    ap.add_argument("--w-ty-ceiling", type=float, default=4.0)
    ap.add_argument("--worst-weight", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=12)
    ap.add_argument("--dtype", choices=("complex64", "complex128"), default="complex64")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eig-backend", choices=("auto", "cpu", "gpu"), default="auto")
    ap.add_argument("--cache-eigenpairs", action="store_true",
                    help="Store forward eigenpairs on CPU for the same checkpoint's backward replay")
    ap.add_argument("--init-global-step", type=int, default=0,
                    help="Continue global step numbering from init-state, with fresh Adam moments")
    ap.add_argument("--holdout-wavelengths", default="")
    ap.add_argument("--holdout-every", type=int, default=25)
    ap.add_argument("--eig-cpu-workers", type=int, default=4)
    ap.add_argument("--cpu-threads", type=int, default=None)
    ap.add_argument("--spectral-progress", action="store_true")
    ap.add_argument("--rayleigh-nudge", type=float, default=1.0e-4)
    ap.add_argument("--early-stop", action="store_true")
    ap.add_argument("--validate-final", action="store_true")
    ap.add_argument("--validation-wavelengths", default="600,610,620,630,640,650,660,670,680,690,700")
    ap.add_argument("--validation-orders", default="12,14")
    ap.add_argument("--validation-order-tolerance", type=float, default=0.01)
    args = ap.parse_args()
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Refusing a silent CPU fallback for the full run.")
    if args.init_global_step < 0 or (args.init_global_step and not args.init_state):
        ap.error("init-global-step must be nonnegative and requires init-state")
    if args.cache_eigenpairs and not args.memory_efficient:
        ap.error("cache-eigenpairs requires memory-efficient spectral checkpoints")
    if args.holdout_every < 1:
        ap.error("holdout-every must be positive")
    if args.eig_cpu_workers < 1 or (args.cpu_threads is not None and args.cpu_threads < 1):
        ap.error("CPU worker and thread counts must be positive")
    if args.max_updates is not None:
        if args.max_updates < 0:
            ap.error("max-updates must be nonnegative")
        args.steps = args.max_updates + 1
    if args.lr_final is not None and (args.max_updates is None or args.lr_final <= 0):
        ap.error("lr-final requires max-updates and a positive learning rate")
    validation_orders = [int(o) for o in args.validation_orders.split(",")]
    if (args.early_stop or args.validate_final) and (len(set(validation_orders)) < 2 or min(validation_orders) <= args.orders):
        ap.error("Validation requires two distinct orders above the training order")
    if min(args.eval_every, args.plot_every, args.snapshot_every, args.beta_every, args.array_save_every) < 1:
        ap.error("save/evaluation intervals must be positive")
    if args.init_state and args.init_binary:
        ap.error("Choose init-state or init-binary, not both")
    if not math.isfinite(args.init_correlation_nm) or args.init_correlation_nm < 0:
        ap.error("init-correlation-nm must be finite and nonnegative")
    if args.init_correlation_nm and (args.init_mode != "random" or args.init_state or args.init_binary):
        ap.error("Correlated initialization requires a fresh random seed, not a parent or patterned init")
    if args.hard_start_step >= 0 and args.hard_full_step <= args.hard_start_step:
        ap.error("hard-full-step must exceed hard-start-step")
    if args.adaptive_beta and args.thresholds != str(args.hard_threshold):
        ap.error("Adaptive beta requires one fixed hard threshold for comparable binary metrics")
    if not math.isfinite(args.softmin_tau) or args.softmin_tau <= 0 or min(args.worst_tx_penalty, args.worst_ty_penalty) < 0:
        ap.error("Invalid soft-min temperature or penalty weight")
    if args.binary_first_step is not None and (not args.adaptive_beta or args.binary_first_step < 0
                                               or args.binary_first_step != args.hard_full_step):
        ap.error("binary-first-step must match hard-full-step and requires adaptive-beta")
    if args.gray_plateau_beta and (args.adaptive_beta or args.hard_start_step >= 0):
        ap.error("Gray plateau beta requires gray training (no adaptive-beta or hard STE)")
    if args.binary_evaluation == "filtered_threshold" and (args.adaptive_beta or args.hard_start_step >= 0
            or args.support_mode != "posts_vented" or args.thresholds != str(args.hard_threshold)):
        ap.error("Filtered threshold evaluation requires gray-only training and one fixed threshold")
    if args.binary_forward_ste and (args.binary_evaluation != "filtered_threshold" or args.gray_plateau_beta
            or args.adaptive_beta or args.hard_start_step >= 0 or args.objective_mode == "legacy_then_softmin"
            or args.beta_step != 1.):
        ap.error("Binary-forward STE requires filtered_threshold, fixed objective, and no adaptive beta/hard ramp")
    if (not math.isfinite(args.gray_penalty_max) or args.gray_penalty_max < 0
            or args.gray_penalty_start < 0 or args.gray_penalty_ramp < 1
            or (args.gray_penalty_max and not args.binary_forward_ste)):
        ap.error("Gray penalty requires binary-forward STE and finite nonnegative weights/schedule")
    if args.sigma_z_nm and not (args.gray_plateau_beta or args.binary_forward_ste):
        ap.error("3D blur currently requires gray-plateau-beta")
    if args.objective_mode == "legacy_then_softmin" and (not args.gray_plateau_beta or args.softmin_start_step < 1):
        ap.error("Staged objective requires gray-plateau-beta and a positive transition step")
    if not math.isfinite(args.gradient_spike_limit) or args.gradient_spike_limit < 0:
        ap.error("gradient-spike-limit must be finite and nonnegative")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # An explicit per-run opt-in preserves legacy runs and checkpoint commands.
    from voxel_binary_continuation import BinaryContinuation, BinaryContinuationRules
    binary_policy_path = out / "binary_continuation.json"
    binary_rules = None
    if binary_policy_path.exists():
        binary_rules = BinaryContinuationRules(**json.loads(binary_policy_path.read_text(encoding="utf-8")))
        if (not args.gray_plateau_beta or args.adaptive_beta or args.hard_start_step >= 0
                or args.binary_evaluation != "filtered_threshold"
                or binary_rules.check_every != args.eval_every or binary_rules.max_beta > args.betamax):
            raise ValueError("Binary continuation requires filtered gray training and matching binary cadence/cap")
    (out / "snapshots").mkdir(exist_ok=True)
    if (args.init_state or args.init_binary) and (out / "history.json").exists():
        raise ValueError("Use a new output directory for init-state; existing history must not be overwritten")
    wavelengths = [float(x) for x in args.wavelengths.split(",") if x.strip()]
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
    eigmod.EigConfig.backend = (("gpu" if device.type == "cuda" else "cpu")
                                if args.eig_backend == "auto" else args.eig_backend)
    eigmod.EigConfig.cpu_workers = args.eig_cpu_workers
    eigmod.WarmStart.enabled = False
    print(f"[compute] device={device} eig={eigmod.EigConfig.backend} "
          f"cpu_threads={torch.get_num_threads()} eig_workers={args.eig_cpu_workers}", flush=True)

    stack, pupil, rays, pitch_nm = build_stack(args)
    sigma_px = args.sigma_nm / pitch_nm
    dtype = torch.complex128 if args.dtype == "complex128" else torch.complex64
    fwd = FastPupilForward(
        stack=stack,
        pupil_spec=pupil,
        wavelengths_nm=wavelengths,
        rays=rays,
        device=device,
        chunk=args.chunk,
        dtype=dtype,
        verbose_guard=False,
    )
    fwd.checkpoint_finish = True
    progress_state = {}

    def spectral_progress(event, tag, phase, wls):
        detail = f"{tag} {','.join(str(int(w)) for w in wls)} nm {phase} {event}"
        print(f"[spectrum] step={progress_state.get('step')} {detail}", flush=True)
        try:
            atomic_json(out / "iteration_status.json", dict(progress_state, detail=detail, observed_at=time.time()))
        except OSError as exc:
            print(f"[progress-warning] {exc}", flush=True)

    if args.memory_efficient:
        fwd.spectral_children = []
        for wl in wavelengths:
            child = FastPupilForward(stack=stack, pupil_spec=pupil, wavelengths_nm=[wl], rays=rays,
                                     device=device, chunk=1, dtype=dtype, verbose_guard=False)
            child.checkpoint_layers = True
            child.cache_eigenpairs = args.cache_eigenpairs
            if args.spectral_progress:
                child.progress_callback = spectral_progress
            fwd.spectral_children.append(child)

    config = dict(vars(args), actual_pixel_nm=pitch_nm, device=str(device), voxel_src=str(VOXEL_SRC),
                  material_fourier="exact_rectangular_pixel_integrals",
                  optimizer_moments_offloaded=bool(args.memory_efficient),
                  release_finished_step_tensors=True)
    if (out / "objective_branch.json").exists():
        config["objective_branch"] = json.loads((out / "objective_branch.json").read_text(encoding="utf-8"))
    beta_rules = BetaRules(max_beta=args.betamax, binary_first_step=args.binary_first_step)
    if args.adaptive_beta:
        config["adaptive_beta_rules"] = AdaptiveBeta(beta_rules).config()
    if binary_rules:
        from dataclasses import asdict
        config["binary_continuation_rules"] = asdict(binary_rules)
    previous_config = out / "config.json"
    if (out / "state.pt").exists() and previous_config.exists() and not (args.init_state or args.init_binary):
        previous = json.loads(previous_config.read_text(encoding="utf-8"))
        if previous.get("binary_continuation_rules") is not None and previous["binary_continuation_rules"] != config.get("binary_continuation_rules"):
            raise ValueError("Changed physics/objective or binary continuation policy; explicit migration required")
        for key, value in dict(objective_mode="legacy", softmin_tau=.03, worst_tx_penalty=2., worst_ty_penalty=2.).items():
            previous.setdefault(key, value)
        for key, value in dict(sigma_z_nm=0., gray_plateau_beta=False, memory_efficient=False, array_save_every=1).items():
            previous.setdefault(key, value)
        previous.setdefault("init_correlation_nm", 0.)
        previous.setdefault("softmin_start_step", 203)
        previous.setdefault("gradient_spike_limit", 0.)
        previous.setdefault("z_filter_boundary", "replicate")
        previous.setdefault("binary_evaluation", "supported_projection")
        previous.setdefault("cache_eigenpairs", False)
        previous.setdefault("holdout_wavelengths", "")
        previous.setdefault("holdout_every", 25)
        for key, value in dict(binary_forward_ste=False, gray_penalty_max=0., gray_penalty_start=5,
                               gray_penalty_ramp=20).items():
            previous.setdefault(key, value)
        fixed = ("pass_pol", "continuation_offset", "period_um", "grid", "layers", "layer_height_nm", "wavelengths", "detector_gap_um", "orders",
                 "material_fourier", "support_mode", "reach", "sigma_nm", "eta", "dtype", "rayleigh_nudge",
                 "w_delta", "w_tx", "w_ty", "target_tx", "target_ty", "w_tx_floor", "w_ty_ceiling", "worst_weight",
                 "objective_mode", "softmin_tau", "worst_tx_penalty", "worst_ty_penalty",
                 "sigma_z_nm", "gray_plateau_beta", "memory_efficient", "array_save_every", "init_correlation_nm",
                 "softmin_start_step", "gradient_spike_limit", "z_filter_boundary", "binary_evaluation",
                 "cache_eigenpairs", "holdout_wavelengths", "holdout_every",
                 "binary_forward_ste", "gray_penalty_max", "gray_penalty_start", "gray_penalty_ramp")
        if args.adaptive_beta or previous.get("adaptive_beta"):
            fixed += ("adaptive_beta", "adaptive_beta_rules", "hard_threshold", "hard_start_step", "hard_full_step", "thresholds")
        changed = [key for key in fixed if previous.get(key) != config.get(key)]
        if changed == ["adaptive_beta_rules"] and args.allow_beta_policy_migration:
            old_rules = previous["adaptive_beta_rules"]
            new_rules = config["adaptive_beta_rules"]
            legacy_rules = {key: value for key, value in new_rules.items()
                            if key not in ("binary_first_step", "binary_growth", "post_increase_steps")}
            if old_rules != legacy_rules or args.binary_first_step is None:
                raise ValueError("Only legacy-to-binary-first beta policy migration is supported")
            backup = out / "config_before_binary_first.json"
            if backup.exists():
                raise FileExistsError("Migration backup already exists")
            backup.write_text(json.dumps(previous, indent=2), encoding="utf-8")
            atomic_json(out / "beta_policy_migration.json", dict(old_rules=old_rules, new_rules=new_rules,
                        reason="User approved binary-first continuation after step 200",
                        physics_and_objective_unchanged=True, gray_penalty_enabled=False))
            changed = []
        if changed:
            raise ValueError(f"Changed physics/objective cannot share one history: {changed}. Use a new out directory and init-state.")
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    st_path = out / "state.pt"
    st = {}
    if args.init_binary:
        parent = np.load(args.init_binary)["stack"]
        if args.init_flip_layers:
            parent = parent[::-1].copy()
        if not np.isin(parent, [0, 1]).all():
            raise ValueError("init-binary must contain a binary stack")
        theta = (args.eta + args.init_binary_contrast * (2 * torch.as_tensor(parent, dtype=torch.float64, device=device) - 1)).clamp(0, 1)
        m, v = torch.zeros_like(theta), torch.zeros_like(theta)
        beta, start = args.beta0, 0
    elif args.init_state:
        st = torch.load(args.init_state, map_location=device)
        theta = st["theta"].to(device=device, dtype=torch.float64)
        m = st["m"].to(device) if args.init_optimizer else torch.zeros_like(theta)
        v = st["v"].to(device) if args.init_optimizer else torch.zeros_like(theta)
        beta = float(st.get("beta", args.beta0))
        if args.init_beta_override is not None:
            beta = args.init_beta_override
        start = args.init_global_step
        # A changed wavelength objective starts a new optimizer, not stale Adam bias correction.
        st = {"adam_updates": int(st["adam_updates"])} if args.init_optimizer else {}
    elif st_path.exists():
        st = torch.load(st_path, map_location=device)
        theta = st["theta"].to(device)
        m = st["m"].to(device)
        v = st["v"].to(device)
        beta = float(st["beta"])
        start = int(st.get("updates_completed", int(st["step"]) + 1))
    elif args.init_correlation_nm > 0:
        if args.init_mode != "random":
            raise ValueError("Correlated initialization requires init-mode=random")
        theta = correlated_random_theta((args.layers, args.grid, args.grid), pitch_nm=pitch_nm,
            layer_height_nm=args.layer_height_nm, block_nm=args.init_correlation_nm,
            seed=args.seed, noise=args.noise).to(device)
        m, v = torch.zeros_like(theta), torch.zeros_like(theta)
        beta, start = args.beta0, 0
    else:
        gen = torch.Generator().manual_seed(args.seed)
        if args.init_mode == "random":
            base = torch.full((args.layers, args.grid, args.grid), 0.5, dtype=torch.float64)
        else:
            x = torch.arange(args.grid, dtype=torch.int64)
            y = torch.arange(args.grid, dtype=torch.int64)
            xx, yy = torch.meshgrid(x, y, indexing="ij")
            if args.init_mode == "x_stripes":
                mask = ((xx % args.stripe_period_px) < args.stripe_fill_px).to(torch.float64)
            elif args.init_mode == "y_stripes":
                mask = ((yy % args.stripe_period_px) < args.stripe_fill_px).to(torch.float64)
            else:
                mask = (((xx // max(1, args.stripe_fill_px)) + (yy // max(1, args.stripe_fill_px))) % 2 == 0).to(torch.float64)
            base = (0.2 + 0.6 * mask)[None].repeat(args.layers, 1, 1)
        theta = (base + args.noise * (2 * torch.rand((args.layers, args.grid, args.grid), generator=gen, dtype=torch.float64) - 1)).clamp(0, 1).to(device)
        m = torch.zeros_like(theta)
        v = torch.zeros_like(theta)
        beta = args.beta0
        start = 0

    if tuple(theta.shape) != (args.layers, args.grid, args.grid):
        raise ValueError("Parent density shape does not match the requested geometry")
    if start == 0 or args.init_state:
        torch.save({"theta": theta.detach().cpu(), "beta": beta, "step": start}, out / "initial_state.pt")
    records: list[dict] = []
    if (out / "history.json").exists() and start > 0:
        records = json.loads((out / "history.json").read_text(encoding="utf-8"))
        records = [r for r in records if r["step"] < start]

    best: dict = {}
    if (out / "best.json").exists():
        best = json.loads((out / "best.json").read_text(encoding="utf-8"))

    if args.reference_binary and start == 0:
        reference = np.load(args.reference_binary)["stack"]
        with torch.no_grad():
            rb = torch.as_tensor(reference, dtype=torch.float64, device=device)
            rtabs = polarized_tabs(fwd, rb[None], tags=("reference",), grad=[False])[0]
        (out / "reference_eval.json").write_text(json.dumps({
            "path": args.reference_binary, "orders": args.orders,
            "metrics": metrics_from_tabs(rtabs, args.pass_pol), "spectrum": spectrum_rows(wavelengths, rtabs, args.pass_pol),
            "unsupported_bottom_up": unsupported_voxels(rb, args.reach),
            "solid_connectivity": solid_connectivity(reference),
        }, indent=2), encoding="utf-8")
        print("[reference] " + json.dumps(metrics_from_tabs(rtabs, args.pass_pol)), flush=True)

    accepted = None
    controller = AdaptiveBeta(beta_rules, st.get("adaptive_beta_state")) if args.adaptive_beta else None
    beta_events = st.get("beta_events", [])
    beta_guard = st.get("beta_guard")
    binary_controller = BinaryContinuation(binary_rules, st.get("binary_continuation_state")) if binary_rules else None
    if binary_controller and not (out / "binary_continuation_migration.json").exists():
        atomic_json(out / "binary_continuation_migration.json", dict(
            start_step=start, rules=config["binary_continuation_rules"],
            reason="User requested binary-aware continuation; gray-only plateau delayed binarization",
            theta_adam_preserved=True, physics_and_objective_unchanged=True,
            hard_threshold=args.hard_threshold, gray_penalty_enabled=False, ste_enabled=False))
    adam_updates = 0 if args.init_state and not args.init_optimizer else int(st.get("adam_updates", start))
    # Keep resume metadata, but not obsolete GPU copies after Adam replaces these tensors.
    for key in ("theta", "m", "v"):
        st.pop(key, None)
    print(f"[resume] evaluated_next={start} adam_updates={adam_updates} beta={beta:g} "
          f"binary_first_step={args.binary_first_step}", flush=True)
    for step in range(start, args.steps):
        t0 = time.time()
        phase_args = active_objective_args(args, step)
        new_phase = phase_changed(records, phase_args.objective_mode)
        phase_records = records
        if new_phase:
            archive = out / "objective_phases" / f"{records[-1]['objective_mode']}_through{records[-1]['step']:04d}"
            archive.mkdir(parents=True, exist_ok=True)
            for name in ("state.pt", "best.json", "best_state.pt", "best_binary.npz", "history.json"):
                if (out/name).exists() and not (archive/name).exists():
                    shutil.copy2(out/name, archive/name)
            m, v = torch.zeros_like(theta), torch.zeros_like(theta)
            adam_updates = 0
            best = {}
            atomic_json(out / "best.json", {})
            phase_records = []
            atomic_json(out / "objective_transition.json", dict(step=step, mode=phase_args.objective_mode,
                adam_moments_reset=True, beta_kept=beta, archived=str(archive), best_reset=True))
            print(f"[objective-transition] step={step} mode={phase_args.objective_mode}; reset Adam/phase ranking", flush=True)
            if binary_controller:
                beta_events.append(binary_controller.phase_reset(step, beta))
        progress_state.update(step=step, phase="forward_gradient", started_at=t0,
                              last_evaluated_step=records[-1]["step"] if records else None)
        atomic_json(out / "iteration_status.json", dict(step=step, phase="forward_gradient",
                    started_at=time.time(), last_evaluated_step=records[-1]["step"] if records else None))
        print(f"[step-start] {step} gray forward/gradient", flush=True)
        beta_already_applied = st.get("beta_applied_step", -1) if step == start and st_path.exists() and not (args.init_state or args.init_binary) else -1
        if binary_controller is not None:
            decision = (binary_controller.decide(phase_records, beta, step)
                        if beta_already_applied != step and (args.max_updates is None or step < args.max_updates) else None)
            if decision:
                if (decision["action"] == "increase" and args.objective_mode == "legacy_then_softmin"
                        and 0 < args.softmin_start_step - step <= binary_rules.min_residence + binary_rules.check_every):
                    decision.update(action="hold", applied_beta=beta, reason="objective_transition_before_guard_finishes")
                if decision["action"] == "increase":
                    anchor = out / "binary_beta_anchors" / f"step_{step:04d}.pt"
                    anchor.parent.mkdir(exist_ok=True)
                    if not anchor.exists():
                        torch.save(dict(theta=theta.detach().cpu(), m=m.detach().cpu(), v=v.detach().cpu(),
                                        adam_updates=adam_updates, beta=beta, step=step), anchor.with_suffix(".tmp"))
                        anchor.with_suffix(".tmp").replace(anchor)
                    binary_controller.begin_trial(decision, anchor)
                elif decision["action"].startswith("restore_"):
                    anchor_state = torch.load(decision["anchor"], map_location="cpu", weights_only=False)
                    theta = anchor_state["theta"].to(device)
                    m, v = anchor_state["m"], anchor_state["v"]
                    adam_updates = anchor_state["adam_updates"]
                    del anchor_state
                beta = decision["applied_beta"]
                beta_events.append(decision)
                atomic_json(out / "binary_continuation_status.json", dict(
                    event=decision, state=binary_controller.state, observed_at=time.time()))
                print("[binary-continuation] " + json.dumps(decision), flush=True)
        elif controller is not None:
            if beta_guard is not None:
                audit = controller.check_post_increase(beta_guard, records, step)
                if audit is not None:
                    if audit["action"].startswith("restore_"):
                        theta = beta_guard["theta"].to(device).clone()
                        m = beta_guard["m"].to(device).clone()
                        v = beta_guard["v"].to(device).clone()
                        adam_updates = beta_guard["adam_updates"]
                        beta = audit["applied_beta"]
                        if audit["action"] == "restore_and_reduce":
                            beta_guard.update(start_step=step, new_beta=beta, attempt=beta_guard["attempt"]+1)
                        else:
                            beta_guard = None
                            atomic_json(out / "gray_penalty_review.json", audit)
                    else:
                        beta_guard = None
                    beta_events.append(audit)
                    print("[post-beta-audit] " + json.dumps(audit), flush=True)
            decision = (controller.decide(records, beta, step)
                        if beta_guard is None and (args.max_updates is None or step < args.max_updates) else None)
            if decision:
                if decision["action"] == "probe":
                    print(f"[beta-probe] step={step} old={beta:g}", flush=True)
                    version = theta._version
                    beta, decision = controller.search(decision, lambda value: probe_beta(theta, value, sigma_px, args, fwd))
                    assert theta._version == version, "Beta probes must not modify theta"
                    if decision.get("post_update_guard_required"):
                        beta_guard = dict(theta=theta.detach().cpu().clone(), m=m.detach().cpu().clone(),
                                          v=v.detach().cpu().clone(), adam_updates=adam_updates,
                                          old_beta=decision["old_beta"], new_beta=beta, start_step=step, attempt=1,
                                          baseline=decision["baseline"]["binary"])
                beta_events.append(decision)
                print("[adaptive-beta] " + json.dumps(decision), flush=True)
        elif args.gray_plateau_beta:
            decision = (gray_plateau_decision(phase_records, beta, step, max_beta=args.betamax)
                        if args.max_updates is None or step < args.max_updates else None)
            if decision and beta_already_applied != step:
                beta = decision["applied_beta"]
                beta_events.append(decision)
                print("[gray-beta] " + json.dumps(decision), flush=True)
        elif step > 0 and step % args.beta_every == 0 and beta_already_applied != step:
            beta = min(args.betamax, beta * args.beta_step)

        hard_weight = 0.0
        if args.hard_start_step >= 0:
            hard_weight = min(1.0, max(0.0, (step - args.hard_start_step) / (args.hard_full_step - args.hard_start_step)))
        if args.binary_forward_ste:
            hard_weight = 1.0
        will_update = args.max_updates is None or step < args.max_updates
        # Adam is idle during the optical solve. Transfers preserve dtype and values.
        if args.memory_efficient:
            m, v = m.cpu(), v.cpu()
        step_memory = {}
        if device.type == "cuda":
            if args.memory_efficient:
                torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            step_memory = dict(cuda_allocated_before_forward_gib=torch.cuda.memory_allocated(device)/2**30,
                               cuda_reserved_before_forward_gib=torch.cuda.memory_reserved(device)/2**30)
        th = theta.clone().requires_grad_(will_update)
        design_version = th._version
        filtered = filter_density(th, sigma_px, args.sigma_z_nm/args.layer_height_nm, args.z_filter_boundary)
        rhos = list(project(filtered, beta, args.eta).clamp(0, 1).unbind(0))
        dens = gate_grey(rhos, args.reach, args.support_mode)
        if step == 0:
            initial = dens.detach()
            atomic_json(out / "initialization_stats.json", dict(
                seed=args.seed, random_block_nm=args.init_correlation_nm,
                shape=list(theta.shape), independent_fine_voxel_parameters=True,
                theta_mean=float(th.detach().mean()), theta_std=float(th.detach().std(correction=0)),
                density_mean=float(initial.mean()), density_std=float(initial.std(correction=0)),
                density_min=float(initial.min()), density_max=float(initial.max()),
                sigma_xy_nm=args.sigma_nm, sigma_z_nm=args.sigma_z_nm, beta=beta))
            del initial
        with torch.set_grad_enabled(will_update and hard_weight < 1):
            tabs = polarized_tabs(fwd, dens[None], tags=("grey",), grad=[will_update and hard_weight < 1])[0]
        fom, tensors = objective_from_tabs(tabs, **objective_kwargs(phase_args))
        hard_tabs = None
        optimization_fom = fom
        if hard_weight > 0 or args.adaptive_beta:
            if args.binary_forward_ste:
                ste_density, hard = binary_forward_ste(dens, filtered, args.hard_threshold)
            else:
                hard = gate_binary(rhos, args.reach, args.hard_threshold, args.support_mode)
                ste_density = dens + (hard - dens).detach()
            with torch.set_grad_enabled(will_update and hard_weight > 0):
                hard_tabs = polarized_tabs(fwd, ste_density[None], tags=("hard_ste",), grad=[will_update and hard_weight > 0])[0]
            hard_fom, _ = objective_from_tabs(hard_tabs, **objective_kwargs(phase_args))
            optimization_fom = hard_fom if hard_weight >= 1 else (1 - hard_weight) * fom + hard_weight * hard_fom
        penalty_weight = gray_penalty_weight(step + args.continuation_offset, args.gray_penalty_max,
                                            args.gray_penalty_start, args.gray_penalty_ramp)
        if penalty_weight:
            optimization_fom = optimization_fom - penalty_weight * (4*dens*(1-dens)).mean()
        if will_update:
            (-optimization_fom).backward()
            grad = th.grad.detach()
        else:
            grad = torch.zeros_like(th)
        if not bool(torch.isfinite(grad).all()):
            raise RuntimeError(f"Non-finite gradient at step {step}; optimizer update rejected")
        grad_norm = float(torch.linalg.vector_norm(grad).cpu())
        alarm = gradient_outlier(phase_records, grad_norm, phase_args.objective_mode, beta,
                                 factor=args.gradient_spike_limit)
        if alarm:
            atomic_json(out / "gradient_alarm.json", dict(alarm, step=step, beta=beta,
                objective_mode=phase_args.objective_mode, objective=float(optimization_fom.detach())))
            torch.save(dict(theta=th.detach().cpu(), gradient=grad.detach().cpu(), beta=beta, step=step,
                            gradient_convention="d(-F)/dtheta; before Adam update"), out / "gradient_alarm_state.pt")
            raise RuntimeError(f"Gradient outlier at step {step}; no optimizer update; see gradient_alarm.json")
        metrics = metrics_from_tabs(tabs, args.pass_pol)
        rec = {
            "step": step,
            "objective_mode": phase_args.objective_mode,
            "beta": beta,
            "fom": float(fom.detach().cpu()),
            **metrics,
            "spectrum": spectrum_rows(wavelengths, tabs, args.pass_pol),
            "grayness": float((4 * dens.detach() * (1 - dens.detach())).mean().cpu()),
            "gray_fraction": float(((dens.detach() > .1) & (dens.detach() < .9)).to(torch.float64).mean().cpu()),
            "gradient_norm": grad_norm,
            "hard_weight": hard_weight,
            "beta_policy": "fixed_beta_binary_forward_ste" if args.binary_forward_ste else "binary_aware_gray_continuation_v1" if binary_controller else "gray_plateau_1p5" if args.gray_plateau_beta else "binary_first" if controller and controller.binary_first(step) else "gray_and_binary",
            "beta_guard_pending": bool(binary_controller.state["pending"]) if binary_controller else beta_guard is not None,
            "gray_penalty_weight": penalty_weight,
            "binary_forward_ste": args.binary_forward_ste,
            "optimization_fom": float(optimization_fom.detach().cpu()),
            "hard_train_spectrum": spectrum_rows(wavelengths, hard_tabs, args.pass_pol) if hard_tabs is not None else None,
            "bin_fom": None,
            "bin_mean_tx": None,
            "bin_min_tx": None,
            "bin_mean_ty": None,
            "bin_max_ty": None,
            "bin_mean_delta": None,
            "bin_min_delta": None,
            "elapsed_s": time.time() - t0,
            **step_memory,
        }

        current_best = None
        target_candidates = []
        first_evaluation = not records
        holdout_due = bool(args.holdout_wavelengths) and (first_evaluation or step % args.holdout_every == 0 or step == args.steps - 1)
        snapshot_due = first_evaluation or step % args.snapshot_every == 0 or step == args.steps - 1 or new_phase
        if step % args.eval_every == 0 or snapshot_due or holdout_due or step == args.steps - 1:
            print(f"[binary-evaluation] step {step}", flush=True)
            progress_state.update(phase="binary_evaluation")
            atomic_json(out / "iteration_status.json", dict(step=step, phase="binary_evaluation", started_at=time.time()))
            with torch.no_grad():
                seen_masks = set()
                for thr in thresholds:
                    bstack = evaluation_binary(filtered, [r.detach() for r in rhos], args, thr)
                    mask_key = np.packbits(bstack.cpu().numpy().astype(np.uint8)).tobytes()
                    if mask_key in seen_masks:
                        continue
                    seen_masks.add(mask_key)
                    btabs = (hard_tabs.detach() if hard_tabs is not None and thr == args.hard_threshold
                             else polarized_tabs(fwd, bstack[None], tags=(f"bin{thr:g}",), grad=[False])[0])
                    bfom, _ = objective_from_tabs(btabs, **objective_kwargs(phase_args))
                    bm = metrics_from_tabs(btabs, args.pass_pol)
                    item = {
                        "threshold": thr,
                        "binary_evaluation": args.binary_evaluation,
                        "fabrication_certified": False,
                        "fom": float(bfom.cpu()),
                        "metrics": bm,
                        "spectrum": spectrum_rows(wavelengths, btabs, args.pass_pol),
                        "unsupported": unsupported_voxels(bstack, args.reach),
                        "solid_connectivity": solid_connectivity(bstack.detach().cpu().numpy()),
                        "stack": bstack.detach().cpu().numpy().astype(np.uint8),
                        "tabs": btabs.detach().cpu().numpy(),
                    }
                    if current_best is None or item["fom"] > current_best["fom"]:
                        current_best = item
                    if args.early_stop and target_reached(item["spectrum"], args.target_tx, args.target_ty, args.pass_pol):
                        target_candidates.append(item)

        # Do not miss a better hard forward between threshold-screen intervals.
        if hard_tabs is not None and (current_best is None or float(hard_fom.detach()) > current_best["fom"]):
            hard_array = hard.detach().cpu().numpy().astype(np.uint8)
            current_best = dict(threshold=args.hard_threshold, fom=float(hard_fom.detach()),
                                binary_evaluation=args.binary_evaluation, fabrication_certified=False,
                                metrics=metrics_from_tabs(hard_tabs, args.pass_pol), spectrum=spectrum_rows(wavelengths, hard_tabs, args.pass_pol),
                                unsupported=unsupported_voxels(hard, args.reach), solid_connectivity=solid_connectivity(hard_array),
                                stack=hard_array, tabs=hard_tabs.detach().cpu().numpy())
        if current_best is not None:
            current_best["objective_mode"] = phase_args.objective_mode
            rec["bin_threshold"] = current_best["threshold"]
            rec["bin_spectrum"] = current_best["spectrum"]
            rec["bin_fom"] = current_best["fom"]
            for key in ("mean_tx", "min_tx", "mean_ty", "max_ty", "mean_delta", "min_delta"):
                rec["bin_"+key] = current_best["metrics"][key]
            if phase_args.objective_mode == "softmin":
                rec["binary_objective_details"] = softmin_diagnostics(current_best["spectrum"], tau=args.softmin_tau,
                    target_tx=args.target_tx, target_ty=args.target_ty,
                    tx_penalty=args.worst_tx_penalty, ty_penalty=args.worst_ty_penalty, pass_pol=args.pass_pol)
            if not best or current_best["fom"] > best.get("fom", -1e99):
                best = persist_best(out, current_best, th, beta, step, wavelengths)

        if binary_controller:
            binary_controller.observe(rec)

        if holdout_due and current_best is not None:
            from voxel_holdout import evaluate_holdout
            progress_state.update(phase="holdout_evaluation")
            holdout = evaluate_holdout(args, current_best, step, device, spectral_progress)
            rec["holdout"] = holdout
            progress_state.update(phase="binary_evaluation")

        if args.early_stop and not target_candidates and hard_tabs is not None:
            hard_spec = spectrum_rows(wavelengths, hard_tabs, args.pass_pol)
            if target_reached(hard_spec, args.target_tx, args.target_ty, args.pass_pol):
                target_candidates.append(dict(stack=hard.detach().cpu().numpy().astype(np.uint8),
                                              threshold=args.hard_threshold, spectrum=hard_spec))
        for item in sorted(target_candidates, key=lambda c: c.get("fom", -float("inf")), reverse=True):
            candidate_assessment = validate_binary_candidate(out, item, step, args, device)
            if candidate_assessment["accepted"]:
                accepted = candidate_assessment
                np.savez_compressed(out / "accepted_binary.npz", stack=item["stack"], step=step, threshold=item["threshold"])
                torch.save(dict(theta=th.detach().cpu(), step=step, beta=beta, threshold=item["threshold"]),
                           out / "accepted_state.pt")
                save_snapshot_plot(out, step, dens.detach().cpu().numpy(), item["stack"], beta,
                                   item["threshold"], args.period_um, args.layer_height_nm,
                                   filename="accepted_structure_3d.png")
                (out / "accepted_validation.json").write_text(json.dumps(accepted, indent=2), encoding="utf-8")
                will_update = False
                break

        if will_update:
            assert th._version == design_version, "Forward/evaluation must not overwrite the design variable"
            with torch.no_grad():
                m, v = m.to(device), v.to(device)
                lr = (args.lr_final + .5*(args.lr-args.lr_final)*(1+math.cos(math.pi*step/max(1, args.max_updates)))
                      if args.lr_final is not None else args.lr)
                k = adam_updates + 1
                m = 0.9 * m + 0.1 * grad
                v = 0.999 * v + 0.001 * grad.square()
                theta = (theta - lr * (m / (1 - 0.9**k)) / ((v / (1 - 0.999**k)).sqrt() + 1e-12)).clamp(0, 1)
                adam_updates = k
                rec["learning_rate"] = lr
        if args.memory_efficient:
            m, v = m.cpu(), v.cpu()
        rec["update_applied"] = will_update
        rec["adam_updates"] = adam_updates

        records.append(rec)
        snapshot_arrays = {"theta": th.detach().cpu().numpy(), "density": dens.detach().cpu().numpy(),
                           "filtered_density": filtered.detach().cpu().numpy(),
                           "step": step, "beta": beta, "period_um": args.period_um,
                           "layer_height_nm": args.layer_height_nm,
                           "axis_order": "layer_top_to_bottom,x,y"}
        if current_best is not None:
            snapshot_arrays.update(binary=current_best["stack"], threshold=current_best["threshold"],
                                   tabs=current_best["tabs"])
        save_step_arrays(out, step, snapshot_arrays, compact=args.compact_snapshots,
            full_due=step % args.array_save_every == 0 or snapshot_due,
            has_binary=current_best is not None)
        if snapshot_due and current_best is not None:
            report_task(out, step, "snapshot_plot", save_snapshot_plot,
                               out / "snapshots", step, snapshot_arrays["density"], current_best["stack"],
                                beta, current_best["threshold"], args.period_um, args.layer_height_nm)
            report_task(out, step, "density_images", save_mask_images,
                        out / "snapshots", snapshot_arrays["density"], f"step_{step:04d}_density")
            report_task(out, step, "binary_images", save_mask_images,
                        out / "snapshots", current_best["stack"], f"step_{step:04d}_binary")
            if step % 50 == 0:
                report_task(out, step, "evolution_overview", save_evolution_overview,
                            out, args.period_um, args.layer_height_nm)
        rec["elapsed_s"] = time.time() - t0
        if device.type == "cuda":
            rec["cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated(device)/2**30
            rec["cuda_peak_reserved_gib"] = torch.cuda.max_memory_reserved(device)/2**30
        atomic_json(out / "history.json", records)
        torch.save({"theta": theta.detach(), "m": m.detach(), "v": v.detach(), "beta": beta, "step": step,
                    "updates_completed": step + int(will_update), "beta_applied_step": step,
                    "adam_updates": adam_updates, "beta_guard": beta_guard,
                    "adaptive_beta_state": controller.state if controller else None,
                    "binary_continuation_state": binary_controller.state if binary_controller else None,
                    "beta_events": beta_events}, st_path.with_suffix(".tmp"))
        st_path.with_suffix(".tmp").replace(st_path)
        if binary_controller:
            report_task(out, step, "binary_continuation_status", atomic_json,
                        out / "binary_continuation_status.json", dict(
                            read_json(out / "binary_continuation_status.json", {}),
                            state=binary_controller.state, evaluated_step=step, beta=beta,
                            observed_at=time.time()))
        atomic_json(out / "run_status.json", dict(status="accepted" if accepted else "running",
            evaluated_step=step, updates_completed=step+int(will_update), max_updates=args.max_updates,
            best_step=best.get("step"), accepted=bool(accepted)))
        atomic_json(out / "iteration_status.json", dict(step=step, phase="checkpoint_saved", started_at=time.time()))
        if controller or args.gray_plateau_beta:
            atomic_json(out / "beta_events.json", beta_events)
        report_task(out, step, "summary", write_summary, out, wavelengths, records, best, args)
        if first_evaluation or step % args.plot_every == 0 or step == args.steps - 1:
            report_task(out, step, "history_plots", save_history_plots, out, records, best, args.target_tx, args.target_ty)
            if args.live_report:
                report_task(out, step, "live_plots", refresh_live_plots, out, records, best, config)
        if args.live_report:
            report_task(out, step, "live_status", write_live_status, out, records, best, config)

        print(
            f"step {step:03d} beta {beta:.2g} F {rec['fom']:.4f} "
            f"TX {rec['mean_tx']:.3f}/{rec['min_tx']:.3f} "
            f"TY {rec['mean_ty']:.3f}/{rec['max_ty']:.3f} "
            f"d {rec['mean_delta']:.3f}/{rec['min_delta']:.3f} "
            f"bin {rec['bin_fom'] if rec['bin_fom'] is not None else '-'} "
            f"{rec['elapsed_s']:.1f}s",
            flush=True,
        )
        # Do not carry the previous design graph and evaluation buffers into the next solve.
        th = filtered = rhos = dens = tabs = fom = tensors = optimization_fom = grad = None
        hard_tabs = hard = ste_density = hard_fom = bstack = btabs = bfom = None
        if (out / "STOP_REQUESTED").exists():
            atomic_json(out / "run_status.json", dict(status="stopped_by_user", evaluated_step=step,
                        updates_completed=step+int(will_update), max_updates=args.max_updates, best_step=best.get("step")))
            print("[stopped] Checkpoint preserved; no automatic resume", flush=True)
            return
        if accepted:
            print(f"[early-stop] Verified binary met targets at step {step}", flush=True)
            break
        if will_update and recycle_due(read_json(out / "worker_recycle.json", {}), step,
                                       step + 1, args.max_updates):
            from watch_voxel_goal import request_checkpoint_reload
            if report_task(out, step, "worker_recycle_request", request_checkpoint_reload, out,
                           f"Bounded worker recycle after committed step {step}; optimizer state unchanged",
                           kind="worker_recycle"):
                atomic_json(out / "run_status.json", dict(status="checkpoint_recycle", evaluated_step=step,
                    updates_completed=step+1, max_updates=args.max_updates, best_step=best.get("step")))
                atomic_json(out / "iteration_status.json", dict(step=step, phase="checkpoint_recycle",
                    last_evaluated_step=step, observed_at=time.time()))
                print(f"[worker-recycle] Saved step {step}; verified goal watcher will resume step {step+1}", flush=True)
                return

    if best:
        print(f"[best] step={best['step']} thr={best['threshold']} F={best['fom']:.6f}", flush=True)
        best_density, best_beta = best_projected_density(out, best, args, sigma_px)
        with np.load(out / "best_binary.npz") as best_binary:
            report_task(out, best["step"], "best_structure_plot", save_snapshot_plot,
                        out, best["step"], best_density,
                        best_binary["stack"], best_beta,
                        best["threshold"], args.period_um, args.layer_height_nm, filename="best_structure_3d.png")
    print(f"[done] {out}", flush=True)
    report_task(out, records[-1]["step"], "history_plots", save_history_plots,
                out, records, best, args.target_tx, args.target_ty)
    report_task(out, records[-1]["step"], "evolution_overview", save_evolution_overview,
                out, args.period_um, args.layer_height_nm)
    if args.validate_final and best and not accepted:
        item = dict(stack=np.load(out / "best_binary.npz")["stack"], threshold=best["threshold"])
        assessment = validate_binary_candidate(out, item, best["step"], args, device)
        (out / "final_validation.json").write_text(json.dumps(assessment, indent=2), encoding="utf-8")
    (out / "run_status.json").write_text(json.dumps(dict(status="completed", stopped_for="verified_target" if accepted else "step_budget",
        evaluated_step=records[-1]["step"], updates_completed=records[-1]["step"]+int(records[-1].get("update_applied", True)),
        max_updates=args.max_updates, best_step=best.get("step"), early_stop_accepted=bool(accepted)), indent=2), encoding="utf-8")
    if args.live_report:
        report_task(out, records[-1]["step"], "live_plots", refresh_live_plots, out, records, best, config)
        report_task(out, records[-1]["step"], "live_status", write_live_status, out, records, best, config, finished=True)


if __name__ == "__main__":
    main()
