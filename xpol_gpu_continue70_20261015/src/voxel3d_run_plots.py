"""Scientific plots from saved RCWA optimization states (no reconstructed history)."""

from pathlib import Path
import json
from itertools import groupby

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np


def save_history_plots(out, records, best, target_tx, target_ty):
    out = Path(out)
    steps = [r["step"] for r in records]
    binary = [r for r in records if r.get("bin_fom") is not None]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    config = json.loads((out / "config.json").read_text()) if (out / "config.json").exists() else {}
    model = config.get("material_fourier", "legacy sampled FFT (not exact pixel integrals)")
    training_label = f"Periodic RCWA training | order {config.get('orders', '?')} | {config.get('wavelengths', '')} nm"
    fig.suptitle(training_label + "\n" + model, fontsize=10)
    for key, label, color in [("mean_tx", "Mean TX", "#087e8b"),
                              ("mean_ty", "Mean TY", "#bb3754")]:
        axes[0, 0].plot(steps, [r[key] for r in records], color=color, alpha=.55, label=label+" density")
        axes[0, 0].plot([r["step"] for r in binary], [r["bin_"+key] for r in binary],
                        "o-", ms=3, color=color, label=label+" binary")
    for key, label, color, target in [("min_tx", "Min TX", "#087e8b", target_tx),
                                    ("max_ty", "Max TY", "#bb3754", target_ty)]:
        axes[0, 1].plot(steps, [r[key] for r in records], color=color, alpha=.55, label=label+" density")
        axes[0, 1].plot([r["step"] for r in binary], [r["bin_"+key] for r in binary],
                        "o-", ms=3, color=color, label=label+" binary")
        axes[0, 1].axhline(target, color=color, ls=":", lw=1)
    hard_records = [r for r in records if r.get("hard_train_spectrum")]
    for key, label, color in [("tx", "TX", "#087e8b"), ("ty", "TY", "#bb3754")]:
        axes[0, 0].plot([r["step"] for r in hard_records],
                        [np.mean([s[key] for s in r["hard_train_spectrum"]]) for r in hard_records],
                        color=color, ls="--", lw=1, label=f"{label} fixed hard threshold")
    for mode, phase in groupby(records, key=lambda r: r.get("objective_mode", "legacy")):
        phase = list(phase)
        axes[1, 0].plot([r["step"] for r in phase], [r["fom"] for r in phase],
                        label=f"Density FoM ({mode})", color="#7768ae")
    if any(r.get("hard_weight", 0) > 0 for r in records):
        axes[1, 0].plot(steps, [r.get("optimization_fom", r["fom"]) for r in records],
                        label="Training FoM (density / hard STE)", color="#555555", ls="--")
    for mode, phase in groupby(binary, key=lambda r: r.get("objective_mode", "legacy")):
        phase = list(phase)
        axes[1, 0].plot([r["step"] for r in phase], [r["bin_fom"] for r in phase], "o-", ms=3,
                        label=f"Binary FoM ({mode})", color="#cf7927")
    axes[1, 1].plot(steps, [r.get("grayness", np.nan) for r in records], label="Mean 4*rho*(1-rho)", color="#555555")
    axes[1, 1].plot(steps, [r["min_delta"] for r in records], label="Min (TX-TY), density", color="#087e8b")
    axes[1, 1].plot([r["step"] for r in binary], [r["bin_min_delta"] for r in binary], "o-", ms=3, label="Min (TX-TY), binary", color="#bb3754")
    for ax, title in zip(axes.ravel(), ["Band-mean transmission", "Worst wavelength (dotted: soft targets)", "Optimization objective", "Binarization and selectivity"]):
        ax.set(title=title, xlabel="Step (evaluated before update)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 1].set_ylim(0, 1.05)
    fig.savefig(out / "efficiency_history.png", dpi=160)
    plt.close(fig)

    spectra = records[-1].get("spectrum", [])
    if spectra:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        for j, row in enumerate(spectra):
            color = plt.get_cmap("viridis")(j/max(1, len(spectra)-1))
            for ax, key in zip(axes, ["tx", "ty"]):
                spectral_records = [r for r in records if r.get("spectrum")]
                ax.plot([r["step"] for r in spectral_records], [r["spectrum"][j][key] for r in spectral_records], color=color, alpha=.45)
                valid = [r for r in binary if r.get("bin_spectrum")]
                ax.plot([r["step"] for r in valid], [r["bin_spectrum"][j][key] for r in valid], "o-", ms=3,
                        color=color, label=f"{row['wavelength_nm']:g} nm")
        for ax, title in zip(axes, ["TX per wavelength", "TY per wavelength"]):
            ax.set(title=title, xlabel="Step", ylabel="Transmission", ylim=(0, 1.05))
            ax.legend(); ax.grid(alpha=.2)
        fig.suptitle(training_label + "\nFaint lines: density; markers: evaluated binary mask", fontsize=10)
        fig.savefig(out / "wavelength_history.png", dpi=160)
        plt.close(fig)
    if best:
        spec = best["spectrum"]
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        for key, label, color in [("tx", "TX", "#087e8b"), ("ty", "TY", "#bb3754"), ("delta", "TX-TY", "#7768ae")]:
            ax.plot([s["wavelength_nm"] for s in spec], [s[key] for s in spec], "o-", label=label, color=color)
        lower = min(0., min(s["delta"] for s in spec))
        ax.set(xlabel="Wavelength (nm)", ylabel="Transmission / TX-TY",
               ylim=(lower - .03 if lower < 0 else 0., 1.05))
        ax.set_title(f"Best binary: step {best['step']}, threshold {best['threshold']:g}\n"
                     f"Periodic RCWA | training order {config.get('orders', '?')}", fontsize=10)
        ax.axhline(0, color=".5", lw=.8)
        ax.grid(alpha=.2); ax.legend()
        fig.savefig(out / "best_binary_spectrum.png", dpi=160)
        plt.close(fig)


def draw_volume(ax, density, period_um, layer_height_nm, binary=False):
    preview = density.size > 160000
    if preview:
        factors = [next(f for f in range(1, n+1) if n % f == 0 and n//f <= 40) for n in density.shape]
        z, x, y = density.shape
        fz, fx, fy = factors
        density = density.reshape(z//fz, fz, x//fx, fx, y//fy, fy).mean(axis=(1, 3, 5))
        layer_height_nm *= fz
    # Solver arrays are [top-to-bottom layer, x, y]; show z rising from substrate.
    volume = np.transpose(density[::-1], (1, 2, 0))
    nx, ny, nz = volume.shape
    xe, ye, ze = np.meshgrid(np.linspace(-period_um/2, period_um/2, nx+1),
                            np.linspace(-period_um/2, period_um/2, ny+1),
                            np.arange(nz+1)*layer_height_nm/1000, indexing="ij")
    colors = plt.get_cmap("Greys")(volume)
    colors[..., 3] = 0.12 + 0.88*volume
    if binary:
        colors[...] = [0.12, .48, .58, 1.0]
        if preview:
            colors[..., 3] = .12+.88*volume
    ax.voxels(xe, ye, ze, volume > .02, facecolors=colors, edgecolor=(.25, .25, .25, .16), linewidth=.1)
    ax.plot_surface(np.array([[-period_um/2, period_um/2]]*2),
                    np.array([[-period_um/2]*2, [period_um/2]*2]), np.zeros((2, 2)),
                    color="#b9c9e2", alpha=.25, shade=False)
    h = nz*layer_height_nm/1000
    ax.set(xlim=(-period_um/2, period_um/2), ylim=(-period_um/2, period_um/2), zlim=(0, h),
           xlabel="x (um)", ylabel="y (um)", zlabel="z (um)",
           xticks=(-period_um/2, 0, period_um/2), yticks=(-period_um/2, 0, period_um/2),
           zticks=(0, h))
    ax.grid(False)
    ax.set_box_aspect((period_um, period_um, h))
    ax.view_init(elev=30, azim=-55)
    ax.tick_params(labelsize=7)


def save_snapshot_plot(folder, step, density, binary, beta, threshold, period_um, layer_height_nm,
                       filename=None, figure_title=None, density_title=None, binary_title=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13, 6), layout="constrained")
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    draw_volume(ax1, density, period_um, layer_height_nm)
    draw_volume(ax2, binary, period_um, layer_height_nm, binary=True)
    ax1.set_title(density_title or f"Projected density | beta={beta:g}")
    ax2.set_title(binary_title or f"Evaluated binary | threshold={threshold:g}")
    title = figure_title or f"Topology step {step} | actual dimensions; layer 1 is top"
    if density.size > 160000:
        title += "\nDisplay-only block-averaged occupancy preview; solver/NPZ arrays remain full resolution"
    fig.suptitle(title)
    fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="Greys"), ax=[ax1, ax2],
                 orientation="horizontal", shrink=.5, pad=.06, fraction=.035,
                 label="TiO2 density (0 air, 1 TiO2); substrate at z=0")
    fig.savefig(folder / (filename or f"step_{step:04d}_3d.png"), dpi=150)
    plt.close(fig)


def save_evolution_overview(out, period_um, layer_height_nm):
    out = Path(out)
    files = sorted((out / "snapshots").glob("step_*_3d.png"))
    if not files:
        return
    from PIL import Image
    indices = np.unique(np.linspace(0, len(files)-1, min(6, len(files))).astype(int))
    fig, axes = plt.subplots(len(indices), 1, figsize=(13, 5*len(indices)))
    for ax, idx in zip(np.atleast_1d(axes), indices):
        ax.imshow(plt.imread(files[idx])); ax.axis("off")
    fig.tight_layout(pad=.1)
    fig.savefig(out / "structure_evolution_3d.png", dpi=110)
    plt.close(fig)
    frames = []
    for path in files:
        with Image.open(path) as im:
            frame = im.convert("RGB")
            frame.thumbnail((1100, 520))
            frames.append(frame)
    frames[0].save(out / "structure_evolution_3d.gif", save_all=True, append_images=frames[1:],
                   duration=900, loop=0)
