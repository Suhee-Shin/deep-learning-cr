from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    import gdstk
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("gdstk is required. Install with: python -m pip install gdstk") from exc


def load_lumapi() -> Any:
    try:
        import lumapi  # type: ignore
        return lumapi
    except ImportError:
        api_hint = os.environ.get("LUMERICAL_API_PYTHON")
        candidates = [
            api_hint,
            r"C:\Program Files\Lumerical\v241\api\python",
            r"C:\Program Files\Lumerical\v242\api\python",
        ]
        for path in candidates:
            if not path:
                continue
            p = Path(path)
            if p.exists() and str(p) not in sys.path:
                sys.path.append(str(p))
        import lumapi  # type: ignore
        return lumapi


def build_resize_weights(src_size: int, dst_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.linspace(0.0, src_size - 1.0, dst_size, dtype=np.float64)
    i0 = np.floor(coords).astype(np.int32)
    i1 = np.clip(i0 + 1, 0, src_size - 1)
    w = coords - i0
    return i0, i1, w


def bilinear_upsample(
    grid_8x8: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    wy: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    wx: np.ndarray,
) -> np.ndarray:
    g_y0 = grid_8x8[y0, :]
    g_y1 = grid_8x8[y1, :]

    g00 = g_y0[:, x0]
    g01 = g_y0[:, x1]
    g10 = g_y1[:, x0]
    g11 = g_y1[:, x1]

    wx_ = wx.reshape(1, -1)
    wy_ = wy.reshape(-1, 1)

    top = (1.0 - wx_) * g00 + wx_ * g01
    bottom = (1.0 - wx_) * g10 + wx_ * g11
    return (1.0 - wy_) * top + wy_ * bottom


def seed_to_binary_structure(
    seed_4x8: np.ndarray,
    threshold: float,
    y0: np.ndarray,
    y1: np.ndarray,
    wy: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    wx: np.ndarray,
) -> np.ndarray:
    mirrored_8x8 = np.concatenate((seed_4x8, seed_4x8[::-1, :]), axis=0)
    interp_100x100 = bilinear_upsample(mirrored_8x8, y0, y1, wy, x0, x1, wx)
    return (interp_100x100 > threshold).astype(np.uint8)


def add_matrix_rectangles(cell: "gdstk.Cell", matrix: np.ndarray, pixel_size_um: float, layer: int, datatype: int) -> None:
    height, _ = matrix.shape

    for r in range(height):
        row = matrix[r]
        if not row.any():
            continue

        padded = np.pad(row.astype(np.int16, copy=False), (1, 1), mode="constant", constant_values=0)
        edges = np.diff(padded)
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1) - 1

        y0 = (height - 1 - r) * pixel_size_um
        y1 = y0 + pixel_size_um

        for s, e in zip(starts, ends, strict=False):
            x0 = s * pixel_size_um
            x1 = (e + 1) * pixel_size_um
            cell.add(gdstk.rectangle((x0, y0), (x1, y1), layer=layer, datatype=datatype))


def write_single_gds(matrix: np.ndarray, out_path: Path, cell_name: str, pixel_size_um: float, layer: int = 1, datatype: int = 0) -> None:
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell(cell_name)
    add_matrix_rectangles(cell, matrix, pixel_size_um=pixel_size_um, layer=layer, datatype=datatype)
    lib.write_gds(out_path)


@dataclass
class EvalResult:
    ce_score: float
    loss: float
    ce_map_path: Path
    status: str
    error: str


def compute_ce_score(
    ce_vector: np.ndarray,
    lambda_vector: np.ndarray | None,
    score_mode: str,
    band_start_nm: float,
    band_end_nm: float,
    band_factor: float,
) -> float:
    ce = np.asarray(ce_vector, dtype=np.float64).squeeze()
    if ce.ndim != 1:
        ce = ce.reshape(-1)

    if score_mode == "mean" or lambda_vector is None:
        return float(np.mean(ce))

    lam_nm = np.asarray(lambda_vector, dtype=np.float64).squeeze() * 1e9
    if lam_nm.ndim != 1:
        lam_nm = lam_nm.reshape(-1)

    n = min(ce.shape[0], lam_nm.shape[0])
    ce = ce[:n]
    lam_nm = lam_nm[:n]

    lo = min(band_start_nm, band_end_nm)
    hi = max(band_start_nm, band_end_nm)
    weights = np.ones_like(ce)
    mask = (lam_nm >= lo) & (lam_nm <= hi)
    if np.any(mask):
        weights[mask] = band_factor
    return float(np.sum(weights * ce) / np.sum(weights))


def approx_seed_from_structure(structure_100x100: np.ndarray) -> np.ndarray:
    s = np.asarray(structure_100x100, dtype=np.float64).squeeze()
    if s.shape != (100, 100):
        raise ValueError(f"init structure must be (100,100), got {s.shape}")

    y0, y1, wy = build_resize_weights(100, 8)
    x0, x1, wx = build_resize_weights(100, 8)
    coarse_8x8 = bilinear_upsample(s, y0, y1, wy, x0, x1, wx)
    coarse_8x8 = np.clip(coarse_8x8, 0.0, 1.0)
    coarse_sym = 0.5 * (coarse_8x8 + coarse_8x8[::-1, :])
    return np.clip(coarse_sym[:4, :], 0.0, 1.0).astype(np.float64)


class FDTDEvaluator:
    def __init__(
        self,
        fsp_path: Path,
        hide: bool,
        max_retries: int,
        score_mode: str,
        band_start_nm: float,
        band_end_nm: float,
        band_factor: float,
    ) -> None:
        self.fsp_path = fsp_path
        self.hide = hide
        self.max_retries = max_retries
        self.score_mode = score_mode
        self.band_start_nm = band_start_nm
        self.band_end_nm = band_end_nm
        self.band_factor = band_factor
        self.lumapi = load_lumapi()

    def evaluate(self, gds_path: Path, cell_name: str, ce_map_path: Path) -> EvalResult:
        last_error = ""
        for attempt in range(self.max_retries + 1):
            fdtd = self.lumapi.FDTD(hide=self.hide)
            try:
                fdtd.load(str(self.fsp_path))
                fdtd.eval("switchtolayout;")
                fdtd.eval('if (getnamednumber("GDS_LAYER_1:0")>0) {select("GDS_LAYER_1:0"); delete;}')

                gds_posix = gds_path.resolve().as_posix()
                fdtd.eval(
                    f'gdsimport("{gds_posix}", "{cell_name}", "1:0", "Si (Silicon) - Palik", 0, 220e-9);'
                )
                fdtd.select("GDS_LAYER_1:0")
                fdtd.set("x", 0.0)
                fdtd.set("y", -5.6e-6)
                fdtd.set("z", 0.11e-6)

                fdtd.run()
                result = fdtd.getresult("mode", "expansion for output_T")
                if (not result) or ("T_backward" not in result):
                    raise RuntimeError("FDTD result missing: mode/expansion for output_T/T_backward")

                t_backward = np.real(np.asarray(result["T_backward"]).squeeze())
                ce_map = np.abs(t_backward).astype(np.float64)
                np.save(ce_map_path, ce_map)
                lam = np.asarray(result["lambda"]).squeeze() if "lambda" in result else None
                ce_score = compute_ce_score(
                    ce_vector=ce_map,
                    lambda_vector=lam,
                    score_mode=self.score_mode,
                    band_start_nm=self.band_start_nm,
                    band_end_nm=self.band_end_nm,
                    band_factor=self.band_factor,
                )
                loss = -ce_score
                return EvalResult(ce_score=ce_score, loss=loss, ce_map_path=ce_map_path, status="ok", error="")
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc!r}"
                if attempt < self.max_retries:
                    continue
            finally:
                fdtd.close()

        return EvalResult(
            ce_score=float("nan"),
            loss=float("inf"),
            ce_map_path=ce_map_path,
            status="error",
            error=last_error,
        )


def mock_evaluate(
    structure_100x100: np.ndarray,
    ce_map_path: Path,
    score_mode: str,
    band_start_nm: float,
    band_end_nm: float,
    band_factor: float,
) -> EvalResult:
    # Dry-run objective: prefers ~45% fill with moderate edge complexity.
    s = structure_100x100.astype(np.float64)
    fill = float(np.mean(s))
    edge = float(np.mean(np.abs(np.diff(s, axis=0))) + np.mean(np.abs(np.diff(s, axis=1))))
    base = float(np.exp(-((fill - 0.45) ** 2) / 0.01) * (1.0 + 0.2 * edge))
    lam_nm = np.linspace(1600.0, 1500.0, 100, dtype=np.float64)
    wav_shape = 1.0 + 0.2 * np.exp(-0.5 * ((lam_nm - 1550.0) / 8.0) ** 2)
    ce_map = base * wav_shape
    np.save(ce_map_path, ce_map)
    ce_score = compute_ce_score(
        ce_vector=ce_map,
        lambda_vector=lam_nm * 1e-9,
        score_mode=score_mode,
        band_start_nm=band_start_nm,
        band_end_nm=band_end_nm,
        band_factor=band_factor,
    )
    return EvalResult(ce_score=ce_score, loss=-ce_score, ce_map_path=ce_map_path, status="ok", error="")


class CMAES:
    def __init__(
        self,
        dim: int,
        popsize: int,
        sigma: float,
        seed: int,
        lower: float = 0.0,
        upper: float = 1.0,
        init_mean: np.ndarray | None = None,
    ) -> None:
        self.dim = dim
        self.popsize = popsize
        self.mu = popsize // 2
        self.sigma = sigma
        self.lower = lower
        self.upper = upper
        self.rng = np.random.default_rng(seed)

        if init_mean is None:
            self.mean = np.full(dim, 0.5, dtype=np.float64)
        else:
            self.mean = np.asarray(init_mean, dtype=np.float64).reshape(dim)

        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights ** 2)

        n = self.dim
        self.cc = (4.0 + self.mueff / n) / (n + 4.0 + 2.0 * self.mueff / n)
        self.cs = (self.mueff + 2.0) / (n + self.mueff + 5.0)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1.0 - self.c1, 2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((n + 2.0) ** 2 + self.mueff))
        self.damps = 1.0 + 2.0 * max(0.0, np.sqrt((self.mueff - 1.0) / (n + 1.0)) - 1.0) + self.cs

        self.pc = np.zeros(n, dtype=np.float64)
        self.ps = np.zeros(n, dtype=np.float64)
        self.B = np.eye(n, dtype=np.float64)
        self.D = np.ones(n, dtype=np.float64)
        self.C = np.eye(n, dtype=np.float64)
        self.invsqrtC = np.eye(n, dtype=np.float64)
        self.chiN = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
        self.counteval = 0
        self.eigeneval = 0
        self.last_arz = None
        self.last_ary = None
        self.last_arx = None

    def ask(self) -> np.ndarray:
        arz = self.rng.standard_normal((self.popsize, self.dim))
        BD = self.B * self.D.reshape(1, -1)
        ary = arz @ BD.T
        arx = self.mean + self.sigma * ary
        arx = np.clip(arx, self.lower, self.upper)

        self.last_arz = arz
        self.last_ary = ary
        self.last_arx = arx
        return arx

    def tell(self, losses: np.ndarray) -> None:
        assert self.last_arz is not None and self.last_ary is not None and self.last_arx is not None
        losses = np.asarray(losses, dtype=np.float64)
        finite_mask = np.isfinite(losses)
        if not np.any(finite_mask):
            # If all simulations fail, keep mean and widen search a bit.
            self.sigma *= 1.2
            return
        if not np.all(finite_mask):
            worst_finite = np.max(losses[finite_mask])
            losses = losses.copy()
            losses[~finite_mask] = worst_finite + 1.0
        order = np.argsort(losses)
        idx = order[: self.mu]

        xold = self.mean.copy()
        xsel = self.last_arx[idx]
        zsel = self.last_arz[idx]
        ysel = self.last_ary[idx]

        self.mean = np.sum(self.weights[:, None] * xsel, axis=0)
        self.mean = np.clip(self.mean, self.lower, self.upper)

        zmean = np.sum(self.weights[:, None] * zsel, axis=0)
        ymean = np.sum(self.weights[:, None] * ysel, axis=0)

        self.counteval += self.popsize

        self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * (self.B @ zmean)
        norm_ps = np.linalg.norm(self.ps)
        left = norm_ps / np.sqrt(1.0 - (1.0 - self.cs) ** (2.0 * self.counteval / self.popsize))
        hsig = float(left / self.chiN < (1.4 + 2.0 / (self.dim + 1.0)))

        self.pc = (1.0 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2.0 - self.cc) * self.mueff) * ymean

        rank_one = np.outer(self.pc, self.pc)
        rank_mu = np.zeros_like(self.C)
        for w, yi in zip(self.weights, ysel, strict=False):
            rank_mu += w * np.outer(yi, yi)

        self.C = (
            (1.0 - self.c1 - self.cmu + (1.0 - hsig) * self.c1 * self.cc * (2.0 - self.cc)) * self.C
            + self.c1 * rank_one
            + self.cmu * rank_mu
        )
        self.sigma *= np.exp((self.cs / self.damps) * (norm_ps / self.chiN - 1.0))

        update_freq = int(self.popsize / (self.c1 + self.cmu) / self.dim / 10.0)
        update_freq = max(update_freq, 1)
        if self.counteval - self.eigeneval >= update_freq:
            self.eigeneval = self.counteval
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-30)
            self.D = np.sqrt(eigvals)
            self.B = eigvecs
            self.invsqrtC = self.B @ np.diag(1.0 / self.D) @ self.B.T


def plot_loss_curve(rows: list[dict[str, Any]], out_path: Path) -> None:
    if plt is None:
        return
    if not rows:
        return

    by_gen: dict[int, list[float]] = {}
    for row in rows:
        loss = row["loss"]
        if not np.isfinite(loss):
            continue
        gen = int(row["generation"])
        by_gen.setdefault(gen, []).append(float(loss))

    if not by_gen:
        return

    gens = sorted(by_gen.keys())
    best = [min(by_gen[g]) for g in gens]
    mean = [float(np.mean(by_gen[g])) for g in gens]

    plt.figure(figsize=(8, 5))
    plt.plot(gens, best, marker="o", label="Best loss (min)")
    plt.plot(gens, mean, marker="s", label="Mean loss")
    plt.xlabel("Generation")
    plt.ylabel("Loss (-CE)")
    plt.title("CMA-ES Loss History")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CMA-ES loop for 4x8 seed -> FDTD CE optimization.")
    parser.add_argument("--fsp-path", type=Path, default=Path(r"C:\Deep learning\fdtd_results\outcoupler.fsp"))
    parser.add_argument("--results-root", type=Path, default=Path(r"C:\Deep learning\fdtd_results\CE_results"))
    parser.add_argument("--population-size", type=int, default=14, help="First generation sample size.")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--sigma-min", type=float, default=0.08)
    parser.add_argument("--sigma-max", type=float, default=0.8)
    parser.add_argument("--stall-patience", type=int, default=12)
    parser.add_argument("--stall-sigma-decay", type=float, default=0.55)
    parser.add_argument("--reset-jitter", type=float, default=0.03)
    parser.add_argument("--improve-tol", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pixel-size-nm", type=float, default=140.0)
    parser.add_argument("--hide-fdtd", action="store_true")
    parser.add_argument("--fdtd-retries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Run without lumapi/FDTD; useful for quick pipeline test.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional fixed run folder name under results-root.")
    parser.add_argument("--ce-score-mode", type=str, default="mean", choices=["mean", "weighted_band"])
    parser.add_argument("--weight-band-start-nm", type=float, default=1540.0)
    parser.add_argument("--weight-band-end-nm", type=float, default=1560.0)
    parser.add_argument("--weight-band-factor", type=float, default=4.0)
    parser.add_argument("--init-mean-file", type=Path, default=None, help="Path to .npy containing (32,) or (4,8) mean.")
    parser.add_argument("--init-mean-from-structure", type=Path, default=None, help="Path to (100,100) structure .npy to approximate initial 4x8 seed mean.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.population_size < 2:
        raise ValueError("population-size must be >= 2")
    if args.generations < 1:
        raise ValueError("generations must be >= 1")
    if args.sigma_min <= 0 or args.sigma_max <= 0 or args.sigma_min > args.sigma_max:
        raise ValueError("sigma range must satisfy: 0 < sigma-min <= sigma-max")
    if args.stall_patience < 1:
        raise ValueError("stall-patience must be >= 1")
    if not (0.0 < args.stall_sigma_decay <= 1.0):
        raise ValueError("stall-sigma-decay must be in (0, 1]")
    if args.reset_jitter < 0:
        raise ValueError("reset-jitter must be >= 0")
    if args.weight_band_factor <= 0:
        raise ValueError("weight-band-factor must be > 0")
    if args.init_mean_file is not None and args.init_mean_from_structure is not None:
        raise ValueError("Use only one of --init-mean-file or --init-mean-from-structure")

    run_id = args.run_name if args.run_name else dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.results_root / run_id
    gds_dir = run_dir / "gds"
    struct_dir = run_dir / "structures"
    ce_dir = run_dir / "ce_maps"

    gds_dir.mkdir(parents=True, exist_ok=True)
    struct_dir.mkdir(parents=True, exist_ok=True)
    ce_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "fsp_path": str(args.fsp_path),
        "population_size": args.population_size,
        "generations": args.generations,
        "sigma": args.sigma,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
        "stall_patience": args.stall_patience,
        "stall_sigma_decay": args.stall_sigma_decay,
        "reset_jitter": args.reset_jitter,
        "improve_tol": args.improve_tol,
        "ce_score_mode": args.ce_score_mode,
        "weight_band_start_nm": args.weight_band_start_nm,
        "weight_band_end_nm": args.weight_band_end_nm,
        "weight_band_factor": args.weight_band_factor,
        "seed": args.seed,
        "threshold": args.threshold,
        "pixel_size_nm": args.pixel_size_nm,
        "structure_size_um": [14.0, 14.0],
        "seed_shape": [4, 8],
        "coarse_shape": [8, 8],
        "final_shape": [100, 100],
        "dry_run": args.dry_run,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history_csv = run_dir / "history.csv"
    global_best_history_csv = run_dir / "global_best_history.csv"
    sigma_history_csv = run_dir / "sigma_history.csv"
    global_best_json = run_dir / "global_best.json"
    best_json = run_dir / "best_result.json"
    loss_png = run_dir / "loss_curve.png"

    init_mean_vec = None
    if args.init_mean_file is not None:
        init_arr = np.load(args.init_mean_file)
        init_arr = np.asarray(init_arr, dtype=np.float64)
        if init_arr.shape == (4, 8):
            init_mean_vec = init_arr.reshape(-1)
        elif init_arr.size == 32:
            init_mean_vec = init_arr.reshape(32)
        else:
            raise ValueError(f"init-mean-file must contain shape (4,8) or size 32, got {init_arr.shape}")
    elif args.init_mean_from_structure is not None:
        init_struct = np.load(args.init_mean_from_structure)
        init_seed = approx_seed_from_structure(init_struct)
        init_mean_vec = init_seed.reshape(-1)

    cma = CMAES(dim=32, popsize=args.population_size, sigma=args.sigma, seed=args.seed, init_mean=init_mean_vec)
    y0, y1, wy = build_resize_weights(8, 100)
    x0, x1, wx = build_resize_weights(8, 100)

    all_rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    global_best_loss = float("inf")
    global_best_ce = float("nan")
    global_best_generation = -1
    global_best_x: np.ndarray | None = None
    last_improve_gen = -1
    last_reset_gen = -1
    pixel_size_um = args.pixel_size_nm / 1000.0

    if args.dry_run:
        evaluator = None
    else:
        evaluator = FDTDEvaluator(
            fsp_path=args.fsp_path,
            hide=args.hide_fdtd,
            max_retries=args.fdtd_retries,
            score_mode=args.ce_score_mode,
            band_start_nm=args.weight_band_start_nm,
            band_end_nm=args.weight_band_end_nm,
            band_factor=args.weight_band_factor,
        )

    with open(history_csv, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "generation",
                "candidate",
                "eval_index",
                "ce_score",
                "loss",
                "status",
                "error",
                "gds_path",
                "structure_path",
                "ce_map_path",
            ],
        )
        writer.writeheader()

        with open(global_best_history_csv, "w", newline="", encoding="utf-8") as f_gb, \
            open(sigma_history_csv, "w", newline="", encoding="utf-8") as f_sigma:
            gb_writer = csv.DictWriter(
                f_gb,
                fieldnames=[
                    "generation",
                    "current_best_loss",
                    "current_best_ce",
                    "previous_global_best_loss",
                    "global_best_loss",
                    "global_best_ce",
                    "global_best_generation",
                    "improved_global_best",
                    "stall_generations",
                    "sigma_capped",
                    "reset_applied",
                    "n_valid",
                    "n_error",
                ],
            )
            sigma_writer = csv.DictWriter(
                f_sigma,
                fieldnames=[
                    "generation",
                    "sigma_before",
                    "sigma_raw_after_tell",
                    "sigma_after_clip_reset",
                    "sigma_capped",
                    "reset_applied",
                    "stall_generations",
                ],
            )
            gb_writer.writeheader()
            sigma_writer.writeheader()

            eval_index = 0
            for gen in range(args.generations):
                sigma_before = float(cma.sigma)
                candidates = cma.ask()
                losses = np.zeros(args.population_size, dtype=np.float64)
                n_error = 0

                print(f"[Generation {gen}] evaluating {args.population_size} candidates (sigma={sigma_before:.6g})")

                for cand_idx, x in enumerate(candidates):
                    eval_index += 1
                    seed_4x8 = x.reshape(4, 8)
                    structure_100x100 = seed_to_binary_structure(
                        seed_4x8=seed_4x8,
                        threshold=args.threshold,
                        y0=y0,
                        y1=y1,
                        wy=wy,
                        x0=x0,
                        x1=x1,
                        wx=wx,
                    )

                    base = f"g{gen:03d}_c{cand_idx:02d}"
                    cell_name = f"S_{base}"
                    gds_path = gds_dir / f"{base}.gds"
                    structure_path = struct_dir / f"{base}.npy"
                    ce_map_path = ce_dir / f"CE_{base}.npy"

                    np.save(structure_path, structure_100x100)
                    write_single_gds(
                        matrix=structure_100x100,
                        out_path=gds_path,
                        cell_name=cell_name,
                        pixel_size_um=pixel_size_um,
                        layer=1,
                        datatype=0,
                    )

                    if args.dry_run:
                        result = mock_evaluate(
                            structure_100x100=structure_100x100,
                            ce_map_path=ce_map_path,
                            score_mode=args.ce_score_mode,
                            band_start_nm=args.weight_band_start_nm,
                            band_end_nm=args.weight_band_end_nm,
                            band_factor=args.weight_band_factor,
                        )
                    else:
                        assert evaluator is not None
                        result = evaluator.evaluate(gds_path=gds_path, cell_name=cell_name, ce_map_path=ce_map_path)

                    if result.status != "ok":
                        n_error += 1

                    losses[cand_idx] = result.loss
                    row = {
                        "generation": gen,
                        "candidate": cand_idx,
                        "eval_index": eval_index,
                        "ce_score": result.ce_score,
                        "loss": result.loss,
                        "status": result.status,
                        "error": result.error,
                        "gds_path": str(gds_path),
                        "structure_path": str(structure_path),
                        "ce_map_path": str(result.ce_map_path),
                    }
                    writer.writerow(row)
                    f_csv.flush()
                    all_rows.append(row)

                    if np.isfinite(result.loss):
                        if (best_row is None) or (result.loss < float(best_row["loss"])):
                            best_row = row.copy()

                    print(
                        f"  - cand {cand_idx:02d}: status={result.status}, "
                        f"CE={result.ce_score:.6g}, loss={result.loss:.6g}"
                    )

                cma.tell(losses)
                sigma_raw_after_tell = float(cma.sigma)

                sigma_capped = False
                if cma.sigma > args.sigma_max:
                    cma.sigma = args.sigma_max
                    sigma_capped = True
                elif cma.sigma < args.sigma_min:
                    cma.sigma = args.sigma_min
                    sigma_capped = True

                finite_mask = np.isfinite(losses)
                n_valid = int(np.sum(finite_mask))
                if n_valid > 0:
                    local_idx = int(np.argmin(np.where(finite_mask, losses, np.inf)))
                    current_best_loss = float(losses[local_idx])
                    current_best_ce = -current_best_loss
                    current_best_x = candidates[local_idx].copy()
                else:
                    current_best_loss = float("inf")
                    current_best_ce = float("nan")
                    current_best_x = None

                prev_global_best_loss = global_best_loss
                improved_global_best = False
                if np.isfinite(current_best_loss) and current_best_loss < (global_best_loss - args.improve_tol):
                    global_best_loss = current_best_loss
                    global_best_ce = -global_best_loss
                    global_best_generation = gen
                    global_best_x = current_best_x
                    last_improve_gen = gen
                    improved_global_best = True

                if last_improve_gen < 0:
                    stall_generations = 0
                else:
                    stall_generations = gen - last_improve_gen

                reset_applied = False
                if (
                    global_best_x is not None
                    and stall_generations >= args.stall_patience
                    and (last_reset_gen < 0 or (gen - last_reset_gen) >= args.stall_patience)
                ):
                    jitter = cma.rng.normal(0.0, args.reset_jitter, size=cma.mean.shape)
                    cma.mean = np.clip(global_best_x + jitter, cma.lower, cma.upper)
                    cma.sigma = max(args.sigma_min, min(args.sigma_max, cma.sigma * args.stall_sigma_decay))
                    last_reset_gen = gen
                    reset_applied = True

                sigma_after = float(cma.sigma)
                plot_loss_curve(all_rows, loss_png)

                gen_losses = [
                    float(r["loss"]) for r in all_rows
                    if int(r["generation"]) == gen and np.isfinite(float(r["loss"]))
                ]

                gb_row = {
                    "generation": gen,
                    "current_best_loss": current_best_loss,
                    "current_best_ce": current_best_ce,
                    "previous_global_best_loss": prev_global_best_loss,
                    "global_best_loss": global_best_loss,
                    "global_best_ce": global_best_ce,
                    "global_best_generation": global_best_generation,
                    "improved_global_best": int(improved_global_best),
                    "stall_generations": stall_generations,
                    "sigma_capped": int(sigma_capped),
                    "reset_applied": int(reset_applied),
                    "n_valid": n_valid,
                    "n_error": n_error,
                }
                gb_writer.writerow(gb_row)
                f_gb.flush()

                sigma_writer.writerow(
                    {
                        "generation": gen,
                        "sigma_before": sigma_before,
                        "sigma_raw_after_tell": sigma_raw_after_tell,
                        "sigma_after_clip_reset": sigma_after,
                        "sigma_capped": int(sigma_capped),
                        "reset_applied": int(reset_applied),
                        "stall_generations": stall_generations,
                    }
                )
                f_sigma.flush()

                global_best_payload = {
                    "generation": gen,
                    "global_best_loss": global_best_loss,
                    "global_best_ce": global_best_ce,
                    "global_best_generation": global_best_generation,
                    "stall_generations": stall_generations,
                    "sigma": sigma_after,
                }
                global_best_json.write_text(json.dumps(global_best_payload, indent=2), encoding="utf-8")

                if gen_losses:
                    print(
                        f"[Generation {gen}] best={current_best_loss:.6g}, "
                        f"mean={np.mean(gen_losses):.6g}, "
                        f"global_best={global_best_loss:.6g}, "
                        f"stall={stall_generations}, "
                        f"sigma={sigma_before:.6g}->{sigma_raw_after_tell:.6g}->{sigma_after:.6g}, "
                        f"capped={int(sigma_capped)}, reset={int(reset_applied)}"
                    )
                else:
                    print(
                        f"[Generation {gen}] no valid evaluations, "
                        f"global_best={global_best_loss:.6g}, "
                        f"stall={stall_generations}, "
                        f"sigma={sigma_before:.6g}->{sigma_raw_after_tell:.6g}->{sigma_after:.6g}, "
                        f"capped={int(sigma_capped)}, reset={int(reset_applied)}"
                    )

    if best_row is not None:
        best_json.write_text(json.dumps(best_row, indent=2), encoding="utf-8")
        print(f"Best result saved: {best_json}")
    else:
        print("No valid result to save in best_result.json")

    print(f"History CSV: {history_csv}")
    print(f"Loss curve:  {loss_png}")
    print(f"Run folder:  {run_dir}")


if __name__ == "__main__":
    main()
