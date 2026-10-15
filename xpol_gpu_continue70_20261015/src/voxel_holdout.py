"""Independent binary checks at wavelengths omitted from training."""
from pathlib import Path
import torch
from voxel_goal_policy import strict_target


def evaluate_holdout(args, candidate, step, device, progress=None):
    import run_tio2_xypol_filter_voxel3d as r
    from voxel_live_report import atomic_json
    wavelengths = sorted({float(w) for w in args.holdout_wavelengths.split(",")})
    train = {float(w) for w in args.wavelengths.split(",")}
    if train.intersection(wavelengths):
        raise ValueError("Holdout wavelengths must not overlap training")
    stack, pupil, rays, _ = r.build_stack(args)
    density = torch.as_tensor(candidate["stack"], dtype=torch.float64, device=device)
    rows = []
    with torch.no_grad():
        for wl in wavelengths:
            if progress:
                progress("start", "binary_holdout", "forward", [wl])
            fwd = r.FastPupilForward(stack=stack, pupil_spec=pupil, wavelengths_nm=[wl], rays=rays,
                device=device, chunk=1, dtype=torch.complex128 if args.dtype == "complex128" else torch.complex64,
                verbose_guard=False)
            tabs = r.polarized_tabs(fwd, density[None], tags=("holdout",), grad=[False])[0]
            rows.extend(r.spectrum_rows([wl], tabs))
            del fwd, tabs
            if progress:
                progress("end", "binary_holdout", "forward", [wl])
    merged = sorted(candidate["spectrum"] + rows, key=lambda row: row["wavelength_nm"])
    report = dict(step=step, training_wavelengths_nm=sorted(train), holdout_spectrum=rows,
                  combined_spectrum=merged, optimizer_updated_by_holdout=False,
                  target_tx=args.target_tx, target_ty=args.target_ty,
                  target_comparison="TX > target_tx and TY < target_ty",
                  all_sampled_points_pass=strict_target(merged, sorted(train.union(wavelengths)),
                                                       target_tx=args.target_tx, target_ty=args.target_ty),
                  continuous_band_validated=False, fabrication_certified=False)
    folder = Path(args.out)/"holdout"
    folder.mkdir(exist_ok=True)
    atomic_json(folder/f"step_{step:04d}.json", report)
    atomic_json(Path(args.out)/"holdout_latest.json", report)
    return report
