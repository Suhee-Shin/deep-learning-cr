"""Explicit objective phases and fail-closed gradient outlier checks."""
from argparse import Namespace
import math
import statistics


def active_objective_args(args, step):
    mode = args.objective_mode
    if mode == "legacy_then_softmin":
        mode = "legacy" if step < args.softmin_start_step else "softmin"
    return Namespace(**dict(vars(args), objective_mode=mode))


def phase_changed(records, mode):
    return bool(records and records[-1].get("objective_mode", "legacy") != mode)


def gradient_outlier(records, norm, mode, beta, *, factor=100., window=20, minimum=10):
    if not math.isfinite(norm):
        return dict(reason="nonfinite", norm=norm)
    if factor <= 0:
        return None
    recent = []
    for row in reversed(records):
        if row.get("objective_mode", "legacy") != mode or row["beta"] != beta:
            break
        value = row.get("gradient_norm")
        if value is not None and math.isfinite(value):
            recent.append(value)
        if len(recent) >= window:
            break
    if len(recent) < minimum:
        return None
    reference = statistics.median(recent)
    if norm > factor * max(reference, 1e-12):
        return dict(reason="gradient_norm_outlier", norm=norm, median=reference,
                    factor=factor, history_count=len(recent), update_applied=False)
    return None
