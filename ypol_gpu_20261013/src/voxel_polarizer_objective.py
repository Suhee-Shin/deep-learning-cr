"""Spectral objectives; transmission arrays are incident-power normalized."""
import math

import torch


def softmin_difference(tx, ty, *, tau=.03, target_tx=.8, target_ty=.1,
                       tx_penalty=2., ty_penalty=2.):
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("soft-min temperature must be positive and finite")
    if tx.shape != ty.shape or tx.ndim != 1 or tx.numel() == 0:
        raise ValueError("Expected equal nonempty wavelength vectors")
    if min(tx_penalty, ty_penalty) < 0:
        raise ValueError("Penalty weights cannot be negative")
    delta = tx - ty
    # Unnormalized log-sum-exp is a lower bound: min(delta)-tau*log(N) <= softmin <= min(delta).
    soft_min = -tau*torch.logsumexp(-delta/tau, dim=0)
    x_violation = torch.relu(target_tx-tx).square().amax()
    y_violation = torch.relu(ty-target_ty).square().amax()
    objective = soft_min - tx_penalty*x_violation - ty_penalty*y_violation
    return objective, dict(tx=tx, ty=ty, delta=delta, soft_min=soft_min,
                           wavelength_weights=torch.softmax(-delta/tau, dim=0),
                           worst_tx_violation_sq=x_violation, worst_ty_violation_sq=y_violation)


def softmin_diagnostics(spectrum, pass_pol="x", **kwargs):
    tx = torch.tensor([r["tx"] for r in spectrum], dtype=torch.float64)
    ty = torch.tensor([r["ty"] for r in spectrum], dtype=torch.float64)
    if pass_pol == "y":
        tx, ty = ty, tx
    fom, details = softmin_difference(tx, ty, **kwargs)
    return dict(fom=float(fom), soft_min=float(details["soft_min"]),
                min_delta=float((tx-ty).min()),
                worst_wavelength_nm=spectrum[int((tx-ty).argmin())]["wavelength_nm"],
                wavelength_weights=details["wavelength_weights"].tolist(),
                pass_pol=pass_pol,
                worst_pass_violation_sq=float(details["worst_tx_violation_sq"]),
                worst_block_violation_sq=float(details["worst_ty_violation_sq"]))
