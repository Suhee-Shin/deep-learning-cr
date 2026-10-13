"""Strict all-wavelength binary acceptance, separate from soft training scores."""
import math


def target_reached(spectrum, target_tx, target_ty, pass_pol="x"):
    passed, blocked = ("tx", "ty") if pass_pol == "x" else ("ty", "tx")
    return bool(spectrum) and all(
        math.isfinite(row["tx"]) and math.isfinite(row["ty"])
        and target_tx < row[passed] <= 1.001
        and 0 <= row[blocked] < target_ty
        for row in spectrum
    )


def assess_validation(results, wavelengths, orders, target_tx, target_ty, tolerance, pass_pol="x"):
    expected = list(wavelengths)
    if len(orders) < 2 or len(set(orders)) != len(orders):
        raise ValueError("At least two distinct validation orders are required")
    if [r["order"] for r in results] != list(orders):
        raise ValueError("Missing or reordered validation orders")
    if any([s["wavelength_nm"] for s in r["spectrum"]] != expected for r in results):
        raise ValueError("Validation wavelength lists differ")
    passed = all(target_reached(r["spectrum"], target_tx, target_ty, pass_pol) for r in results)
    changes = []
    for a, b in zip(results, results[1:]):
        differences = [abs(x[key]-y[key]) for x, y in zip(a["spectrum"], b["spectrum"]) for key in ("tx", "ty")]
        finite = all(math.isfinite(d) for d in differences)
        changes.append(dict(from_order=a["order"], to_order=b["order"],
                            max_absolute_change=max(differences) if finite else None,
                            within_tolerance=finite and max(differences) <= tolerance))
    converged = all(r["within_tolerance"] for r in changes)
    return dict(optical_target_passed=passed, order_agreement_passed=converged,
                accepted=passed and converged, order_tolerance=tolerance, order_changes=changes,
                target_pass_min=target_tx, target_block_max=target_ty, pass_pol=pass_pol,
                scope="periodic RCWA binary validation; not finite FDTD or fabrication certification")
