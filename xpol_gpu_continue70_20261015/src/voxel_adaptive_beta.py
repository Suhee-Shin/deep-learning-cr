"""Performance-based projection continuation; no fixed-step beta escalation."""
from dataclasses import asdict, dataclass
import math
import statistics


@dataclass
class BetaRules:
    window: int = 10
    check_every: int = 10
    min_residence: int = 20
    growth: float = 1.5
    max_beta: float = 64.0
    minimum_growth: float = 1.05
    improvement_absolute: float = .01
    improvement_relative: float = .005
    gray_fraction_floor: float = .10
    grayness_floor: float = .10
    collapse_relative: float = .05
    collapse_absolute: float = .05
    transmission_drop: float = .03
    max_trials: int = 4
    binary_first_step: int | None = None
    binary_growth: float = 1.25
    post_increase_steps: int = 10


class AdaptiveBeta:
    def __init__(self, rules=None, state=None):
        self.rules = rules or BetaRules()
        self.state = dict(growth=self.rules.growth, last_change=0, last_check=-1,
                          last_trial=-self.rules.min_residence)
        if state:
            self.state.update(state)

    def config(self):
        values = asdict(self.rules)
        if self.rules.binary_first_step is None:
            for key in ("binary_first_step", "binary_growth", "post_increase_steps"):
                values.pop(key)
        return values

    def binary_first(self, step):
        return self.rules.binary_first_step is not None and step >= self.rules.binary_first_step

    def decide(self, records, beta, step):
        r = self.rules
        if step == 0 or step % r.check_every or self.state["last_check"] == step:
            return None
        self.state["last_check"] = step
        binary_first = self.binary_first(step)
        event = dict(step=step, old_beta=beta, action="hold", binary_first=binary_first)
        if beta >= r.max_beta:
            return dict(event, reason="beta_cap")
        if step - max(self.state["last_change"], self.state["last_trial"]) < r.min_residence:
            return dict(event, reason="minimum_residence")
        segment = []
        for row in reversed(records):
            if not math.isclose(row["beta"], beta, rel_tol=0, abs_tol=1e-12):
                break
            if binary_first and row["step"] < r.binary_first_step:
                break
            segment.append(row)
        segment.reverse()
        if len(segment) < 2*r.window:
            return dict(event, reason="insufficient_same_beta_history")
        old, new = segment[-2*r.window:-r.window], segment[-r.window:]
        gains = {}
        for key in (("bin_fom",) if binary_first else ("fom", "bin_fom")):
            values = [row.get(key) for row in old+new]
            if any(value is None or not math.isfinite(value) for value in values):
                return dict(event, reason="missing_or_nonfinite_metrics")
            before = statistics.mean(row[key] for row in old)
            after = statistics.mean(row[key] for row in new)
            tolerance = max(r.improvement_absolute, r.improvement_relative*max(1, abs(before)))
            gains[key] = dict(before=before, after=after, gain=after-before, threshold=tolerance)
        event["window_metrics"] = gains
        if any(item["gain"] > item["threshold"] for item in gains.values()):
            return dict(event, reason="performance_improving")
        gray_fraction = statistics.mean(row["gray_fraction"] for row in new)
        grayness = statistics.mean(row["grayness"] for row in new)
        event.update(gray_fraction=gray_fraction, grayness=grayness)
        if gray_fraction < r.gray_fraction_floor or grayness < r.grayness_floor:
            return dict(event, reason="little_gray_material_remains")
        growth = min(self.state["growth"], r.binary_growth) if binary_first else self.state["growth"]
        return dict(event, action="probe", reason="plateau_with_gray_material",
                    proposed_beta=min(r.max_beta, beta*growth))

    def collapse_reasons(self, baseline, trial, kinds=("gray", "binary")):
        r = self.rules
        reasons = []
        for kind in kinds:
            before, after = baseline[kind], trial[kind]
            if not all(math.isfinite(v) for v in after.values()):
                reasons.append(kind+"_nonfinite")
                continue
            limit = max(r.collapse_absolute, r.collapse_relative*max(1, abs(before["fom"])))
            if before["fom"] - after["fom"] > limit:
                reasons.append(kind+"_fom_drop")
            if before["mean_tx"] - after["mean_tx"] > r.transmission_drop:
                reasons.append(kind+"_tx_drop")
            if after["mean_ty"] - before["mean_ty"] > r.transmission_drop:
                reasons.append(kind+"_ty_increase")
            if before["min_delta"] - after["min_delta"] > r.transmission_drop:
                reasons.append(kind+"_worst_delta_drop")
            if kinds == ("binary",):
                if "min_tx" in before and "min_tx" in after and before["min_tx"] - after["min_tx"] > r.transmission_drop:
                    reasons.append("binary_worst_tx_drop")
                if "max_ty" in before and "max_ty" in after and after["max_ty"] - before["max_ty"] > r.transmission_drop:
                    reasons.append("binary_worst_ty_increase")
        return reasons

    def search(self, decision, evaluate):
        """Probe the SAME theta without any Adam updates; retain theta/m/v on rejection."""
        old = decision["old_beta"]
        baseline = evaluate(old)
        event = dict(decision, baseline=baseline, trials=[])
        binary_first = decision.get("binary_first", False)
        growth = min(self.state["growth"], self.rules.binary_growth) if binary_first else self.state["growth"]
        self.state["last_trial"] = decision["step"]
        for _ in range(self.rules.max_trials):
            candidate = min(self.rules.max_beta, old*growth)
            trial = evaluate(candidate)
            reasons = self.collapse_reasons(baseline, trial, ("binary",) if binary_first else ("gray", "binary"))
            event["trials"].append(dict(beta=candidate, metrics=trial, rejected_for=reasons))
            if not reasons:
                self.state.update(growth=growth, last_change=decision["step"])
                return candidate, dict(event, action="increase", applied_beta=candidate,
                                       post_update_guard_required=binary_first)
            growth = 1 + (growth-1)/2
            self.state["growth"] = max(self.rules.minimum_growth, growth)
            if growth < self.rules.minimum_growth:
                break
        return old, dict(event, action="hold_after_rejected_probes", applied_beta=old)

    def check_post_increase(self, guard, records, step):
        """Judge actual hard-mask performance after updates, not gray/binary proximity."""
        if step - guard["start_step"] < self.rules.post_increase_steps:
            return None
        rows = [row for row in records if row["step"] > guard["start_step"]]
        if len(rows) < self.rules.post_increase_steps:
            return None
        recent = rows[-self.rules.post_increase_steps:]
        if any(not math.isclose(row["beta"], guard["new_beta"]) for row in recent):
            raise ValueError("Post-increase audit must use the same beta")
        mapping = dict(fom="bin_fom", mean_tx="bin_mean_tx", mean_ty="bin_mean_ty", min_delta="bin_min_delta")
        for key in ("min_tx", "max_ty"):
            if key in guard["baseline"]:
                mapping[key] = "bin_"+key
        after = {key: statistics.mean(row[source] for row in recent) for key, source in mapping.items()}
        reasons = self.collapse_reasons({"binary": guard["baseline"]}, {"binary": after}, ("binary",))
        event = dict(step=step, action="confirm_after_updates", binary_first=True,
                     increase_step=guard["start_step"], attempt=guard["attempt"],
                     old_beta=guard["old_beta"], evaluated_beta=guard["new_beta"],
                     baseline=guard["baseline"], after=after, rejected_for=reasons,
                     gray_penalty_enabled=False)
        if not reasons:
            self.state["last_trial"] = step
            return event
        growth = 1 + (guard["new_beta"]/guard["old_beta"] - 1)/2
        self.state.update(growth=max(self.rules.minimum_growth, growth), last_trial=step, last_change=step)
        retry = growth >= self.rules.minimum_growth and guard["attempt"] < self.rules.max_trials
        return dict(event, action="restore_and_reduce" if retry else "restore_and_hold",
                    applied_beta=guard["old_beta"]*growth if retry else guard["old_beta"],
                    gray_penalty_review_recommended=not retry)
