"""Binary-aware beta continuation for filtered-density gray training.

No STE, threshold changes, or gray-to-binary overwrite. Decisions use evaluated
hard masks; rollback anchors contain continuous parameters and Adam moments.
"""
from dataclasses import asdict, dataclass
import math
import statistics


@dataclass(frozen=True)
class BinaryContinuationRules:
    version: int = 1
    growth: float = 1.5
    max_beta: float = 64.
    min_residence: int = 20
    evaluations: int = 4
    check_every: int = 5
    improvement: float = .005
    transmission_tolerance: float = .01
    gray_fraction_floor: float = .1
    grayness_floor: float = .1
    gray_binary_gap_floor: float = .05
    collapse_tolerance: float = .03
    minimum_growth: float = 1.05
    max_attempts: int = 3

    def __post_init__(self):
        if self.version != 1 or self.evaluations < 4 or self.evaluations % 2:
            raise ValueError("Unsupported binary continuation version/evaluation window")
        if min(self.check_every, self.min_residence, self.max_attempts) < 1:
            raise ValueError("Continuation intervals must be positive")
        if not 1 < self.minimum_growth <= self.growth <= 1.5 or self.max_beta <= 0:
            raise ValueError("Invalid beta growth/cap")
        for key, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid continuation rule: {key}")


METRICS = ("mean_tx", "min_tx", "mean_ty", "max_ty", "mean_delta", "min_delta")


def binary_metrics(row):
    values = {key: row.get("bin_" + key) for key in METRICS}
    if any(value is None or not math.isfinite(value) for value in values.values()):
        raise ValueError("Binary continuation requires finite evaluated binary metrics")
    return values


def mean_metrics(rows):
    values = [binary_metrics(row) for row in rows]
    return {key: statistics.mean(item[key] for item in values) for key in METRICS}


class BinaryContinuation:
    def __init__(self, rules, state=None):
        self.rules = rules
        self.state = dict(last_check=-1, last_change=-rules.min_residence,
                          growth=rules.growth, pending=None)
        if state:
            self.state.update(state)

    def event(self, step, beta, **values):
        return dict(step=step, old_beta=beta, applied_beta=beta,
                    policy="binary_aware_gray_continuation_v1", action="hold", **values)

    def phase_reset(self, step, beta):
        pending = self.state.get("pending")
        self.state.update(pending=None, last_change=step, last_check=-1)
        return dict(self.event(step, beta, reason="objective_phase_changed"),
                    action="reset_phase_guard", cancelled_trial=pending,
                    theta_preserved=True)

    def decide(self, records, beta, step):
        r = self.rules
        if step % r.check_every or self.state["last_check"] == step:
            return None
        self.state["last_check"] = step
        event = self.event(step, beta)
        # A phase or beta change starts a new residence window; never mix scores.
        segment = []
        mode = records[-1].get("objective_mode") if records else None
        for row in reversed(records):
            if not math.isclose(row["beta"], beta, abs_tol=1e-12, rel_tol=0) or row.get("objective_mode") != mode:
                break
            segment.append(row)
        segment.reverse()
        pairs = [row for row in segment if row.get("bin_mean_tx") is not None]
        for row in pairs:
            binary_metrics(row)
        pending = self.state.get("pending")
        if pending:
            after = [row for row in pairs if row["step"] > pending["start_step"]]
            if not pending.get("baseline") or len(after) < r.evaluations or after[-1]["step"] - pending["start_step"] < r.min_residence:
                return dict(event, reason="post_increase_settling", increase_step=pending["start_step"])
            recent = mean_metrics(after[-2:])
            baseline = pending["baseline"]
            reasons = []
            for key in ("mean_tx", "min_tx", "mean_delta", "min_delta"):
                if baseline[key] - recent[key] > r.collapse_tolerance:
                    reasons.append(key + "_drop")
            for key in ("mean_ty", "max_ty"):
                if recent[key] - baseline[key] > r.collapse_tolerance:
                    reasons.append(key + "_increase")
            audit = dict(event, baseline=baseline, after=recent, rejected_for=reasons,
                         audited_steps=[row["step"] for row in after[-r.evaluations:]],
                         anchor=pending["anchor"], increase_step=pending["start_step"])
            self.state["last_change"] = step
            if not reasons:
                self.state["pending"] = None
                return dict(audit, action="confirm_after_binary_updates", reason="no_material_binary_regression")
            growth = 1 + (pending["new_beta"] / pending["old_beta"] - 1) / 2
            retry = growth >= r.minimum_growth and pending["attempt"] < r.max_attempts
            self.state["growth"] = max(r.minimum_growth, growth)
            beta_next = pending["old_beta"] * growth if retry else pending["old_beta"]
            self.state["pending"] = (dict(pending, new_beta=beta_next, start_step=step,
                baseline=None, attempt=pending["attempt"] + 1) if retry else None)
            return dict(audit, action="restore_and_reduce" if retry else "restore_and_hold",
                        applied_beta=beta_next, reason="binary_regression_after_updates",
                        review_required=not retry)
        if beta >= r.max_beta:
            return dict(event, reason="beta_cap_review_binary_training" )
        if step - self.state["last_change"] < r.min_residence or len(segment) < r.min_residence or len(pairs) < r.evaluations:
            return dict(event, reason="minimum_same_beta_residence")
        if step - pairs[-1]["step"] > r.check_every:
            return dict(event, reason="stale_binary_evaluation")
        old, new = pairs[-r.evaluations:-r.evaluations//2], pairs[-r.evaluations//2:]
        before, after = mean_metrics(old), mean_metrics(new)
        gains = {key: after[key] - before[key] for key in METRICS}
        grayness = statistics.mean(row["grayness"] for row in new)
        fraction = statistics.mean(row["gray_fraction"] for row in new)
        gap = statistics.mean(row["mean_delta"] - row["bin_mean_delta"] for row in new)
        event.update(before=before, after=after, gains=gains, grayness=grayness,
                     gray_fraction=fraction, gray_binary_gap=gap,
                     binary_steps=[row["step"] for row in old + new])
        improving = ((gains["mean_delta"] > r.improvement and gains["min_delta"] >= -r.improvement)
                     or (gains["min_delta"] > r.improvement and gains["mean_delta"] >= -r.improvement))
        if improving and gains["mean_tx"] >= -r.transmission_tolerance and gains["mean_ty"] <= r.transmission_tolerance:
            return dict(event, reason="binary_improving")
        if fraction < r.gray_fraction_floor or grayness < r.grayness_floor or gap < r.gray_binary_gap_floor:
            return dict(event, reason="insufficient_gray_material_or_gap")
        candidate = min(r.max_beta, beta * self.state["growth"])
        return dict(event, action="increase", applied_beta=candidate,
                    reason="binary_stalled_with_large_gray_gap")

    def begin_trial(self, event, anchor):
        self.state.update(last_change=event["step"], pending=dict(
            start_step=event["step"], old_beta=event["old_beta"], new_beta=event["applied_beta"],
            attempt=1, anchor=str(anchor), baseline=None))

    def observe(self, row):
        pending = self.state.get("pending")
        if pending and row["step"] == pending["start_step"]:
            # The hard mask at this theta is independent of projection beta.
            pending["baseline"] = binary_metrics(row)
            pending["baseline_step"] = row["step"]
