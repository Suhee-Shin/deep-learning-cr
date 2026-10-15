"""Adopt a running job, retry bounded failures, and verify strict binary targets."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import psutil

from voxel_dashboard import read_json, write_changed, publish_pipeline_dashboard
from voxel_goal_policy import continuation_command, strict_target

ROOT = Path(__file__).resolve().parent


def active(pid, script, out):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        command = " ".join(psutil.Process(pid).cmdline())
        return script in command and str(out) in command
    except (psutil.Error, TypeError):
        return False


def stamp():
    return datetime.now().astimezone().isoformat()


def write_status(out, **values):
    status = dict(read_json(out / "goal_watch.json", {}), **values,
                  watcher_pid=os.getpid(), observed_at=stamp())
    write_changed(out / "goal_watch.json", json.dumps(status, indent=2))


def validated_checkpoint(out, records):
    import torch
    state = torch.load(out / "state.pt", map_location="cpu", weights_only=False)
    if not all(torch.isfinite(state[key]).all().item() for key in ("theta", "m", "v")):
        raise ValueError("Non-finite checkpoint: diagnosis required")
    saved = next((r for r in reversed(records) if r["step"] == state["step"]), None)
    if not saved or state["adam_updates"] != saved["adam_updates"] or \
            state["updates_completed"] != state["step"] + int(saved["update_applied"]):
        raise ValueError("Checkpoint and evaluated history disagree: diagnosis required")
    return int(state["updates_completed"])


def request_checkpoint_reload(out, reason, *, kind="code_reload"):
    """Ask the existing watcher for a checkpoint-boundary code reload, not a reset."""
    if kind not in ("code_reload", "worker_recycle"):
        raise ValueError("Unknown checkpoint reload kind")
    watched = read_json(out / "goal_watch.json", {})
    process = read_json(out / "process.json", {})
    pending = read_json(out / "goal_watch_state.json", {}).get("pending")
    if watched.get("status") != "monitoring" or pending is not None or \
            not active(watched.get("watcher_pid"), "watch_voxel_goal.py", out) or \
            not active(process.get("runner_pid"), "run_tio2_xypol_filter_voxel3d.py", out):
        raise RuntimeError("Reload requires a verified live runner and monitoring watcher")
    stop_path = out / "STOP_REQUESTED"
    if stop_path.exists() or (out / "GOAL_STOP_REQUESTED").exists():
        raise RuntimeError("An existing stop request must not be overwritten")
    archive = out / "maintenance" / datetime.now().strftime("%Y%m%d_%H%M%S")
    archive.mkdir(parents=True)
    for name in ("run_tio2_xypol_filter_voxel3d.py", "voxel_memory_forward.py", "watch_voxel_goal.py"):
        shutil.copy2(ROOT / name, archive / name)
    request = dict(status="requested", requested_at=stamp(), reason=reason, kind=kind,
                   runner_pid=process["runner_pid"], watcher_pid=watched["watcher_pid"],
                   iteration=read_json(out / "iteration_status.json", {}), code_archive=str(archive),
                   command_unchanged=True, checkpoint_boundary_only=True)
    write_changed(archive / "request.json", json.dumps(request, indent=2))
    write_changed(out / "checkpoint_reload_request.json", json.dumps(request, indent=2))
    write_changed(stop_path, json.dumps(dict(owner="voxel_goal_watch", out=str(out))))
    return request


def archive_restart(out, archive, full_checkpoint, step, updates):
    """Keep the latest resumable state untouched; describe historical retention explicitly."""
    archive.mkdir(parents=True)
    source = out / "state.pt"
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    names = ["config.json", "history.json", "process.json", "run_status.json", "completion.json"]
    if full_checkpoint:
        names.insert(0, "state.pt")
    for name in names:
        if (out / name).exists():
            shutil.copy2(out / name, archive / name)
    manifest = dict(step=step, updates_completed=updates, sha256=digest,
                    size_bytes=source.stat().st_size, full_checkpoint_archived=full_checkpoint,
                    retention="full" if full_checkpoint else "routine_restart_metadata_only",
                    source_checkpoint=str(source), source_is_mutable=True,
                    historical_checkpoint="state.pt" if full_checkpoint else None,
                    archived_at=stamp())
    write_changed(archive / "checkpoint_manifest.json", json.dumps(manifest, indent=2))
    return manifest


def main():
    # A live runner may import only request_checkpoint_reload with an older cached policy module.
    from voxel_goal_policy import full_restart_checkpoint
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--request-checkpoint-reload", metavar="REASON")
    args = parser.parse_args()
    root, out = args.pipeline.resolve(), args.out.resolve()
    if args.request_checkpoint_reload:
        print(json.dumps(request_checkpoint_reload(out, args.request_checkpoint_reload), indent=2), flush=True)
        return
    prior = read_json(out / "goal_watch.json", {})
    if active(prior.get("watcher_pid"), "watch_voxel_goal.py", out):
        raise RuntimeError("Goal watcher is already running")
    own_stop = json.dumps(dict(owner="voxel_goal_watch", out=str(out)))
    stop_path = out / "STOP_REQUESTED"
    state = read_json(out / "goal_watch_state.json", dict(attempts={}, checked=[], pending=None))
    write_status(out, status="monitoring", strict_tx_gt=.8, strict_ty_lt=.1,
                 started_at=stamp(), initial_budget=500, continuation_block=100,
                 dense_validation_wavelengths_nm=list(range(400, 701, 10)))
    print(f"Goal watcher adopting {out}", flush=True)
    while True:
        try:
            if (out / "GOAL_STOP_REQUESTED").exists():
                write_status(out, status="stopped_by_user")
                return
            config = read_json(out / "config.json", {})
            records = read_json(out / "history.json", [])
            process = read_json(out / "process.json", {})
            status = read_json(out / "run_status.json", {})
            running = active(process.get("runner_pid"), "run_tio2_xypol_filter_voxel3d.py", out)
            managed_stop = stop_path.exists() and stop_path.read_text(encoding="utf-8") == own_stop
            if stop_path.exists() and not managed_stop:
                write_status(out, status="user_stop_observed", runner_active=running)
                return
            row = records[-1] if records else {}
            write_status(out, status="monitoring" if running else "checking_exit", runner_active=running,
                         evaluated_step=row.get("step"), runner_pid=process.get("runner_pid"))
            train_wls = [float(w) for w in config.get("wavelengths", "").split(",") if w]
            candidates = [r for r in records if r["step"] <= status.get("evaluated_step", -1)
                          and r["step"] not in state["checked"]
                          and strict_target(r.get("bin_spectrum", []), train_wls)]
            if candidates and state["pending"] is None:
                state["pending"] = candidates[-1]["step"]
                write_changed(stop_path, own_stop)
                write_changed(out / "goal_watch_state.json", json.dumps(state, indent=2))
                managed_stop = True
                print(f"Requesting checkpoint boundary for binary candidate {state['pending']}", flush=True)
            if running:
                iteration = read_json(out / "iteration_status.json", {})
                age = time.time() - iteration.get("observed_at", iteration.get("started_at", time.time()))
                if age > 5400:
                    write_status(out, status="long_phase_needs_inspection", no_progress_seconds=age)
                time.sleep(args.poll_seconds)
                continue
            supervisor = process.get("supervisor_pid")
            if supervisor != os.getpid() and any(active(supervisor, script, out) for script in (
                    "resume_voxel_after_report_failure.py", "recover_voxel_fourpoint_launch.py",
                    "launch_voxel_fourpoint_cached.py")):
                time.sleep(args.poll_seconds)
                continue
            if state["pending"] is not None:
                step = state["pending"]
                write_status(out, status="dense_binary_validation", candidate_step=step)
                folder = out / "goal_validation"
                folder.mkdir(exist_ok=True)
                command = [sys.executable, "-u", str(ROOT / "evaluate_voxel_goal_candidate.py"),
                           "--out", str(out), "--step", str(step)]
                with (folder / f"step_{step:04d}.log").open("a", encoding="utf-8") as log:
                    validation = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                  creationflags=subprocess.CREATE_NO_WINDOW)
                    while validation.poll() is None:
                        write_status(out, status="dense_binary_validation", candidate_step=step,
                                     validation_pid=validation.pid)
                        publish_pipeline_dashboard(root)
                        time.sleep(args.poll_seconds)
                report = read_json(folder / f"step_{step:04d}.json", {})
                if validation.returncode or report.get("status") != "completed":
                    raise RuntimeError("Dense binary validation failed; inspect validation log")
                state["checked"].append(step)
                state["pending"] = None
                write_changed(out / "goal_watch_state.json", json.dumps(state, indent=2))
                if report.get("passed") and strict_target(report.get("spectrum", []), range(400, 701, 10)):
                    write_changed(out / "goal_result.json", json.dumps(report, indent=2))
                    write_status(out, status="binary_sampled_target_met", accepted_step=step,
                                 continuous_band_certified=False, order_convergence_certified=False)
                    print(f"Strict binary sampled target met at step {step}", flush=True)
                    return
            if status.get("status") == "stopped_by_user" and not managed_stop:
                write_status(out, status="user_stop_observed")
                return
            updates = validated_checkpoint(out, records)
            extending = updates >= config["max_updates"]
            if not extending and not managed_stop:
                log_text = (out / "run.log").read_text(encoding="utf-8", errors="replace")
                last_start = max(log_text.rfind("[goal-resume]"), log_text.rfind("[report-recovery]"), 0)
                latest_log = log_text[last_start:].lower()
                if any(reason in latest_log for reason in (
                        "non-finite gradient", "gradient outlier", "response contains non-finite",
                        "changed physics/objective", "parent density shape does not match")):
                    raise RuntimeError("Numerical/configuration guard stopped the run; agent diagnosis required")
                key = str(updates)
                attempts = state["attempts"].get(key, 0)
                if attempts >= 3:
                    raise RuntimeError(f"Three failures at update {updates}; agent diagnosis required")
                state["attempts"][key] = attempts + 1
                write_changed(out / "goal_watch_state.json", json.dumps(state, indent=2))
                write_status(out, status="restart_backoff", retry=attempts+1, next_step=updates)
                time.sleep((30, 120, 300)[attempts])
                if (out / "GOAL_STOP_REQUESTED").exists() or (stop_path.exists() and not managed_stop):
                    continue
            archive = out / "goal_restarts" / datetime.now().strftime("%Y%m%d_%H%M%S")
            full_checkpoint = full_restart_checkpoint(read_json(out / "worker_recycle.json", {}),
                row, status, process, read_json(out / "checkpoint_reload_request.json", {}), managed_stop)
            archive_restart(out, archive, full_checkpoint, row.get("step"), updates)
            command = continuation_command(process["command"], config, updates)
            write_changed(archive / "restart.json", json.dumps(dict(command=command, updates=updates,
                          budget_extended=extending, theta_adam_beta_preserved=True, created_at=stamp()), indent=2))
            if managed_stop:
                stop_path.unlink()
            (out / "completion.json").unlink(missing_ok=True)
            with (out / "run.log").open("a", encoding="utf-8") as log:
                log.write(f"\n[goal-resume] {stamp()} next_step={updates} extend={extending}\n")
                log.flush()
                child = subprocess.Popen(command, cwd=ROOT.parent, stdout=log, stderr=subprocess.STDOUT,
                                         creationflags=subprocess.CREATE_NO_WINDOW)
            write_changed(out / "process.json", json.dumps(dict(command=command, supervisor_pid=os.getpid(),
                          runner_pid=child.pid, started_at=stamp(), resume_step=updates), indent=2))
            status.update(status="running")
            status.pop("exit_code", None)
            write_changed(out / "run_status.json", json.dumps(status, indent=2))
            pipeline = read_json(root / "pipeline_status.json", {})
            for key in ("finished_at", "exit_code"):
                pipeline.pop(key, None)
            pipeline.update(stage="optimization", supervisor_pid=os.getpid(), runner_pid=child.pid,
                            last_resume_at=stamp(), last_resume_step=updates, goal_managed=True,
                            max_updates=updates+100 if extending else config["max_updates"])
            write_changed(root / "pipeline_status.json", json.dumps(pipeline, indent=2))
            print(f"Goal resumed runner {child.pid} from update {updates}", flush=True)
        except Exception:
            error = traceback.format_exc()
            print(error, flush=True)
            write_status(out, status="needs_agent_diagnosis", error=error)
            return
        finally:
            try:
                publish_pipeline_dashboard(root)
            except Exception as exc:
                print(f"[dashboard-warning] {exc}", flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
