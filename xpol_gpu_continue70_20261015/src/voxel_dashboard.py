"""Static, file:// compatible run dashboard. Never imports or changes the solver."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.parse import quote


TEMPLATE = Path(__file__).with_name("voxel_dashboard.html")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_changed(path, text):
    path = Path(path)
    # Windows readers may briefly prevent replacement; never truncate the live file.
    for attempt in range(5):
        temporary = None
        try:
            if path.exists() and path.read_text(encoding="utf-8") == text:
                return
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(text)
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.05 * 2**attempt)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


def dashboard_payload(out, records, best, config, finished=False):
    out = Path(out)
    row = records[-1] if records else {}
    binary_row = next((r for r in reversed(records) if r.get("bin_spectrum")), {})
    graph = read_json(out / "graph_refresh.json", {})
    status = read_json(out / "run_status.json", {})
    process = read_json(out / "process.json", {})
    worker_running = False
    if process.get("runner_pid"):
        import psutil
        try:
            command = " ".join(psutil.Process(process["runner_pid"]).cmdline())
            worker_running = "run_tio2_xypol_filter_voxel3d.py" in command and str(out.resolve()) in command
        except (psutil.Error, TypeError, ValueError):
            pass
    figures = {}
    for name in ("step_metrics.png", "worst_band_history.png", "wavelength_history.png",
                 "beta_history.png", "gray_binary_gap.png", "best_binary_spectrum.png",
                 "structure_evolution_3d.gif", "best_binary_layers.png"):
        path = out / name
        if path.exists():
            figures[name] = str(path.stat().st_mtime_ns)
    snapshots = sorted((out / "snapshots").glob("step_*_3d.png"))
    structure = []
    for path in dict.fromkeys(snapshots[:1] + snapshots[-1:]):
        name = path.relative_to(out).as_posix()
        figures[name] = str(path.stat().st_mtime_ns)
        structure.append({"file": name, "step": int(path.stem.split("_")[1])})
    config_keys = ("max_updates", "orders", "actual_pixel_nm", "layers", "layer_height_nm",
                   "period_um", "wavelengths", "plot_every", "objective_mode", "target_tx", "target_ty", "gray_plateau_beta", "eval_every",
                   "init_correlation_nm", "sigma_nm", "sigma_z_nm", "softmin_start_step", "binary_continuation_rules",
                   "z_filter_boundary", "binary_evaluation", "dtype", "holdout_wavelengths", "holdout_every", "cache_eigenpairs",
                   "binary_forward_ste", "gray_penalty_max")
    history_path = out / "history.json"
    resource_hold = read_json(out.parent/"RESOURCE_HOLD.json", {})
    return dict(row=row, binary_row=binary_row, iteration=read_json(out/"iteration_status.json", {}),
                resource_hold=resource_hold,
                audit=read_json(out/"gradient_audit.json", {}),
                holdout=read_json(out/"holdout_latest.json", {}),
                goal=read_json(out/"goal_watch.json", {}),
                report_health=read_json(out/"report_health.json", {}),
                best=best, graph=graph, structure=structure, figures=figures,
                config={k: config.get(k) for k in config_keys},
                status="resource_hold" if resource_hold and not worker_running else "finished" if finished else
                       "running" if worker_running else status.get("status", "unknown"),
                worker_running=worker_running,
                observed_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                evaluated_at=(datetime.fromtimestamp(history_path.stat().st_mtime).astimezone()
                              .isoformat(timespec="seconds") if history_path.exists() else None),
                run_name=out.name)


def publish_pipeline_dashboard(folder):
    """Publish one stable feed while the solver moves from audit to optimization."""
    folder = Path(folder).resolve()
    pipeline = read_json(folder/"pipeline_status.json", {})
    stage = pipeline.get("stage", "gradient_audit")
    if stage in ("gradient_audit", "blocked_by_gradient_audit"):
        active = folder/"gradient_audit"
        report = read_json(active/"gradient_audit.json", {})
        config = report.get("config", {})
        worker_pid = pipeline.get("audit_pid")
    else:
        active = Path(pipeline.get("training_directory", str(folder/"random10_corr100_staged_h1000_o10")))
        config = read_json(active/"config.json", {})
        worker_pid = read_json(active/"process.json", {}).get("runner_pid")
    if not config:
        return None
    payload = dashboard_payload(active, read_json(active/"history.json", []), read_json(active/"best.json", {}), config)
    import psutil
    running = False
    if worker_pid:
        try:
            proc = psutil.Process(worker_pid)
            expected_script = ("audit_voxel_staged_gradient.py" if active.name == "gradient_audit" else
                               "benchmark_voxel_eigen_replay.py" if stage == "eigen_cache_comparison" else
                               "run_tio2_xypol_filter_voxel3d.py")
            running = proc.is_running() and expected_script in " ".join(proc.cmdline())
        except psutil.Error:
            pass
    payload["pipeline"] = dict(pipeline, worker_pid=worker_pid, worker_running=running)
    payload["asset_prefix"] = quote(os.path.relpath(active, folder).replace("\\", "/"), safe="/.") + "/"
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    write_changed(folder/"pipeline_data.js", f"window.receiveDashboard({encoded});\n")
    write_changed(folder/"index.html", render_dashboard(payload, payload["asset_prefix"]))
    return payload


def render_dashboard(payload, asset_prefix="./"):
    # Escape embedded JSON as script data; all dynamic text uses textContent.
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    prefix = json.dumps(asset_prefix).replace("<", "\\u003c")
    return (TEMPLATE.read_text(encoding="utf-8")
            .replace("__INITIAL_DATA__", encoded).replace("__ASSET_PREFIX__", prefix))


def publish_dashboard(out, records, best, config, finished=False, aliases=()):
    out = Path(out).resolve()
    payload = dashboard_payload(out, records, best, config, finished)
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    artifacts = [(out / "dashboard_data.js", f"window.receiveDashboard({encoded});\n")]
    for folder in [out] + [Path(p).resolve() for p in aliases]:
        prefix = quote(os.path.relpath(out, folder).replace("\\", "/"), safe="/.") + "/"
        page = render_dashboard(payload, prefix)
        artifacts.append((folder / "index.html", page))
        if folder == out:
            artifacts.append((folder / "dashboard.html", page))
    for path, content in artifacts:
        try:
            write_changed(path, content)
        except OSError as exc:
            # A preview is expendable; it must not abort a saved optimizer step.
            print(f"[dashboard-warning] Refresh deferred for {path.name}: {exc}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alias", type=Path, action="append", default=[])
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--pipeline", action="store_true", help="Follow pipeline_status without changing solver state")
    args = parser.parse_args()
    if not args.out.is_dir() or any(not p.is_dir() for p in args.alias):
        parser.error("Run and alias folders must already exist")
    if args.interval < 1:
        parser.error("Interval must be at least one second")
    if args.watch:
        write_changed(args.out / "dashboard_process.json", json.dumps({
            "pid": os.getpid(), "started_at": datetime.now().isoformat(),
            "purpose": "Read-only solver data; refresh HTML/JS only", "aliases": list(map(str, args.alias))}, indent=2))
    while True:
        if args.pipeline:
            try:
                payload = publish_pipeline_dashboard(args.out)
            except (OSError, ValueError) as exc:
                print(f"Pipeline dashboard retry: {exc}", flush=True)
                payload = None
            terminal = (payload and payload["pipeline"].get("stage") != "gradient_audit"
                        and payload["status"] in ("completed", "failed", "validation_failed", "stopped_by_user"))
            if not args.watch or terminal:
                break
            time.sleep(args.interval)
            continue
        records = read_json(args.out / "history.json", [])
        config = read_json(args.out / "config.json")
        best = read_json(args.out / "best.json", {})
        # The legacy live process can briefly expose partially written JSON.
        if records is not None and config and best is not None:
            try:
                payload = publish_dashboard(args.out, records, best, config, aliases=args.alias)
                if payload["status"] in ("finished", "completed", "complete", "failed", "error", "stopped_by_user", "stopped", "paused"):
                    break
            except (OSError, ValueError) as exc:
                print(f"Dashboard retry: {exc}", flush=True)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
