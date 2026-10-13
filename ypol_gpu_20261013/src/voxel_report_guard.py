"""Keep optional presentation failures separate from numerical/checkpoint failures."""
from datetime import datetime
import json
from pathlib import Path
import traceback

from voxel_dashboard import read_json, write_changed


def report_task(out, step, name, callback, *args, **kwargs):
    out = Path(out)
    try:
        callback(*args, **kwargs)
    except Exception:
        error = dict(step=step, task=name, timestamp=datetime.now().astimezone().isoformat(),
                     traceback=traceback.format_exc())
        print(f"[report-warning] step={step} task={name}\n{error['traceback']}", flush=True)
        try:
            with (out / "report_errors.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(error, ensure_ascii=False) + "\n")
            status = read_json(out / "report_health.json", {})
            status[name] = error
            write_changed(out / "report_health.json", json.dumps(status, indent=2))
        except Exception as exc:
            print(f"[report-warning] Could not persist report error: {exc}", flush=True)
        return False
    try:
        status = read_json(out / "report_health.json", {})
        status.pop(name, None)
        write_changed(out / "report_health.json", json.dumps(status, indent=2))
    except Exception as exc:
        print(f"[report-warning] Could not update report health: {exc}", flush=True)
    return True
