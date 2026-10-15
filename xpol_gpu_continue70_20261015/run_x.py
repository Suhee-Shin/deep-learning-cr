"""Resume the frozen X-pass state at global 200 for exactly 70 more updates."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

from filelock import FileLock
import psutil
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
from portable_report import atomic, json_save, publish, read, save_history_plots, write_summary


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_code():
    manifest = read(ROOT/'seed/manifest.json')
    for name, expected in manifest['code_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen numerical source changed: '+name)


def verified_updates(out, shape=(100, 400, 400)):
    state = torch.load(out/'state.pt', map_location='cpu', weights_only=False)
    updates = int(state['updates_completed'])
    history = read(out/'history.json', [])
    rows = [r for r in history if r['step'] == state['step']]
    if len(rows) != 1 or not history or history[-1]['step'] != state['step']:
        raise ValueError('Checkpoint and final history row disagree')
    row = rows[0]
    if updates != state['step']+int(row['update_applied']) or state['adam_updates'] != updates or row['adam_updates'] != updates:
        raise ValueError('Adam/update/evaluation counters disagree')
    for key in ('theta', 'm', 'v'):
        if tuple(state[key].shape) != shape or not bool(torch.isfinite(state[key]).all()):
            raise ValueError('Invalid checkpoint: '+key)
    return updates


def restore(out):
    manifest = read(ROOT/'seed/manifest.json')
    if any((out/name).exists() for name in manifest['files']):
        raise RuntimeError('Partial/existing output; use a new output folder, never overwrite a checkpoint')
    with tempfile.TemporaryDirectory(prefix='restore_', dir=out) as folder:
        folder = Path(folder)
        archive = folder/'checkpoint.zip'
        with archive.open('wb') as stream:
            for part in manifest['parts']:
                path = ROOT/'seed'/part['file']
                if path.stat().st_size != part['bytes'] or digest(path) != part['sha256']:
                    raise ValueError('Checkpoint part corrupt: '+part['file'])
                with path.open('rb') as source:
                    shutil.copyfileobj(source, stream)
        if digest(archive) != manifest['archive_sha256']:
            raise ValueError('Joined checkpoint archive is corrupt')
        stage = folder/'files'
        stage.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            if set(bundle.namelist()) != set(manifest['files']):
                raise ValueError('Archive manifest differs')
            for name, meta in manifest['files'].items():
                destination = (stage/name).resolve()
                if not destination.is_relative_to(stage.resolve()):
                    raise ValueError('Unsafe archive member')
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(name) as source, destination.open('wb') as target:
                    shutil.copyfileobj(source, target)
                if destination.stat().st_size != meta['bytes'] or digest(destination) != meta['sha256']:
                    raise ValueError('Restored file mismatch: '+name)
        if verified_updates(stage) != 200:
            raise ValueError('Not the frozen step-200 continuation')
        for name in manifest['files']:
            destination = out/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            (stage/name).replace(destination)
    config = read(out/'config.json')
    config.update(out=str(out), max_updates=270, steps=271,
                  voxel_src=str(ROOT/'third_party/information-optimal-color-router-latest/src'))
    json_save(out/'config.json', config)
    json_save(out/'run_status.json', dict(status='checkpoint_ready', evaluated_step=200,
                                        updates_completed=200, max_updates=270))
    json_save(out/'seed_provenance.json', {k: v for k, v in manifest.items() if k not in ('parts', 'files', 'code_sha256')})
    history, best = read(out/'history.json'), read(out/'best.json')
    save_history_plots(out, history, best)
    write_summary(out, [], history, best, argparse.Namespace(max_updates=270))
    gc.collect()


def worker_command(out):
    settings = read(ROOT/'settings.json')
    return [sys.executable, '-u', str(ROOT/'worker_entry.py'), '--out', str(out), *settings['arguments']]


def live_worker(out):
    info = read(out/'process.json', {})
    try:
        proc = psutil.Process(info['runner_pid'])
        if proc.create_time() != info['create_time']:
            return None
        command = proc.cmdline()
        return proc if str(ROOT/'worker_entry.py') in command and str(out) in command else None
    except (KeyError, psutil.Error):
        return None


def preflight(out):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Install NVIDIA driver and CUDA PyTorch first.')
    free, total = torch.cuda.mem_get_info()
    report = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), free_vram_gib=free/2**30, total_vram_gib=total/2**30,
        free_ram_gib=psutil.virtual_memory().available/2**30,
        free_disk_gib=shutil.disk_usage(out).free/2**30)
    json_save(out/'environment.json', report)
    if free < 14*2**30 or report['free_ram_gib'] < 24 or report['free_disk_gib'] < 25:
        raise RuntimeError('Need 14 GiB free VRAM, 24 GiB free RAM and 25 GiB free disk; see environment.json')
    print(json.dumps(report, indent=2), flush=True)


def supervise(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out/'supervisor.lock'), timeout=0):
        if (out/'state.pt').exists() and not (args.resume or args.prepare_only):
            raise RuntimeError('Already initialized; use --resume to preserve and continue the saved state.')
        verify_code()
        status = dict(status='preparing', supervisor_pid=os.getpid(), start_step=200, budget=270)
        def update(**values):
            status.update(values, observed_at=time.time())
            json_save(out/'supervisor.json', status)
            publish(out, status)
        child = None
        update()
        try:
            if not (out/'state.pt').exists():
                restore(out)
            if args.prepare_only:
                update(status='ready', updates_completed=verified_updates(out))
                print('Prepared (no optical run): '+str(out/'index.html'))
                return
            (out/'PAUSE_REQUESTED').unlink(missing_ok=True)
            if live_worker(out) is None:
                # Explicit recovery is allowed only when the last history and state agree.
                saved_run = read(out/'run_status.json', {})
                if args.resume and saved_run.get('status') == 'running':
                    updates = verified_updates(out)
                    json_save(out/'run_status.json', dict(saved_run, status='checkpoint_ready',
                                                         updates_completed=updates))
                subprocess.run([sys.executable, str(ROOT/'run_x.py'), '--out', str(out), '--preflight-only'], check=True)
            while True:
                worker = live_worker(out)
                if worker is not None:
                    update(status='running', worker_pid=worker.pid)
                    time.sleep(10)
                    continue
                if child is not None and child.wait() != 0:
                    raise RuntimeError('Worker failed; inspect worker.log before explicit --resume. No blind retry.')
                updates = verified_updates(out)
                run = read(out/'run_status.json', {})
                if updates > 270 or updates < 200:
                    raise ValueError('Unexpected saved update count')
                if run.get('status') == 'completed' and updates == 270 and run.get('evaluated_step') == 270:
                    update(status='completed', worker_pid=None, updates_completed=updates)
                    return
                if (out/'PAUSE_REQUESTED').exists():
                    update(status='paused', worker_pid=None, updates_completed=updates)
                    return
                if run.get('status') not in ('completed', 'stopped_by_user', 'checkpoint_ready'):
                    raise RuntimeError('Unclean exit. Check checkpoint/log consistency before restarting.')
                if shutil.disk_usage(out).free < 12*2**30:
                    raise RuntimeError('Less than 12 GiB disk space; checkpoint retained.')
                if updates < 270:
                    atomic(out/'STOP_REQUESTED', 'Recycle worker after one committed update; supervisor resumes.\n')
                else:
                    (out/'STOP_REQUESTED').unlink(missing_ok=True)
                command = worker_command(out)
                with (out/'worker.log').open('a', encoding='utf-8') as log:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                proc = psutil.Process(child.pid)
                json_save(out/'process.json', dict(command=command, runner_pid=child.pid,
                    create_time=proc.create_time(), resume_step=updates))
                update(status='running', worker_pid=child.pid, updates_completed=updates)
                time.sleep(2)
        except KeyboardInterrupt:
            atomic(out/'PAUSE_REQUESTED', 'User requested pause.\n')
            atomic(out/'STOP_REQUESTED', 'Stop after current complete checkpoint.\n')
            update(status='pausing', detail='Worker finishes its checkpoint before exiting.')
        except Exception as exc:
            update(status='error', error=str(exc))
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT/'runs/x_pass_270')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--preflight-only', action='store_true')
    ap.add_argument('--background', action='store_true')
    args = ap.parse_args()
    if args.preflight_only:
        args.out.mkdir(parents=True, exist_ok=True)
        preflight(args.out)
    elif args.background:
        args.out.resolve().mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', str(ROOT/'run_x.py'), *[x for x in sys.argv[1:] if x != '--background']]
        with (args.out.resolve()/'supervisor.log').open('a', encoding='utf-8') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                start_new_session=os.name != 'nt')
        print(f'Supervisor PID {child.pid}. Dashboard: {args.out.resolve()/"index.html"}')
    else:
        supervise(args)


if __name__ == '__main__':
    main()
