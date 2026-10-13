"""Supervise an isolated Y-pass continuation on Windows or Linux."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import psutil
from filelock import FileLock
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
from portable_report import atomic, json_save, publish, read


def reconstruct_seed(out):
    seed = ROOT/'seed'
    manifest = read(seed/'manifest.json')
    arrays = {}
    for name, meta in manifest['arrays'].items():
        chunks = []
        for part in meta['parts']:
            path = seed/part['file']
            if hashlib.sha256(path.read_bytes()).hexdigest() != part['sha256']:
                raise ValueError('Seed checksum mismatch: '+part['file'])
            with np.load(path, allow_pickle=False) as saved:
                chunks.append(saved['array'])
        array = np.concatenate(chunks)
        if list(array.shape) != meta['shape'] or str(array.dtype) != meta['dtype'] or \
                hashlib.sha256(array.tobytes()).hexdigest() != meta['sha256']:
            raise ValueError('Reconstructed seed mismatch: '+name)
        arrays[name] = torch.from_numpy(array)
    if not all(torch.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Nonfinite seed')
    state = dict(arrays, step=-1, updates_completed=0, adam_updates=manifest['adam_updates'],
                 beta=manifest['beta'], beta_applied_step=-1, beta_events=[])
    torch.save(state, out/'seed_state.tmp')
    (out/'seed_state.tmp').replace(out/'state.pt')
    json_save(out/'seed_provenance.json', manifest)
    return manifest


def verified_updates(out):
    state = torch.load(out/'state.pt', map_location='cpu', weights_only=False)
    updates = int(state['updates_completed'])
    if updates < 0 or any(tuple(state[k].shape) != (100, 400, 400) or
                          not bool(torch.isfinite(state[k]).all()) for k in ('theta', 'm', 'v')):
        raise ValueError('Invalid checkpoint shape/values')
    history = read(out/'history.json', [])
    if updates and (not history or history[-1]['step'] not in (updates-1, updates)):
        raise ValueError('History and checkpoint do not agree')
    return updates


def worker_command(out, steps):
    options = read(ROOT/'settings.json')['arguments'][:]
    options[options.index('--max-updates')+1] = str(steps)
    offset = read(ROOT/'seed/manifest.json')['source_updates_completed']
    return [sys.executable, '-u', str(ROOT/'src/run_tio2_xypol_filter_voxel3d.py'),
            '--out', str(out), '--pass-pol', 'y', '--continuation-offset', str(offset), *options]


def live_worker(out):
    process = read(out/'process.json', {})
    try:
        p = psutil.Process(process['runner_pid'])
        if p.create_time() != process['create_time']:
            return None
        cmd = p.cmdline()
        if str(out) not in cmd or str(ROOT/'src/run_tio2_xypol_filter_voxel3d.py') not in cmd:
            return None
        return p if p.is_running() else None
    except (KeyError, psutil.Error):
        return None


def preflight(out):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Install the CUDA PyTorch wheel and an NVIDIA driver.')
    free, total = torch.cuda.mem_get_info()
    report = dict(torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), free_vram_gib=free/2**30, total_vram_gib=total/2**30,
        free_ram_gib=psutil.virtual_memory().available/2**30,
        free_disk_gib=shutil.disk_usage(out).free/2**30)
    json_save(out/'environment.json', report)
    if free < 14*2**30:
        raise RuntimeError('Less than 14 GiB free VRAM. Do not share this GPU with another large run.')
    if report['free_disk_gib'] < 25:
        raise RuntimeError('At least 25 GiB free disk is required for checkpoints and snapshots.')
    print(json.dumps(report, indent=2), flush=True)


def supervise(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out/'supervisor.lock'), timeout=0):
        if (out/'state.pt').exists() and not args.resume and not args.prepare_only:
            raise RuntimeError('Output already initialized. Use --resume or a different --out.')
        status = dict(status='preparing', supervisor_pid=os.getpid(), budget=args.steps)
        def update(**values):
            status.update(values, observed_at=time.time())
            json_save(out/'supervisor.json', status)
            publish(out, status)
        update()
        child = None
        try:
            worker = live_worker(out)
            if worker is None and not args.prepare_only:
                # Release the CUDA probe's context before starting the memory-heavy worker.
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--preflight-only',
                                '--out', str(out)], check=True)
            if not (out/'state.pt').exists():
                reconstruct_seed(out)
                gc.collect()
            if args.prepare_only:
                update(status='ready')
                print(f'Prepared: {out / "index.html"}')
                return
            (out/'PAUSE_REQUESTED').unlink(missing_ok=True)
            while True:
                worker = live_worker(out)
                if worker is not None:
                    update(status='running', worker_pid=worker.pid)
                    time.sleep(10)
                    continue
                if child is not None and child.wait() != 0:
                    raise RuntimeError('Worker failed; inspect worker.log. No blind retry was performed.')
                run = read(out/'run_status.json', {})
                updates = verified_updates(out)
                if run.get('status') == 'completed' and updates >= args.steps:
                    update(status='completed', worker_pid=None, updates_completed=updates)
                    return
                if (out/'PAUSE_REQUESTED').exists():
                    update(status='paused', worker_pid=None, updates_completed=updates)
                    return
                if run and run.get('status') not in ('completed', 'stopped_by_user', 'checkpoint_ready'):
                    raise RuntimeError('Unclean worker exit; inspect logs before explicitly resuming.')
                if updates > args.steps:
                    raise ValueError('Requested budget is below the saved update count')
                if shutil.disk_usage(out).free < 12*2**30:
                    raise RuntimeError('Disk space below 12 GiB; checkpoint retained.')
                stop = out/'STOP_REQUESTED'
                if updates < args.steps:
                    atomic(stop, 'Recycle worker after one committed update. Supervisor resumes it.\n')
                else:
                    stop.unlink(missing_ok=True)
                command = worker_command(out, args.steps)
                with (out/'worker.log').open('a', encoding='utf-8') as log:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                process = psutil.Process(child.pid)
                json_save(out/'process.json', dict(command=command, runner_pid=child.pid,
                    create_time=process.create_time(), resume_step=updates))
                update(status='running', worker_pid=child.pid, updates_completed=updates)
                time.sleep(2)
        except KeyboardInterrupt:
            atomic(out/'PAUSE_REQUESTED', 'User interrupted the supervisor.\n')
            atomic(out/'STOP_REQUESTED', 'Stop after the current complete checkpoint.\n')
            update(status='paused', detail='Current worker will finish its checkpoint; resume with --resume.')
        except Exception as exc:
            update(status='error', error=str(exc))
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT/'runs/y_pass')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--background', action='store_true')
    ap.add_argument('--preflight-only', action='store_true')
    args = ap.parse_args()
    if args.steps < 1:
        ap.error('steps must be positive')
    if args.preflight_only:
        args.out.mkdir(parents=True, exist_ok=True)
        preflight(args.out)
    elif args.background:
        args.out.resolve().mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', str(Path(__file__).resolve()),
                   *[x for x in sys.argv[1:] if x != '--background']]
        with (args.out.resolve()/'supervisor.log').open('a', encoding='utf-8') as log:
            p = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                start_new_session=os.name != 'nt')
        print(f'Supervisor PID {p.pid}. Dashboard: {args.out.resolve() / "index.html"}')
    else:
        supervise(args)


if __name__ == '__main__':
    main()
