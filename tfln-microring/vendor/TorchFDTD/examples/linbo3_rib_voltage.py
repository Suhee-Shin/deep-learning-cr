"""Finite-thickness X-cut rib-ring voltage traces in principal-axis coordinates.

Default geometry is the 20 um research ring. Local radial impressed sources
excite ring modes; this is NOT yet a verified TE0 bus-port simulation.
No electrode metal/loss is included. DC field is the explicit effective arc
approximation. Use --plan first, then a short profile before a long run.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import platform

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torchfdtd import (AdjointOptions, ElectroOpticSimulation, LithiumNiobate,
                       Monitor, Project, Region, Source, effective_arc_field_per_volt,
                       x_cut_simulation_frame)
from torchfdtd.anisotropy import TensorDielectricSimulation


def project_for(args):
    margin = (args.pml_cells + 2)*args.mesh_um + .6
    lateral = 2*(args.radius_um + .7 + margin)
    # Physical substrate top is lab z=0. Region is centred on its own origin.
    z_bottom = -2.
    z_top = .6 + (args.pml_cells + 2)*args.mesh_z_um + .8
    z_origin = (z_top + z_bottom)/2
    region = Region(dimension='3d', size=(lateral, lateral, z_top-z_bottom),
                    mesh_steps=(args.mesh_um, args.mesh_um, args.mesh_z_um),
                    steps=args.steps, backend=args.device, precision=args.precision,
                    pml_cells=args.pml_cells, material_sampling='yee')
    source = Source(component='Ex', center=(args.radius_um, 0., .35-z_origin),
                    wavelength=1.55, pulse='gaussian', time_definition='standard',
                    pulse_length=12e-15, pulse_offset=60e-15)
    monitors = [Monitor(name=f'ring_{angle}_{component}', component=component,
                       center=(args.radius_um*math.cos(math.radians(angle)),
                               args.radius_um*math.sin(math.radians(angle)), .35-z_origin))
                for angle in (0, 45, 90, 180) for component in ('Ex', 'Ey', 'Ez')]
    return Project(name='lab_shshin-ln-rib-voltage', region=region,
                   sources=[source], monitors=monitors), z_origin


def build_model(args, project, z_origin):
    dtype = getattr(torch, args.precision)
    axes = [torch.tensor(n[:-1].copy(), dtype=dtype, device=args.device)
            for n in project.region.mesh_nodes]
    x, y, z = torch.meshgrid(*axes, indexing='ij')
    lab_z = z + z_origin
    radial = (torch.sqrt(x*x+y*y)-args.radius_um).abs() <= .7
    slab = (lab_z >= 0.) & (lab_z < .1)
    ridge = radial & (lab_z >= .1) & (lab_z < .6)
    mask = slab | ridge
    background = torch.where(lab_z < 0., args.substrate_index**2, 1.)
    base = torch.diag_embed(background[..., None].expand(*background.shape, 3))
    q, rotation = x_cut_simulation_frame(45.)
    q = torch.as_tensor(q, dtype=dtype, device=args.device)
    # Rotationally invariant ring; arc angles are specified in the LAB frame.
    xy_lab = torch.stack((x*q[0, 0]+y*q[0, 1], x*q[1, 0]+y*q[1, 1]), dim=-1)
    field_lab = effective_arc_field_per_volt(
        xy_lab, mask & radial, effective_distance_um=args.effective_distance_um,
        start_deg=args.arc_start_deg, span_deg=args.arc_span_deg)
    return ElectroOpticSimulation(
        project, base, mask, field_lab @ q, rotation=rotation,
        material=LithiumNiobate(r33=30.9e-12),
        options=AdjointOptions(checkpoints=0), cpml_material='tensor')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--radius-um', type=float, default=20.)
    p.add_argument('--mesh-um', type=float, default=.1)
    p.add_argument('--mesh-z-um', type=float, default=.05)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--pml-cells', type=int, default=8)
    p.add_argument('--precision', choices=['float32', 'float64'], default='float32')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--voltages', type=float, nargs='+', default=[0., -1., 1., -2., 2.])
    p.add_argument('--effective-distance-um', type=float, default=2.5)
    p.add_argument('--arc-start-deg', type=float, default=135.)
    p.add_argument('--arc-span-deg', type=float, default=180.)
    p.add_argument('--substrate-index', type=float, default=1.444,
                   help='Nondispersive exploratory SiO2 index, not a paper-calibrated fit')
    p.add_argument('--plan', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.radius_um <= .7 or not all(math.isfinite(v) for v in args.voltages):
        p.error('Require radius > 0.7 um and finite voltages')
    torch.set_num_threads(1)
    project, z_origin = project_for(args)
    parameters = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                  if k not in ('out', 'plan')}
    code_hashes = {name: hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
                   for name in ('examples/linbo3_rib_voltage.py', 'torchfdtd/electro_optic.py',
                                'torchfdtd/anisotropy.py', 'torchfdtd/differentiable.py',
                                'torchfdtd/tensor_linalg.py')}
    fingerprint = hashlib.sha256(json.dumps([parameters, code_hashes], sort_keys=True).encode()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out/'run.json'
    if metadata_path.exists() and json.loads(metadata_path.read_text())['fingerprint'] != fingerprint:
        raise ValueError('Output contains a different run. Select a new output directory.')
    metadata = dict(fingerprint=fingerprint, parameters=parameters, code_hashes=code_hashes,
                    execution_environment=dict(python=sys.executable, torch=torch.__version__,
                        cuda_build=torch.version.cuda, hostname=platform.node(),
                        gpu=torch.cuda.get_device_name() if args.device=='cuda' else None),
                    project_json_role='Source/monitor/grid description only; run this script to reconstruct dielectric tensors',
                    grid=project.region.shape, time_step_s=project.region.time_step,
                    duration_s=project.region.time_step*args.steps,
                    lab_z_of_sim_origin_um=z_origin,
                    six_field_bytes=math.prod(project.region.shape)*6*(8 if args.precision=='float64' else 4),
                    status='planned', limitations=[
                        'Isolated ring: intrinsic modes, no loaded bus response',
                        'Local radial excitation; TE0 identity not verified',
                        'Effective radial DC field confined to ring footprint; no electrostatic fringing or metal loss',
                        'Staircase interfaces, nondispersive optical indices; convergence required',
                        'Short default run is a throughput profile, not resonance data'])
    metadata_path.write_text(json.dumps(metadata, indent=2))
    project.save(args.out/'project.json')
    # Project JSON alone cannot encode tensor arrays; reconstruct via this script.
    solver = TensorDielectricSimulation(project, AdjointOptions(checkpoints=0), cpml_material='tensor')
    reservation = solver.reservation(device=args.device)
    metadata['reservation'] = reservation
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)
    if args.plan:
        return
    if args.device == 'cuda':
        free, total = torch.cuda.mem_get_info()
        if free < total/2 or metadata['six_field_bytes'] > total/2:
            raise RuntimeError('GPU half-memory guard: ask the user before proceeding.')
    model = build_model(args, project, z_origin)
    completed = []
    with torch.no_grad():
        for index, voltage in enumerate(args.voltages):
            target = args.out/f'trace_{index:03d}.npz'
            if target.exists():
                with np.load(target) as data:
                    if str(data['fingerprint']) != fingerprint or float(data['voltage_v']) != voltage:
                        raise ValueError('Existing trace does not match this run')
                completed.append(index)
                continue
            if args.device == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            result = model(voltage)
            if args.device == 'cuda':
                torch.cuda.synchronize()
            signals = result.signals.detach().cpu().numpy()
            if not np.isfinite(signals).all() or not np.any(signals):
                raise RuntimeError('Missing or nonfinite fields; do not interpret as a resonance')
            elapsed = time.perf_counter()-start
            temporary = target.with_suffix('.tmp.npz')
            peak_bytes = torch.cuda.max_memory_allocated() if args.device=='cuda' else 0
            np.savez(temporary, signals=signals, time_step_s=result.time_step,
                     voltage_v=voltage, fingerprint=fingerprint, elapsed_s=elapsed,
                     cuda_peak_allocated_bytes=peak_bytes,
                     components=np.array([m.component for m in project.monitors]))
            temporary.replace(target)
            completed.append(index)
            metadata.update(status='running', completed_indices=completed)
            metadata_path.write_text(json.dumps(metadata, indent=2))
            print(f'{voltage:+g} V: {elapsed:.3f} s; CUDA peak {peak_bytes} bytes; finite trace saved to {target}', flush=True)
    metadata.update(status='traces_complete_not_resonance_validated', completed_indices=completed)
    metadata_path.write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
