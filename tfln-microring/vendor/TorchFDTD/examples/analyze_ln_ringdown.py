"""Extract provisional voltage-dependent peaks from saved LN ring traces.

FFT interpolation does not establish physical accuracy or mode identity.
Require independent time/mesh convergence and TE0 verification before reporting
these candidates as device tuning measurements.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.fft import next_fast_len, rfft, rfftfreq
from scipy.signal import find_peaks

C0 = 299792458.


def peak_from_trace(signals, dt, *, start_s, target_um, half_window_um,
                    minimum_duration_s=10e-12):
    signals = np.asarray(signals)
    if (signals.ndim != 2 or not np.isfinite(signals).all()
            or not np.isfinite(dt) or dt <= 0 or not np.isfinite(start_s) or start_s < 0):
        raise ValueError('Require finite 2D signals, positive dt and nonnegative start.')
    if not 0 < half_window_um < target_um:
        raise ValueError('Invalid positive wavelength search window.')
    first = int(np.ceil(start_s/dt))-1
    first = max(0, first)
    trace = signals[first:].astype(np.float64)
    duration = len(trace)*dt
    if len(trace) < 32 or duration < minimum_duration_s:
        raise ValueError('Insufficient post-source observation duration; no resonance is reported.')
    # Sum monitor powers, avoiding destructive cancellation between spatial probes.
    nfft = next_fast_len(16*len(trace))
    trace = trace-trace.mean(axis=0)
    power = np.sum(np.abs(rfft(trace*np.hanning(len(trace))[:, None], n=nfft, axis=0))**2, axis=1)
    frequency = rfftfreq(nfft, dt)
    low, high = C0/((target_um+half_window_um)*1e-6), C0/((target_um-half_window_um)*1e-6)
    if high >= .5/dt:
        raise ValueError('Optical band exceeds Nyquist frequency.')
    selected = (frequency >= low) & (frequency <= high)
    f, power = frequency[selected], power[selected]
    if len(f) < 5 or not np.isfinite(power).all() or power.max() <= 0:
        raise ValueError('No usable optical signal in target band.')
    normalized = power/power.max()
    peaks, _ = find_peaks(normalized, prominence=.05)
    # This rejects resolved competing peaks, not unresolved multimode mixtures.
    if len(peaks) != 1:
        raise ValueError(f'Expected one isolated candidate in tracking window; found {len(peaks)}.')
    k = int(peaks[0])
    curvature = normalized[k-1]-2*normalized[k]+normalized[k+1]
    offset = .5*(normalized[k-1]-normalized[k+1])/curvature
    f_peak = f[k]+offset*(f[1]-f[0])
    return dict(wavelength_um=C0/f_peak*1e6, duration_s=duration,
                nominal_frequency_resolution_hz=1/duration,
                interpolated_fft_spacing_hz=float(f[1]-f[0]),
                wavelength_um_axis=C0/f[::-1]*1e6, normalized_power=normalized[::-1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_directory', type=Path)
    p.add_argument('--target-um', type=float, required=True)
    p.add_argument('--half-window-um', type=float, default=.0005)
    p.add_argument('--start-ps', type=float, default=.2)
    p.add_argument('--second-start-ps', type=float, default=.4)
    p.add_argument('--minimum-duration-ps', type=float, default=10.)
    args = p.parse_args()
    metadata = json.loads((args.run_directory/'run.json').read_text())
    if metadata['status'] != 'traces_complete_not_resonance_validated':
        raise ValueError('The voltage trace run has not completed.')
    traces = []
    for index, voltage in enumerate(metadata['parameters']['voltages']):
        with np.load(args.run_directory/f'trace_{index:03d}.npz') as data:
            if str(data['fingerprint']) != metadata['fingerprint'] or float(data['voltage_v']) != voltage:
                raise ValueError('Trace provenance does not match run metadata.')
            windows = [peak_from_trace(data['signals'], float(data['time_step_s']),
                        start_s=start*1e-12, target_um=args.target_um,
                        half_window_um=args.half_window_um,
                        minimum_duration_s=args.minimum_duration_ps*1e-12)
                       for start in (args.start_ps, args.second_start_ps)]
            traces.append((voltage, windows))
    refs = [item for item in traces if item[0] == 0.]
    if len(refs) != 1:
        raise ValueError('Require exactly one 0 V reference.')
    reference = refs[0][1]
    rows = []
    for voltage, windows in traces:
        shift = [(w['wavelength_um']-r['wavelength_um'])*1e6 for w, r in zip(windows, reference)]
        rows.append([voltage, windows[0]['wavelength_um']*1000, shift[0], shift[1], abs(shift[1]-shift[0])])
    with (args.run_directory/'candidate_resonances.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['voltage_v', 'candidate_wavelength_nm', 'candidate_shift_pm',
                         'second_window_shift_pm', 'window_shift_difference_pm'])
        writer.writerows(rows)
    analysis = dict(status='candidate_peaks_only', mode_identity_verified=False,
                    mesh_convergence_verified=False, time_convergence_verified=False,
                    source='actual saved FDTD traces',
                    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                    nominal_frequency_resolution_hz=traces[0][1][0]['nominal_frequency_resolution_hz'],
                    warning='Two time windows are a diagnostic, not proof of convergence. Unresolved modes can bias peaks.')
    (args.run_directory/'candidate_analysis.json').write_text(json.dumps(analysis, indent=2))
    print(json.dumps(analysis, indent=2))
    print('Saved candidate_resonances.csv; physical validation remains required.')


if __name__ == '__main__':
    main()
