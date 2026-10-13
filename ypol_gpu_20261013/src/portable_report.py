"""Portable, file://-compatible reports with physical X/Y labels."""
import csv
import json
import os
from pathlib import Path
import time

import numpy as np


def read(path, fallback=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return fallback


def atomic(path, text):
    path = Path(path)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def json_save(path, value):
    atomic(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def publish(out, supervisor=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    history = read(out/'history.json', [])
    last = history[-1] if history else {}
    payload = dict(observed_at=time.time(), config=read(out/'config.json', {}),
        supervisor=supervisor or read(out/'supervisor.json', {}),
        iteration=read(out/'iteration_status.json', {}), run=read(out/'run_status.json', {}),
        latest=last, best=read(out/'best.json', {}), holdout=read(out/'holdout_latest.json', {}),
        seed=read(out/'seed_provenance.json', {}), history_count=len(history),
        evaluated_at=(out/'history.json').stat().st_mtime if history else None,
        plots_at=(out/'performance.png').stat().st_mtime if (out/'performance.png').exists() else None,
        structures=[p.name for p in sorted((out/'snapshots').glob('step_*_structure_3d.png'))][-1:])
    if not payload['structures']:
        payload['structures'] = [p.name for p in sorted((out/'snapshots').glob('step_*_3d.png'))][-1:]
    atomic(out/'status.js', 'window.receiveStatus('+json.dumps(payload, ensure_ascii=False,
           allow_nan=False).replace('<', '\\u003c')+');\n')
    template = Path(__file__).with_name('portable_dashboard.html').read_text(encoding='utf-8')
    if not (out/'index.html').exists() or (out/'index.html').read_text(encoding='utf-8') != template:
        atomic(out/'index.html', template)
    return payload


def write_live_status(out, records, best, config, finished=False):
    publish(out)


def save_history_plots(out, records, best, target_tx=.8, target_ty=.1):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    config = read(out/'config.json', {})
    pass_pol = config.get('pass_pol', 'y')
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), layout='constrained', sharex=True)
    for key, style, name in [('spectrum', '--', 'Gray'), ('bin_spectrum', '-', 'Binary')]:
        rows = [r for r in records if r.get(key)]
        if not rows:
            continue
        steps = [r['step'] for r in rows]
        tx = np.array([[s['tx'] for s in r[key]] for r in rows])
        ty = np.array([[s['ty'] for s in r[key]] for r in rows])
        pa, bl = (ty, tx) if pass_pol == 'y' else (tx, ty)
        axes[0].plot(steps, 100*ty.mean(1), style, color='#007f73', label=name+' mean TY')
        axes[0].plot(steps, 100*tx.mean(1), style, color='#c14c41', label=name+' mean TX')
        axes[1].plot(steps, (10*np.log10(np.maximum(pa, 1e-15)/np.maximum(bl, 1e-15))).mean(1),
                     style, color='#285ba6', label=name+' mean ER')
        axes[2].plot(steps, (pa-bl).mean(1), style, color='#285ba6', label=name+' mean difference')
        if key == 'bin_spectrum':
            axes[0].plot(steps, 100*pa.min(1), ':', color='#007f73', label='Binary worst pass')
            axes[0].plot(steps, 100*bl.max(1), ':', color='#c14c41', label='Binary worst block')
            axes[2].plot(steps, (pa-bl).min(1), ':', color='#bb8600', label='Binary worst difference')
    for value in (100*target_tx, 100*target_ty):
        axes[0].axhline(value, color='#999999', lw=.8, alpha=.7)
    axes[0].set_ylabel('Transmittance (%)')
    axes[0].set_ylim(-2, 102)
    axes[1].set_ylabel('Extinction ratio (dB)')
    axes[2].set_ylabel('Transmission difference, '+('TY - TX' if pass_pol == 'y' else 'TX - TY'))
    axes[2].set_xlabel('Additional optimization step')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle(f'TiO2 {pass_pol.upper()}-pass | Gray and binary | training wavelengths only', fontsize=13)
    temp = out/'performance.tmp.png'
    fig.savefig(temp, dpi=150)
    temp.replace(out/'performance.png')
    plt.close(fig)


def refresh_live_plots(out, records, best, config):
    # The runner calls save_history_plots immediately before this hook.
    publish(out)


def write_summary(out, wavelengths, records, best, args):
    out = Path(out)
    with (out/'history.csv').open('w', newline='', encoding='utf-8') as handle:
        fields = ['step', 'beta', 'mean_tx', 'mean_ty', 'mean_delta', 'min_delta',
                  'bin_mean_tx', 'bin_mean_ty', 'bin_mean_delta', 'bin_min_delta',
                  'gradient_norm', 'elapsed_s']
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    with (out/'spectra.csv').open('w', newline='', encoding='utf-8') as handle:
        fields = ['step', 'type', 'wavelength_nm', 'tx', 'ty', 'delta', 'contrast', 'er_db']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            for name, key in [('gray', 'spectrum'), ('binary', 'bin_spectrum')]:
                for spectrum in row.get(key, []) or []:
                    writer.writerow(dict(step=row['step'], type=name, **spectrum))
    latest = records[-1] if records else {}
    lines = ['# TiO2 Y-pass 결과 보고서', '',
        f"- 추가 평가 step: {latest.get('step', '미실행')} / {args.max_updates}",
        '- TX와 TY는 실제 X/Y 입사 편광의 투과율입니다. 채널 이름을 바꾼 결과가 아닙니다.',
        '- Y-pass 목표: 모든 평가 파장에서 TY > 0.8, TX < 0.1.',
        '- ER = 10 log10(TY/TX), 차분 = TY - TX.',
        '- 400/500/600/700 nm 최적화, 450/550/650 nm 별도 검증.',
        '- 정상 입사, XY 주기 RCWA, TiO2/air + SiO2 반무한 기판. 실제 컬러라우터 없음.',
        '- 10 nm 설계 격자는 10 nm 제작 가능성을 의미하지 않습니다.',
        '- 현재 Binary 후보의 제작 가능성 및 촘촘한 파장/차수 수렴은 아직 인증되지 않았습니다.',
        f"- 최고 Binary 후보 step: {best.get('step', '없음')}", '',
        '## 최근 Binary 평가', '', '| nm | TX (%) | TY (%) | TY-TX | ER (dB) |',
        '| --- | --- | --- | --- | --- |']
    for row in latest.get('bin_spectrum', []) or []:
        lines.append(f"| {row['wavelength_nm']:g} | {100*row['tx']:.3f} | {100*row['ty']:.3f} | {row['delta']:.5f} | {row['er_db']:.3f} |")
    atomic(out/'REPORT_KO.md', '\n'.join(lines)+'\n')
