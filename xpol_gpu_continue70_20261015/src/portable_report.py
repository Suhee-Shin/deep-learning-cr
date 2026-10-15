"""Local HTML and scientific plots for a global-step X-pass continuation."""
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
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(text, encoding='utf-8')
    temp.replace(path)


def json_save(path, data):
    atomic(path, json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2)+'\n')


def publish(out, supervisor=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    history = read(out/'history.json', [])
    payload = dict(observed_at=time.time(), supervisor=supervisor or read(out/'supervisor.json', {}),
        latest=history[-1] if history else {}, best=read(out/'best.json', {}),
        config=read(out/'config.json', {}), run=read(out/'run_status.json', {}),
        iteration=read(out/'iteration_status.json', {}), holdout=read(out/'holdout_latest.json', {}),
        evaluated_at=(out/'history.json').stat().st_mtime if history else None,
        plots_at=(out/'performance.png').stat().st_mtime if (out/'performance.png').exists() else None,
        structures=[p.name for p in sorted((out/'snapshots').glob('step_*_3d.png'))][-1:])
    atomic(out/'status.js', 'window.receiveStatus('+json.dumps(payload, ensure_ascii=False,
           allow_nan=False).replace('<', '\\u003c')+');\n')
    template = Path(__file__).with_name('portable_dashboard.html').read_text(encoding='utf-8')
    if not (out/'index.html').exists() or (out/'index.html').read_text(encoding='utf-8') != template:
        atomic(out/'index.html', template)
    return payload


def write_live_status(out, records, best, config, finished=False):
    publish(out)


def refresh_live_plots(out, records, best, config):
    publish(out)


def save_history_plots(out, records, best, target_tx=.8, target_ty=.1):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, layout='constrained')
    for key, style, label in [('spectrum', '--', 'Gray'), ('bin_spectrum', '-', 'Binary')]:
        rows = [r for r in records if r.get(key)]
        if not rows:
            continue
        steps = np.array([r['step'] for r in rows])
        tx = np.array([[s['tx'] for s in r[key]] for r in rows])
        ty = np.array([[s['ty'] for s in r[key]] for r in rows])
        axes[0].plot(steps, 100*tx.mean(1), style, color='#087e8b', label=label+' mean TX')
        axes[0].plot(steps, 100*ty.mean(1), style, color='#bb3754', label=label+' mean TY')
        er = 10*np.log10(np.maximum(tx, 1e-15)/np.maximum(ty, 1e-15))
        axes[1].plot(steps, er.mean(1), style, color='#2861a3', label=label+' mean ER')
        axes[2].plot(steps, (tx-ty).mean(1), style, color='#2861a3', label=label+' mean TX-TY')
        if key == 'bin_spectrum':
            axes[0].plot(steps, 100*tx.min(1), ':', color='#087e8b', label='Binary min TX')
            axes[0].plot(steps, 100*ty.max(1), ':', color='#bb3754', label='Binary max TY')
            axes[1].plot(steps, er.min(1), ':', color='#bb8600', label='Binary min ER')
            axes[2].plot(steps, (tx-ty).min(1), ':', color='#bb8600', label='Binary min TX-TY')
    axes[0].axhline(100*target_tx, color='#087e8b', lw=.7)
    axes[0].axhline(100*target_ty, color='#bb3754', lw=.7)
    axes[0].set(ylabel='Transmittance (%)', ylim=(-2, 102))
    axes[1].set_ylabel('Extinction ratio (dB)')
    axes[2].set(ylabel='Transmission difference, TX-TY', xlabel='Global optimization step')
    for ax in axes:
        ax.axvline(200, color='.4', lw=.8, label='GPU transfer at step 200')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle('TiO2 X-pass | Training wavelengths: 400 / 500 / 600 / 700 nm', fontsize=13)
    target = Path(out)/'performance.png'
    temp = target.with_name('performance.tmp.png')
    fig.savefig(temp, dpi=150)
    temp.replace(target)
    plt.close(fig)


def write_summary(out, wavelengths, records, best, args):
    out = Path(out)
    with (out/'history.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['step', 'beta', 'adam_updates', 'update_applied',
            'mean_tx', 'mean_ty', 'mean_delta', 'min_delta', 'bin_mean_tx', 'bin_mean_ty',
            'bin_mean_delta', 'bin_min_delta', 'gradient_norm', 'elapsed_s'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    with (out/'spectra.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['step', 'type', 'wavelength_nm', 'tx', 'ty',
                                                   'delta', 'contrast', 'er_db'], extrasaction='ignore')
        writer.writeheader()
        for row in records:
            for kind, key in [('gray', 'spectrum'), ('binary', 'bin_spectrum')]:
                for spectrum in row.get(key) or []:
                    writer.writerow(dict(step=row['step'], type=kind, **spectrum))
    latest = records[-1] if records else {}
    holdout = read(out/'holdout_latest.json', {})
    lines = ['# TiO2 X-pass 이어달리기 결과', '',
        f"- 최근 평가: global step {latest.get('step', '-')} / {args.max_updates}",
        '- 시작: 완료된 step 200. theta와 Adam m/v, 누적 업데이트 200회 보존. 추가 70회 후 step 270 최종 평가.',
        '- 최초 계산은 저장된 step 200을 새 환경에서 재평가하고 다음 업데이트를 수행합니다. 번호를 0으로 초기화하지 않습니다.',
        '- 목표: 각 평가 파장에서 TX > 80%, TY < 10%. 평균값 달성만으로 전 대역 통과를 선언하지 않습니다.',
        '- 최적화: 400/500/600/700 nm. 별도 검증: 450/550/650 nm, 5 step마다.',
        '- ER = 10 log10(TX/TY), 차분 = TX-TY. TX/TY는 실제 X/Y 입사 정규화 검출 투과입니다.',
        '- beta=4.5, LR=0.002, sigma_xy=sigma_z=75 nm, Fourier order=10 고정.',
        '- 정상 입사 XY 주기 RCWA, TiO2/air와 반무한 SiO2. 실제 컬러라우터는 포함하지 않습니다.',
        '- 10 nm 설계 격자는 10 nm 제작 가능성을 의미하지 않습니다. 공정 및 차수 수렴은 미인증입니다.',
        f"- 최고 학습 Binary 후보: step {best.get('step', '-')}. 최신 후보와 다를 수 있습니다.",
        '', '## 최근 Binary 학습 파장', '', '| nm | TX (%) | TY (%) | TX-TY | ER (dB) |',
        '| --- | --- | --- | --- | --- |']
    def table(spectrum):
        return [f"| {s['wavelength_nm']:g} | {100*s['tx']:.3f} | {100*s['ty']:.3f} | {s['delta']:.5f} | {s['er_db']:.3f} |" for s in spectrum]
    lines += table(latest.get('bin_spectrum') or [])
    lines += ['', f"## 별도 검증 파장 (step {holdout.get('step', '-')})", '',
              '| nm | TX (%) | TY (%) | TX-TY | ER (dB) |', '| --- | --- | --- | --- | --- |']
    lines += table(holdout.get('holdout_spectrum', []))
    lines += ['', '학습 파장과 별도 검증 파장은 구분해서 해석해야 합니다. 저장된 step 200은 중간 파장 선택성이 약하며 광대역 목표를 달성한 상태가 아닙니다.']
    atomic(out/'REPORT_KO.md', '\n'.join(lines)+'\n')
