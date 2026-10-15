"""Korean report for the bounded binary-forward experiment, not legacy training."""
import json
from voxel_dashboard import publish_dashboard, write_changed


def same_step_spectrum(row):
    holdout = row.get('holdout') or {}
    if holdout.get('step') != row['step']:
        return None
    training = row.get('bin_spectrum') or []
    extra = holdout.get('holdout_spectrum') or []
    if not training or not extra:
        return None
    wavelengths = [s['wavelength_nm'] for s in training + extra]
    if len(wavelengths) != len(set(wavelengths)):
        raise ValueError('Training and holdout wavelengths overlap')
    return sorted(training + extra, key=lambda s: s['wavelength_nm'])


def plot_final_spectrum(out, first, last, filename='final_binary_7point.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    final = same_step_spectrum(last)
    if final is None:
        return
    baseline = same_step_spectrum(first)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout='constrained')
    training = {s['wavelength_nm'] for s in last['bin_spectrum']}
    for rows, alpha, style, label in ((baseline, .4, '--', f"Step {first['step']}"),
                                      (final, 1., '-', f"Step {last['step']}")):
        if rows is None:
            continue
        wl = [r['wavelength_nm'] for r in rows]
        for key, color in (('tx', '#007f8b'), ('ty', '#ce3758')):
            axes[0].plot(wl, [100*r[key] for r in rows], style, color=color,
                         alpha=alpha, label=f'{label} {key.upper()}')
        for ax, key, color in ((axes[1], 'er_db', '#a86700'),
                                (axes[2], 'delta', '#6051a2')):
            ax.plot(wl, [r[key] for r in rows], style, color=color, alpha=alpha, label=label)
    for ax, keys, scale in ((axes[0], ('tx', 'ty'), 100),
                            (axes[1], ('er_db',), 1), (axes[2], ('delta',), 1)):
        for r in final:
            for key in keys:
                ax.scatter(r['wavelength_nm'], scale*r[key], marker='o' if r['wavelength_nm'] in training else 's',
                           s=30, facecolors='white', edgecolors='#222222', zorder=3)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_xticks([r['wavelength_nm'] for r in final])
        ax.grid(alpha=.18)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Transmittance (%)')
    axes[0].set_ylim(0, 100)
    axes[0].axhline(80, color='#007f8b', linestyle=':', alpha=.5)
    axes[0].axhline(10, color='#ce3758', linestyle=':', alpha=.5)
    axes[1].set_ylabel('Extinction ratio (dB)')
    axes[2].set_ylabel('Transmission difference, TX - TY')
    for ax in axes[1:]:
        ax.axhline(0, color='#555555', linewidth=.7)
    fig.suptitle(f"Actual binary response: step {first['step']} vs {last['step']}\n"
                 'Circles: training; squares: holdout. Lines guide the eye; no dense-band validation.')
    fig.savefig(out/filename, dpi=170)
    plt.close(fig)


def write_report(out, records, best, config, finished=False):
    first, last = records[0], records[-1]
    lines = ["Binary-forward / STE 비교 실험", "",
        f"상태: {'업데이트 완료, 판단 대기' if finished else '진행 중'}; 누적 step {last['step']} / {config['max_updates']}",
        f"기존 런은 중단 보존. parent: {config.get('objective_branch', {}).get('parent_directory', config.get('init_state', ''))}",
        "local step 0은 보존된 parent의 새 기준 평가이며, 새로운 random seed가 아닙니다.",
        f"4x4 um XY 주기; 10 nm voxel, 100층 / 1 um; order {config['orders']}; {config['wavelengths']} nm.",
        f"sigma XY/Z={config['sigma_nm']}/{config['sigma_z_nm']} nm; Z={config['z_filter_boundary']}.",
        "매 step 실제 Binary=(filtered density>=0.5)의 전자기 응답으로 학습. Gray는 별도 관찰값입니다.",
        "Backward는 projection과 정규화된 Gaussian filter를 통과하는 STE 근사입니다. 이산 목적함수의 정확한 미분이나 단조 개선 보장은 아닙니다.",
        f"Beta={last['beta']:g} 고정; Adam 새 초기화; LR={config['lr']:g} 고정.",
        "Fopt = -tau*logsumexp(-(TX-TY)/tau) - 2*max(relu(0.8-TX)^2) - 2*max(relu(TY-0.1)^2) - weight*mean(4*rho_projected*(1-rho_projected)).",
        f"tau={config['softmin_tau']}; gray penalty 현재={last['gray_penalty_weight']:.4f}, step 0-5=0, 이후 25까지 최대 0.02로 선형 증가.",
        "실제 Binary 물리 성능을 우선 판단. 서로 다른 목적함수의 FoM 수치를 이전 런과 직접 비교하지 않습니다.",
        f"그래프는 매 {config.get('plot_every', 5)} step, 구조 그림은 매 {config.get('snapshot_every', 5)} step, 최신 Binary 수치는 매 step 저장합니다.", "",
        "step / Binary 평균 TX / 평균 TY / 평균 차분 / 최악 차분 / Grayness"]
    selected = [r for r in records if r['step'] % 5 == 0 or r is last]
    branch = config.get('objective_branch') or {}
    if branch.get('adam_preserved'):
        lines.insert(3, f"이번 분기는 step {branch['start_step']}부터 theta/Adam을 보존해 재개했습니다. 이전 step 이력은 비교용으로 포함합니다.")
    beta_change = branch.get('changes', {}).get('beta')
    if beta_change:
        lines.insert(4, f"Beta 실험: step {branch['start_step']}부터 {beta_change['previous']:g} -> {beta_change['current']:g}. 이전 이력의 beta와 구분합니다. 시작 Binary 형상은 동일하며 surrogate projection/gray penalty 경로만 변경했습니다.")
    for row in selected:
        lines.append(f"{row['step']} / {100*row['bin_mean_tx']:.3f}% / {100*row['bin_mean_ty']:.3f}% / {row['bin_mean_delta']:.5f} / {row['bin_min_delta']:.5f} / {row['grayness']:.5f}")
    lines += ["", "최신 파장별 Binary: nm / TX% / TY% / TX-TY / strict 80/10"]
    for row in last['bin_spectrum']:
        lines.append(f"{row['wavelength_nm']:g} / {100*row['tx']:.3f} / {100*row['ty']:.3f} / {row['delta']:.5f} / {'PASS' if row['tx']>.8 and row['ty']<.1 else 'FAIL'}")
    comparison = dict(baseline_step=first['step'], latest_step=last['step'], best_step=best.get('step'),
        changes={key: last[key]-first[key] for key in ('bin_mean_tx','bin_mean_ty','bin_mean_delta','bin_min_delta')},
        all_training_targets_met=all(r['tx']>.8 and r['ty']<.1 for r in last['bin_spectrum']),
        fabrication_certified=False, order_convergence_certified=False)
    branch_first = next((r for r in records if r['step'] == branch.get('start_step')), None)
    if branch_first:
        comparison['branch_start_step'] = branch_first['step']
        comparison['branch_changes'] = {key: last[key]-branch_first[key] for key in
            ('bin_mean_tx', 'bin_mean_ty', 'bin_mean_delta', 'bin_min_delta')}
        lines += ['', f"이번 분기 step {branch_first['step']} -> {last['step']}의 학습 4파장 평균 변화:",
                  f"TX: {100*comparison['branch_changes']['bin_mean_tx']:+.3f} percentage points; TY: {100*comparison['branch_changes']['bin_mean_ty']:+.3f} percentage points."]
    if branch.get('adam_preserved'):
        lines = [line.replace('Adam 새 초기화', 'Adam 상태 보존') for line in lines]
        lines = [('step 0은 이전 25회 실험의 초기 기준입니다. 이번 분기의 출발점과 구분합니다.'
                  if line.startswith('local step 0은') else line) for line in lines]
    eligible = [(r, same_step_spectrum(r)) for r in records]
    eligible = [(r, spectrum) for r, spectrum in eligible if spectrum]
    if eligible:
        chosen, checked = max(eligible, key=lambda item: min(s['delta'] for s in item[1]))
        seven_point = dict(step=chosen['step'], criterion='max_min_delta_over_same_candidate_sampled_points',
                          spectrum=checked, min_delta=min(s['delta'] for s in checked),
                          all_targets_met=all(s['tx']>.8 and s['ty']<.1 for s in checked),
                          wavelength_count=len(checked), independent_holdout=False,
                          fabrication_certified=False, order_convergence_certified=False)
        write_changed(out/'best_sampled_band.json', json.dumps(seven_point, indent=2))
        lines += ['', f"추가 파장을 포함한 최악 차분 기준 후보: step {chosen['step']}; min delta={seven_point['min_delta']:.5f}"]
    combined = same_step_spectrum(last)
    if combined:
        train_wl = {r['wavelength_nm'] for r in last['bin_spectrum']}
        lines += ['', f"동일한 step {last['step']}의 학습+미학습 파장 평가 (실제 Binary)",
                  'nm / 구분 / TX% / TY% / 차분 / ER(dB) / strict 80/10']
        for r in combined:
            kind = '학습' if r['wavelength_nm'] in train_wl else '미학습'
            passed = r['tx']>.8 and r['ty']<.1
            lines.append(f"{r['wavelength_nm']:g} / {kind} / {100*r['tx']:.3f} / {100*r['ty']:.3f} / {r['delta']:.5f} / {r['er_db']:.3f} / {'PASS' if passed else 'FAIL'}")
        comparison['all_sampled_targets_met'] = all(r['tx']>.8 and r['ty']<.1 for r in combined)
        comparison['sampled_count'] = len(combined)
        comparison['same_step_holdout'] = last['step']
        lines += [f"위 표는 최종 step {last['step']} 기준이며, best step {best.get('step')}의 추가 파장 성능으로 임의 전용하지 않습니다.",
                  '미학습 파장은 optimizer 업데이트에 사용하지 않았습니다. 7개 점의 결과가 연속 대역 성능을 보장하지 않습니다.']
        if finished:
            plot_final_spectrum(out, first, last)
            if branch_first:
                plot_final_spectrum(out, branch_first, last, filename='branch_binary_7point.png')
    write_changed(out/'trial_comparison.json', json.dumps(comparison, indent=2))
    lines += ["", "공정 적합성, Fourier order 수렴, 유한 면적 FDTD 및 실제 컬러라우터 stack은 미검증입니다.",
        "제한된 파장/후보의 실수 유전율 directional gradient 검사는 별도 audit 결과를 참조합니다. 다른 파장과 STE 자체의 정확성까지 증명한 것은 아닙니다.",
        "최종 목표 성공 여부는 모든 파장의 실제 Binary TX>0.80, TY<0.10으로 판단하며, 평균 개선만으로 성공이라고 하지 않습니다."]
    write_changed(out/'TEST_REPORT_KO.txt', '\n'.join(lines)+'\n')
    publish_dashboard(out, records, best, config, finished)
