"""Live plots and reports from evaluated states only."""
import json
from pathlib import Path
import shutil
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from voxel_dashboard import publish_dashboard, write_changed


def atomic_json(path, value):
    write_changed(path, json.dumps(value, ensure_ascii=False, indent=2))


def series(rows, key):
    x, values = [], []
    for row in rows:
        spectrum = row.get(key)
        if not spectrum:
            continue
        tx = np.array([s["tx"] for s in spectrum])
        ty = np.array([s["ty"] for s in spectrum])
        x.append(row["step"])
        values.append([100*tx.mean(), 100*ty.mean(),
                       np.mean(10*np.log10(np.maximum(tx, 1e-15)/np.maximum(ty, 1e-15))),
                       np.mean(tx-ty)])
    return x, np.array(values)


def compatible_plot_records(records, config):
    """Never join averages or spectral indices across different wavelength sets."""
    branch = config.get("objective_branch") or {}
    path = branch.get("parent_history_path")
    if not path:
        return list(records)
    expected = tuple(sorted(float(w) for w in config["wavelengths"].split(",") if w.strip()))
    parent = json.loads(Path(path).read_text(encoding="utf-8"))
    compatible = []
    for row in parent:
        spectra = [row.get(key) for key in ("spectrum", "bin_spectrum") if row.get(key)]
        if row["step"] < branch["start_step"] and spectra and all(
            tuple(sorted(float(s["wavelength_nm"]) for s in spectrum)) == expected
            for spectrum in spectra
        ):
            compatible.append(row)
    return compatible + list(records)


def refresh_live_plots(out, records, best, config):
    out = Path(out)
    records = compatible_plot_records(records, config)
    if not records:
        return
    branch = config.get("objective_branch")
    step = records[-1]["step"]
    first_step = records[0]["step"]
    limits = (first_step-.5, step+.5) if first_step == step else (first_step, step)
    markers = [(branch["start_step"], "Branch start")] if branch else []
    continuation_rows = [r for r in records if r.get("beta_policy") == "binary_aware_gray_continuation_v1"]
    if continuation_rows:
        markers.append((continuation_rows[0]["step"], "Binary-aware beta"))
    if config.get("objective_mode") == "legacy_then_softmin" and step >= config["softmin_start_step"]:
        markers.append((config["softmin_start_step"], "Soft-min start"))
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), layout="constrained")
    for key, style, label in (("spectrum", "--", "Gray"), ("bin_spectrum", "-", "Binary")):
        x, values = series(records, key)
        if not len(values):
            continue
        marker = "o" if len(x) == 1 else None
        for index, color, pol in ((0, "#087E8B", "TX"), (1, "#BF3F55", "TY")):
            axes[0].plot(x, values[:, index], style, color=color, label=f"{pol} {label}", marker=marker)
        axes[1].plot(x, values[:, 2], style, color="#A56A00", label=label, marker=marker)
        axes[2].plot(x, values[:, 3], style, color="#6549A4", label=label, marker=marker)
    axes[0].set(ylabel="Transmittance (%)", title="투과 성능", ylim=(0, 100))
    axes[1].set(ylabel="Extinction ratio (dB)", title="편광 구분 능력")
    axes[2].set(ylabel="Transmission difference, $T_X-T_Y$", title="감도 차분 성능")
    for ax in axes:
        ax.set(xlabel="Step", xlim=limits)
        if config.get("binary_first_step") is not None and step >= config["binary_first_step"]:
            ax.axvline(config["binary_first_step"], color=".4", ls=":", label="Binary-only STE")
        for marker_step, label in markers:
            if first_step <= marker_step <= step:
                ax.axvline(marker_step, color=".25", ls="-.", label=label)
        if first_step == step:
            ax.set_xticks([step])
        ax.grid(alpha=.2)
        ax.legend(loc="best")
    fig.suptitle(f"{config.get('actual_pixel_nm',100):g} nm voxel · 높이 {config.get('layers',10)*config.get('layer_height_nm',100)/1000:g} um · step {first_step}–{step}\n"
                 f"{config['wavelengths'].replace(',', '/')} nm 평균 · 주기적 RCWA order {config['orders']} · 수렴 미검증", fontsize=13)
    fig.savefig(out / "step_metrics.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
    x = [r["step"] for r in records]
    axes[0].step(x, [r["beta"] for r in records], where="post", color="#087E8B", marker="o" if len(x)==1 else None)
    axes[0].set(ylabel="Beta", title="Adaptive beta history")
    axes[1].plot(x, [r["gray_fraction"] for r in records], label="Fraction with 0.1 < density < 0.9", marker="o" if len(x)==1 else None)
    axes[1].plot(x, [r["grayness"] for r in records], label="Mean 4*density*(1-density)", marker="o" if len(x)==1 else None)
    axes[1].axhline(.1, color=".4", ls=":")
    axes[1].set(xlabel="Step", ylabel="Gray material measure", ylim=(0, 1.02))
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_xlim(limits)
        if first_step == step:
            ax.set_xticks([step])
    fig.savefig(out / "beta_history.png", dpi=180)
    plt.close(fig)
    paired = [r for r in records if r.get("bin_mean_tx") is not None]
    if paired:
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
        x = [r["step"] for r in paired]
        axes[0].plot(x, [100*(r["mean_tx"]-r["bin_mean_tx"]) for r in paired], label="TX gray - binary (pp)")
        axes[0].plot(x, [100*(r["mean_ty"]-r["bin_mean_ty"]) for r in paired], label="TY gray - binary (pp)")
        axes[0].axhline(0, color=".5", lw=.8)
        axes[0].set(ylabel="Transmission gap (percentage points)", title="Gap reduction alone is not performance improvement")
        axes[1].plot(x, [r["gray_fraction"] for r in paired], label="Gray voxel fraction")
        axes[1].plot(x, [r["hard_weight"] for r in paired], label="Binary objective weight")
        axes[1].set(xlabel="Step", ylabel="Fraction / weight", ylim=(0, 1.02))
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend()
        fig.savefig(out / "gray_binary_gap.png", dpi=180)
        plt.close(fig)
    if config.get("objective_mode") in ("softmin", "legacy_then_softmin") and any(r.get("bin_spectrum") for r in records):
        paired = [r for r in records if r.get("bin_spectrum")]
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.3), layout="constrained")
        x = [r["step"] for r in paired]
        marker = "o" if len(x) == 1 else None
        axes[0].plot(x, [r["bin_min_delta"] for r in paired], label="Binary worst wavelength", marker=marker)
        axes[0].plot(x, [r["min_delta"] for r in paired], "--", label="Gray worst wavelength", marker=marker)
        axes[0].set(title="Minimum difference (not mean)", ylabel="min(TX - TY)", xlabel="Global step")
        for j, s in enumerate(paired[-1]["bin_spectrum"]):
            axes[1].plot(x, [r["bin_spectrum"][j]["delta"] for r in paired], label=f"{s['wavelength_nm']:g} nm", marker=marker)
        axes[1].set(title="Binary selectivity by wavelength", ylabel="TX - TY", xlabel="Global step")
        for ax in axes[:2]:
            for marker_step, _ in markers:
                if first_step <= marker_step <= step:
                    ax.axvline(marker_step, color=".4", ls=":")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        latest = paired[-1]
        weights = latest.get("binary_objective_details", {}).get("wavelength_weights")
        if weights:
            labels = [f"{s['wavelength_nm']:g}" for s in latest["bin_spectrum"]]
            axes[2].bar(labels, weights, color="#087e8b")
        axes[2].set(title="Soft-min delta term weights", xlabel="Wavelength (nm)", ylabel="Output-space weight", ylim=(0, 1))
        axes[2].grid(axis="y", alpha=.2)
        fig.suptitle(f"Step {step}: actual minimum tracked separately from soft-min score; order {config['orders']}")
        fig.savefig(out / "worst_band_history.png", dpi=180)
        plt.close(fig)
    archive = out / "milestones"
    archive.mkdir(exist_ok=True)
    for name in ("step_metrics.png", "beta_history.png", "gray_binary_gap.png", "worst_band_history.png", "efficiency_history.png", "wavelength_history.png", "best_binary_spectrum.png"):
        if (out / name).exists():
            shutil.copy2(out / name, archive / f"step_{step:04d}_{name}")
    atomic_json(out / "graph_refresh.json", dict(evaluated_step=step, updated_at=datetime.now().isoformat(),
                                               interval=config["plot_every"], next_step=min(config["max_updates"], (step//config["plot_every"]+1)*config["plot_every"])))


def write_live_status(out, records, best, config, finished=False):
    out = Path(out)
    if config.get("binary_forward_ste"):
        from voxel_binary_trial_report import write_report
        return write_report(out, records, best, config, finished)
    if config.get("gray_plateau_beta"):
        from voxel_gray_report import write_gray_report
        return write_gray_report(out, records, best, config, finished)
    row = records[-1]
    graph = json.loads((out / "graph_refresh.json").read_text()) if (out / "graph_refresh.json").exists() else {}
    binary_first = config.get("binary_first_step")
    branch = config.get("objective_branch")
    objective_lines = []
    if config.get("objective_mode") == "softmin":
        objective_lines = ["## 최악 파장 중심 목적함수", "",
            "- Delta = TX-TY. F = -tau*log(sum(exp(-Delta/tau))) - a*max(relu(0.8-TX)^2) - b*max(relu(TY-0.1)^2).",
            f"- tau={config['softmin_tau']:g}, a={config['worst_tx_penalty']:g}, b={config['worst_ty_penalty']:g}. 평균 성능 보상은 제거함.",
            "- TX/TY 항은 최악 파장 기준의 soft penalty이며 하드 제약을 만족했다고 보장하지 않음. 80/10 통과 여부는 각 파장에서 별도로 판정.",
            "- soft-min 가중치는 차분 항의 출력단 민감도이며 전체 형상 gradient의 비율은 아님. 실제 min(TX-TY)도 별도 저장.",
            "- tau와 벌점 계수는 이번 런의 운영값이며 최적값으로 검증된 수치가 아님.",
            "- 전략 참고: https://meep.readthedocs.io/en/latest/Python_Tutorials/Adjoint_Solver/ (다파장 worst-case 접근). 실제 구현은 위 soft-min + worst penalty이며 Meep epigraph solver를 실행하는 것은 아님.", ""]
        if branch:
            objective_lines += [f"- global step {branch['start_step']}부터 새 목적함수. 기존 theta/beta를 이어받고 Adam m/v와 beta 판단 이력은 초기화.",
                "- 이전 FoM과 새 FoM은 다른 함수이므로 직접 비교하지 않음. history.json은 새 목적함수 구간만 포함; 물리 성능 그래프는 부모 이력을 포함하고 변경 시점을 표시함.", ""]
    policy_lines = []
    if binary_first is not None:
        policy_lines = [
            f"- 아래 기존 규칙은 step {binary_first} 이전에만 적용. 이후에는 binary FoM만으로 개선/정체를 판정하며 gray 개선으로 beta 증가를 막지 않음.",
            f"- step {binary_first} 이후 같은 목적함수·beta의 20-step 이력이 필요. 두 10-step 구간 비교하며 목적함수 변경 전 이력은 판정에 사용하지 않음.",
            "- 정체 + 회색 영역 잔존 시 beta를 최대 1.25배로 증가 시험. gray 성능 하락만으로 거절하지 않으며 binary 성능을 보호함.",
            "- 증가 후 실제 업데이트된 hard 구조 10개의 평균을 재검사. binary FoM 5% 또는 0.05 초과 하락, 평균/최악 TX 하락·TY 증가·최악 차분 하락 3%p 초과 시 이전 theta와 Adam m/v/누적 횟수를 복원하고 증가 폭을 절반으로 줄임.",
            "- 복원 이전 시도의 이력도 삭제하지 않으며 총 500-step 예산에 포함. beta_events.json에 복원 시점을 기록함.",
            "- 회색 페널티는 현재 0. 반복 실패하면 gray_penalty_review.json에 검토 필요를 기록할 뿐 자동 활성화하지 않음.",
            f"- 현재 binary 목적함수 비중 {row['hard_weight']:.1%}; beta 증가 후 관찰 중: {bool(row.get('beta_guard_pending'))}. Gray 곡선 하락만으로 간격이 줄어든 것은 성공이 아님.",
            "", "### Step 200 이전 규칙", "",
        ]
    lines = ["# TiO2 광대역 랜덤 topology 런", "",
             f"상태: {'완료' if finished else '진행 중'}. 평가 step {row['step']} / {config['max_updates']}. "
             f"현재 beta={row['beta']:g}. 그래프 마지막 갱신 step {graph.get('evaluated_step', '준비 중')}.", "",
             "## 설정", "",
             f"- random seed {config['seed']}, 100×100×100 nm voxel, 40×40×10, 높이 1 um, x-y 주기 4×4 um.",
             f"- 파장 {config['wavelengths']} nm. 모든 광학 계산은 order {config['orders']}.",
             f"- Adam 학습률 {config['lr']:g} → {config['lr_final']:g} cosine. 목표 TX ≥ 0.80, TY ≤ 0.10.",
             "- 기존 TiO2 Siefke/air, 반무한 SiO2. 전체 투과율 평가이며 실제 컬러라우터 stack이나 RY 계산이 아님.",
             (f"- 원래 랜덤 시작에서 이어온 목적함수 변경 branch (시작 step {branch['start_step']}). 고정 threshold 0.5 이진 forward."
              if branch else "- 원본 후보를 초기값으로 사용하지 않은 새 랜덤 시작. 고정 threshold 0.5 이진 forward를 평가함."),
             "- step 0~80 gray objective, 80~200 hard STE 비중 증가, 200 이후 hard forward objective. beta는 별도 적응 규칙으로 결정.", "",
             *objective_lines,
             "## Beta 규칙", "",
             *policy_lines,
             "- 같은 beta에서 최소 20 step, 두 10-step 구간의 gray/binary FoM 평균을 비교. 한쪽이라도 max(0.01, 기존 평균 절댓값의 0.5%)보다 개선되면 유지.",
             "- 둘 다 정체되고 0.1<density<0.9 영역 비율 및 평균 4*density*(1-density)가 모두 0.1 이상이면 증가 시험. 시작 beta=2, 최초 배율 1.5, 상한 64.",
             "- 같은 theta에서 beta만 바꿔 무업데이트 시험. gray/binary FoM 하락이 max(0.05, 기준 절댓값의 5%) 초과하거나 평균 TX, 평균 TY, 최악 차분이 3%p 악화하면 거절.",
             "- 거절하면 증가 배율의 1 초과분을 절반으로 줄임. 최대 4회 시험, 실패하면 원 beta 유지하고 재최적화. 거절 시험에서는 theta와 Adam m/v를 변경하지 않음.",
             "- 세부 결정과 시험 성능은 beta_events.json에 저장. 임계값은 이번 런의 운영 설정이며 물리적 보장 조건은 아님.", "",
             "## 현재 보존된 최고 이진 후보", "",
             f"선정 step {best.get('step', '?')}. 동일 7-point/order-10 FoM 기준. 고차수 수렴 검증은 이번 요청에 따라 실행하지 않음.", "",
             "| 파장 nm | TX (%) | TY (%) | ER (dB) | TX-TY | 80/10 목표 |",
             "|---:|---:|---:|---:|---:|:---:|"]
    for r in best.get("spectrum", []):
        passed = r["tx"] >= config["target_tx"] and r["ty"] <= config["target_ty"]
        lines.append(f"| {r['wavelength_nm']:g} | {100*r['tx']:.2f} | {100*r['ty']:.2f} | {r['er_db']:.3f} | "
                     f"{r['delta']:.4f} | {'PASS' if passed else 'FAIL'} |")
    lines.extend(["", "## 저장 자료", "",
                  "- [세 가지 성능 지표](step_metrics.png)",
                  "- [파장별 성능](wavelength_history.png)",
                  "- [beta 및 회색 밀도 변화](beta_history.png)",
                  "- [gray/binary 간격 및 binary 목적함수 비중](gray_binary_gap.png)",
                  *( ["- [최소 차분·파장별 차분·soft-min 가중치](worst_band_history.png)"] if config.get("objective_mode") == "softmin" else []),
                  "- [3D 구조 변화](structure_evolution_3d.gif)",
                  "- milestones/: 50 step마다 보존한 그래프; snapshots/: 실제 theta/density/binary 배열.",
                  "- [자동 새로고침 보기](index.html)", "",
                  "order 10의 7개 표본만으로 연속 전 대역의 실물 성능을 확정하지 않음. 100 nm 격자 및 지지/공극 변환은 최소 neck/gap과 전체 제작 적합성을 인증하지 않음.", ""])
    text = "\n".join(lines)
    (out / "LIVE_STATUS_KO.md").write_text(text, encoding="utf-8")
    if finished:
        (out / "RESULTS_KO.md").write_text(text, encoding="utf-8")
    publish_dashboard(out, records, best, config, finished=finished)
