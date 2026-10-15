"""Report for the new gray-only, sparsely binary-evaluated run."""
import json
from voxel_dashboard import publish_dashboard


def write_gray_report(out, records, best, c, finished=False):
    row = records[-1]
    binary = next((r for r in reversed(records) if r.get("bin_spectrum")), {})
    graph = json.loads((out/"graph_refresh.json").read_text()) if (out/"graph_refresh.json").exists() else {}
    mode = row.get("objective_mode", c.get("objective_mode", "softmin"))
    formula = ("- F=6 mean(TX-TY)+1.2 mean(TX)-2 mean(TY)+5 min(TX-TY)"
               "-40 mean(relu(0.8-TX)^2)-20 mean(relu(TY-0.1)^2)." if mode == "legacy" else
               "- F=-0.03*logsumexp(-(TX-TY)/0.03)-2*max(relu(0.8-TX)^2)-2*max(relu(TY-0.1)^2).")
    branch = c.get("objective_branch")
    policy = c.get("binary_continuation_rules")
    beta_lines = ([
        "- Binary 중심 beta continuation: Gray 성능 개선만으로 beta 증가를 막지 않음.",
        f"- 같은 beta/목적함수에서 최소 {policy['min_residence']} step과 Binary {policy['evaluations']}개 평가를 확보. 앞/뒤 절반의 평균·최악 TX-TY 및 TX/TY 변화를 비교.",
        f"- Binary 개선이 없고 회색 밀도와 Gray-Binary 차분 간극이 남으면 beta 최대 {policy['growth']:g}배 증가. 개선 중이면 유지. 상한 {policy['max_beta']:g}.",
        "- 증가 시 연속 theta와 Adam을 보존. 같은 theta의 Binary는 beta와 무관하므로 증가 step의 Binary를 비교 기준으로 기록.",
        f"- 증가 후 최소 {policy['min_residence']} step 및 {policy['evaluations']}개 Binary 평가로 재검사. 최근 두 평가 평균에서 TX 감소·TY 증가·차분 감소가 {100*policy['collapse_tolerance']:g}%p를 넘으면 theta/Adam을 복원하고 beta 증가 폭을 절반으로 축소.",
        f"- 최대 {policy['max_attempts']}회 시도 후에도 악화하면 이전 beta로 복원하여 유지하고 검토 표시. 이전 시도 이력/최고 Binary는 삭제하지 않음.",
        "- 이 기준은 운영 휴리스틱이며 매 step의 개선 보장은 아님. beta_events.json과 binary_continuation_status.json에 근거 기록.",
    ] if policy else [
        "- Beta=2 시작. 같은 beta의 gray FoM 20개를 앞/뒤 10개씩 비교; 평균 변화 절댓값이 max(0.005, 이전 평균 절댓값의 0.5%) 이하이면 포화.",
        "- 5 step마다 검사. 개선 중 또는 유의하게 악화 중이면 유지. 포화 시 1.5배 증가(상한 64); 이후 새 20-step 이력 필요.",
    ])
    lines = ["# TiO2 10 nm voxel 편광 필터 최적화", "",
        f"상태: {'완료' if finished else '진행 중'}. Gray 평가 step {row['step']} / {c['max_updates']}.",
        f"최근 Binary step {binary.get('step','대기')}; 그래프 step {graph.get('evaluated_step','대기')}.", "",
        "## 설정", "",
        (f"- 보존된 parent에서 전체 step {branch['start_step']}부터 계속. theta/beta 유지, Adam/후보 순위 초기화."
         if branch else f"- 새 random seed {c['seed']}. 기존 구조/체크포인트를 초기값으로 사용하지 않음."),
        (f"- 초기 랜덤 블록 {c['init_correlation_nm']:g} nm를 fine grid에 펼침. 초기화 후 각 voxel은 독립 변수."
         if c.get('init_correlation_nm', 0) > 0 else "- 초기화: 각 voxel에 독립적인 랜덤 값."),
        f"- 격자 {c['grid']} x {c['grid']} x {c['layers']}; voxel {c['actual_pixel_nm']:g} x {c['actual_pixel_nm']:g} x {c['layer_height_nm']:g} nm.",
        f"- 높이 {c['layers']*c['layer_height_nm']/1000:g} um; x-y 주기 {c['period_um']:g} um.",
        f"- 파장 {c['wavelengths']} nm; Fourier order {c['orders']}.",
        f"- Gray step 0부터 정규화 Gaussian: sigma_xy={c['sigma_nm']:g} nm, sigma_z={c['sigma_z_nm']:g} nm. XY circular; Z {c.get('z_filter_boundary','replicate')}.",
        "- 연속 학습 변수 theta(=rho)를 유지. 매 forward마다 filter -> projection -> RCWA, 두 변환 모두 역전파.",
        ("- Binary는 5 step마다 같은 필터링 밀도 >= 0.5로 직접 평가. 연결성 보정 없음; 공정 적합성 미인증."
         if c.get('binary_evaluation') == 'filtered_threshold' else
         "- Binary는 5 step마다 projection 밀도 threshold 및 지지/통로 투영 후 EM 평가."),
        "- Blur 결과 또는 Binary 마스크로 theta를 덮어쓰지 않음. 최고 Binary의 마스크/스펙트럼/원본 theta를 별도 저장.",
        "- Z domain_normalized인 경우 설계 영역 경계에서 가중치 재정규화; 외부 공기/SiO2는 고정 광학 재료로 유지.",
        f"- 전자기 계산 정밀도: {c['dtype']}.",
        f"- 현재 목적함수 단계: {mode}.", formula,
        (f"- Step {c['softmin_start_step']}부터 soft-min 전환. Adam moment/포화 이력/최고 후보 순위 분리."
         if c.get('objective_mode') == 'legacy_then_softmin' else "- 목적함수 단계 전환 없음."),
        *beta_lines,
        f"- 현재 beta={row['beta']:g}. Gray penalty 및 Binary STE 학습은 사용하지 않음.",
        "- 그래프/3D 미리보기: 50 step마다. 전체 배열: 5 step마다. 현재 theta/Adam checkpoint: 매 step 원자적 교체.",
        "- 큰 3D 그림은 표시 전용 block 평균이며 실제 solver/NPZ는 10 nm. 레이어 그림은 100층 중 16개 선택 단면.",
        ("- 같은 step의 고유모드는 CPU에 보관하여 backward에서 재사용. 층별 checkpoint 유지."
         if c.get('cache_eigenpairs') else "- 파장별 재계산 및 층별 checkpoint로 메모리 절약."),
        (f"- 학습에서 빠진 {c['holdout_wavelengths']} nm는 Binary로 시작/매 {c['holdout_every']} step/종료 시 별도 평가. holdout_latest.json 참조."
         if c.get('holdout_wavelengths') else "- 별도 holdout 파장 없음."), "",
        f"- Gradient norm 급증 보호 한도: {c.get('gradient_spike_limit',0):g}배 (0이면 비활성). 초과 시 update 전 종료.", "",
        "## 최고 Binary 후보", "", f"Step {best.get('step','대기')} (동일 목적함수 기준)", "",
        "| 파장 nm | TX (%) | TY (%) | ER (dB) | TX-TY | 80/10 목표 |", "|---:|---:|---:|---:|---:|:---:|"]
    for s in best.get("spectrum", []):
        passed = s['tx'] > c['target_tx'] and s['ty'] < c['target_ty']
        lines.append(f"| {s['wavelength_nm']:g} | {100*s['tx']:.2f} | {100*s['ty']:.2f} | {s['er_db']:.3f} | {s['delta']:.4f} | {'PASS' if passed else 'FAIL'} |")
    lines += ["", "## 해석 범위", "",
        "- 10 nm는 설계 격자이며 제작 최소 폭이나 전자기 수렴 해상도가 아님. Blur/지지 투영만으로 공정 적합성을 인증하지 않음.",
        "- TiO2/air, 반무한 SiO2, 주기적 RCWA 총 투과율. 반사율, 유한 소자 FDTD, 실제 컬러라우터 stack은 포함되지 않음.",
        f"- Order {c['orders']} 및 {len(c['wavelengths'].split(','))}개 학습 파장 기준; 고차수 수렴과 연속 대역은 미검증. 80/10은 벌점 기준이며 달성을 보장하지 않음.",
        "- [현황](index.html) / [그래프](step_metrics.png) / [수치 이력](history.csv) / [Beta 기록](beta_events.json)", ""]
    text = "\n".join(lines)
    (out/"LIVE_STATUS_KO.md").write_text(text, encoding="utf-8")
    if finished:
        (out/"RESULTS_KO.md").write_text(text, encoding="utf-8")
    publish_dashboard(out, records, best, c, finished=finished)
