# X-pass: step 200에서 추가 70회 이어가기

**현재 X 투과 / Y 억제 런의 완료된 step 200을 그대로 이어받아, 총 270회까지 최적화합니다.**
새 random seed가 아니며, Y-pass 회전 패키지도 아닙니다. 원래 컴퓨터에서는 새 런을 시작하지 않았습니다.

## Windows에서 바로 실행

Python 3.12, Git, NVIDIA 드라이버가 설치된 새 GPU PC의 PowerShell에서:

```powershell
git clone --filter=blob:none --sparse --single-branch --depth 1 --branch codex/xpol-continue70-20261015 https://github.com/Suhee-Shin/deep-learning-cr.git xpol-continue70
cd xpol-continue70
git sparse-checkout set xpol_gpu_continue70_20261015
cd xpol_gpu_continue70_20261015
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\python.exe run_x.py --background
```

실시간 화면: **`runs/x_pass_270/index.html`**

기존 결과만 보기: **`baseline/index.html`**. 이 페이지는 step 200의 정적 보존본이며 실시간 페이지가 아닙니다.
시작 및 오류 로그는 `runs/x_pass_270/supervisor.log`, 계산 세부 로그는 `worker.log`입니다.

## Linux

위의 Git 명령으로 같은 폴더를 받은 뒤:

```bash
bash setup.sh
.venv/bin/python run_x.py --background
```

가상환경에는 CUDA 12.8용 PyTorch 2.9.0을 설치합니다.
[공식 설치 명령](https://pytorch.org/get-started/previous-versions/#v290)을 사용합니다.
원래 환경은 `2.9.0.dev20250907+cu128`이므로 새로운 PyTorch/드라이버/GPU의 연산 결과가 비트 단위로 동일하다고 보장하지 않습니다.
반면 **전달되는 체크포인트 파일 및 theta/m/v 값은 무손실로 동일**하며 SHA256으로 검증합니다.
시작 구조(step 200)를 새 GPU에서 재평가한 후 업데이트하므로 첫 평가값을 기존 `baseline`과 비교할 수 있습니다.

## 무엇을 이어받는가

| 항목 | 설정 |
| --- | --- |
| 시작 상태 | 평가 200, 업데이트 200, Adam 누적 200 |
| 최종 상태 | 업데이트 270, step 270에서 업데이트 없이 최종 평가 |
| 추가 업데이트 | 정확히 70회, 학습 변수/Adam 재초기화 없음 |
| 최고 후보 | 기존 best step 180 및 Binary mask 보존, 더 좋은 후보가 나오면 별도로 갱신 |
| 구조 | 4 × 4 µm XY 주기 셀, 높이 1 µm, 400 × 400 × 100 voxel |
| 필터 | sigma_xy = sigma_z = 75 nm, XY periodic, z domain_normalized |
| 이진화 | filtered density >= 0.5, Binary forward + projection/filter STE backward |
| beta / LR | 4.5 / 0.002 고정 |
| 학습 파장 | 400, 500, 600, 700 nm |
| 별도 검증 | 450, 550, 650 nm, 매 5 step |
| 계산 설정 | Fourier order 10, complex128, CUDA + CPU eig 6 workers, eigenpair cache |
| 결과 갱신 | 수치 매 step, 3개 지표 그래프 매 5 step, 3D 그림 매 10 step |

첫 계산 로그는 `evaluated_next=200 adam_updates=200`이 정상입니다. 기존 final step 200은 업데이트 없이 평가된 상태입니다.
새 실행에서 같은 구조를 다시 평가하여 201번째 업데이트를 수행합니다. 그다음 평가 번호는 201입니다.
`history.json`은 전역 step 0부터 이어지며 step 200을 중복 추가하지 않습니다.
전송 전 원래 step 200 기록은 `baseline/`과 불변 `seed/`에 남습니다.

## 목적함수와 검증 범위

```text
delta(lambda) = TX(lambda) - TY(lambda)
F = -0.03 * logsumexp(-delta / 0.03)
    - 2 * max_lambda[relu(0.8 - TX)^2]
    - 2 * max_lambda[relu(TY - 0.1)^2]
    - 0.02 * mean[4 * rho_projected * (1 - rho_projected)]
```

최적화 조건은 변경하지 않았습니다. 정상 입사, XY 주기 RCWA, TiO2/air와 반무한 SiO2 기판입니다.
`detector_gap_um=0.35`는 SiO2 내부 검출면 깊이이며, 유한 spacer나 실제 컬러라우터 적층을 뜻하지 않습니다.
현재 step 200의 학습 파장 Binary TX는 약 77.6–79.8%, TY는 29.3–29.7%입니다.
별도 450/550/650 nm에서는 TX 약 60.6–66.7%, TY 약 55.8–59.3%여서 **광대역 목표 미달**입니다.
추가 70회만으로 이 문제가 해결된다는 보장은 없습니다. 학습 파장 개선과 검증 파장 개선을 구분해 봐야 합니다.
현재 수치의 출처는 `baseline/holdout_latest.json`, `baseline/history.json`입니다.

10 nm는 설계 격자이지 제작 가능한 최소 선폭 인증이 아닙니다. 공정 연결성, 전체 대역의 촘촘한 파장 검사,
Fourier order 수렴 및 독립 FDTD 검증은 아직 필요합니다. 이전 제한적 gradient 검증도 새 후보의 전 대역 검증을 대신하지 않습니다.

## 중단 / 재개 / 원격 보기

- 일시중단: `runs/x_pass_270/PAUSE_REQUESTED` 빈 파일을 만들면 현재 업데이트를 저장한 뒤 멈춥니다.
- 재개: `.\.venv\Scripts\python.exe run_x.py --resume --background`
- 복원만 확인: `run_x.py --prepare-only` (광학 계산 없음). 이후에는 `--resume`으로 실행합니다.
- 출력 폴더 변경: `run_x.py --out D:\xpol_results --background` (재개할 때도 같은 `--out` 필요).
- worker는 매 업데이트마다 종료·재시작하여 메모리를 회수합니다. 중간 `stopped_by_user`는 이 재개 방식의 내부 상태입니다.
- 실제 오류, NaN, 저장 공간 부족은 화면에 표시하고 중단합니다. 오류를 숨기는 무한 재시도는 하지 않습니다.
- 컴퓨터 재부팅 후 자동 시작은 설치하지 않습니다. 위의 `--resume` 명령으로 재개합니다.

HTML 파일은 브라우저에서 직접 열어도 됩니다. GPU PC에서 `python serve.py`를 실행하면
`http://127.0.0.1:8768/index.html`로 볼 수 있습니다. 다른 PC에서 보려면 SSH 포워딩 예:
`ssh -L 8768:127.0.0.1:8768 USER@GPU_PC`.
서버는 localhost에만 바인딩하며 방화벽 변경이나 인터넷 공개를 자동으로 하지 않습니다.
**GitHub에 올린 HTML 자체가 GPU PC에서 나온 새 결과를 자동 업로드하지는 않습니다.**

## 자원과 파일 보존

16 GB 이상 NVIDIA GPU를 단독 사용하며, 시작 시 빈 VRAM 14 GiB 이상, 빈 시스템 RAM 24 GiB 이상,
빈 디스크 25 GiB 이상을 검사합니다. 실제 소요 시간은 새 GPU뿐 아니라 CPU eig 성능에도 의존합니다.
설계 격자·파장·차수를 자동으로 낮추지 않습니다.

- `seed/`: 현재 checkpoint, 최고 구조, 기록과 그림을 압축한 17개 분할 파일 (약 549 MB).
- `seed/manifest.json`: 원본 및 분할 파일 SHA256, 원본 코드 해시, upstream 커밋.
- `src/`, `third_party/`: 기존 광학·최적화 소스의 무수정 복사본. `worker_entry.py`는 보고서 함수만 교체합니다.
- `baseline/`: step 200까지의 이력·표·그림 보존본. 대형 임시 FDTD 파일이나 unrelated 연구 파일은 포함하지 않습니다.
- `runs/x_pass_270/`: 새 컴퓨터의 checkpoint, 최고 Binary, HTML, PNG, CSV, `REPORT_KO.md`.

출처: [hyoseokp/information-optimal-color-router](https://github.com/hyoseokp/information-optimal-color-router).
upstream MIT LICENSE를 함께 포함합니다. 이 패키지의 objective는 컬러 라우팅이 아닌 X/Y 선택 투과입니다.
