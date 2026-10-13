# TiO2 Y-pass: 별도 GPU PC용 200-step 패키지

현재 X-pass 실행과 독립된 **Y 투과 / X 억제** 최적화입니다. 원래 X 런을 수정하거나 중단하지 않습니다.

## 바로 실행: Windows

Python 3.12, Git, NVIDIA 드라이버가 있는 PC에서 PowerShell을 열어 실행합니다.

```powershell
git clone --filter=blob:none --sparse --branch codex/ypol-gpu-20261013 --single-branch --depth 1 https://github.com/Suhee-Shin/deep-learning-cr.git
cd deep-learning-cr
git sparse-checkout set ypol_gpu_20261013
cd ypol_gpu_20261013
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\python.exe run_y.py --background
```

실시간 화면: **`runs/y_pass/index.html`**. 파일을 브라우저에서 직접 열면 됩니다. 실행 전 화면은 `preview/index.html`에 있습니다.
시작 오류는 `runs/y_pass/supervisor.log`, 계산 로그는 `worker.log`를 확인합니다.

## Linux

```bash
git clone --filter=blob:none --sparse --branch codex/ypol-gpu-20261013 --single-branch --depth 1 https://github.com/Suhee-Shin/deep-learning-cr.git
cd deep-learning-cr
git sparse-checkout set ypol_gpu_20261013
cd ypol_gpu_20261013
bash setup.sh
.venv/bin/python run_y.py --background
```

설치 스크립트는 별도 가상환경에 CUDA 12.8용 PyTorch 2.9.0을 설치합니다.
[공식 PyTorch 설치 명령](https://pytorch.org/get-started/previous-versions/#v290)을 사용합니다.
원래 PC는 `2.9.0.dev20250907+cu128`을 사용했으므로, 다른 PC의 수치 결과가 비트 단위로 같다고 보장하지 않습니다.
설치 후 CPU 소형 RCWA 회전 대칭 검사와 재개/목적함수 테스트가 자동 실행됩니다. 이는 전체 4 µm 구조의 고차 검증을 대체하지 않습니다.

## 조건과 시작 구조

| 항목 | 설정 |
| --- | --- |
| 시작점 | X 런의 step 125 평가 후 저장된 126회 업데이트 체크포인트 |
| 회전 | `[layer, x, y]` 배열에서 XY 평면 90도; z 순서는 유지 |
| 이어지는 상태 | 연속 밀도 theta, Adam m/v와 누적 126회 업데이트 수 모두 보존·회전 |
| 추가 업데이트 | 200회, 로컬 평가 step 0–200; 종료 step에서는 추가 업데이트 없음 |
| 광학 모델 | 정상 입사, XY 주기 RCWA; TiO2/air 위상 구조 + 반무한 SiO2 |
| 설계 영역 | 4 × 4 µm, 높이 1 µm, 400 × 400 × 100, voxel 10 nm |
| 파장 | 400/500/600/700 nm 학습; 450/550/650 nm는 5 step마다 별도 평가 |
| 계산 | Fourier order 10, complex128, CPU eig 6 worker + CUDA, eigenpair cache |
| 필터 | sigma_xy = sigma_z = 75 nm; XY 주기, z domain_normalized |
| 이진화 | filtered density >= 0.5; 매 업데이트 Binary EM forward, projection/filter STE backward |
| beta / LR | 4.5 고정 / 0.002 고정 |
| gray penalty | 기존 진행 시점의 0.02 유지; ramp를 처음부터 다시 시작하지 않음 |
| 검출면 | 구조 바닥에서 SiO2 내부로 0.35 µm; 네 영역 합계 투과 |
| 출력 | 수치 매 step, 그래프 5 step, 3D 구조 10 step, 별도 Binary 후보/최고 후보 |

**실제 컬러라우터는 포함되어 있지 않습니다.** 이 모델에서 `detector_gap_um=0.35`는 반무한 SiO2 내부 검출면까지 거리입니다. 유한 두께 SiO2 spacer와 실제 하부 컬러라우터를 쌓은 모델로 해석하면 안 됩니다.

원래 X 계열 목적함수의 pass/block 역할만 바꿉니다.

```text
delta(lambda) = TY(lambda) - TX(lambda)
F = -0.03 * logsumexp(-delta / 0.03)
    - 2 * max_lambda[relu(0.8 - TY)^2]
    - 2 * max_lambda[relu(TX - 0.1)^2]
    - 0.02 * mean[4 * rho_projected * (1 - rho_projected)]
```

`settings.json`의 원래 인자 이름 `target_tx`, `target_ty`는 호환성을 위해 각각 **pass 하한 0.8 / block 상한 0.1**으로 유지됩니다. `--pass-pol y`에서 실제 적용 대상은 TY / TX입니다. `config.json`의 `pass_pol`과 함께 해석해야 합니다. 저장된 `tx`, `ty` 필드는 항상 실제 X/Y 입사 편광의 투과율이며 이름을 뒤바꾸지 않습니다.

현재 패키지의 seed는 아직 Y 방향으로 전체 재평가되지 않았습니다. 90도 회전 대칭은 작은 모델에서 확인하지만, 전체 구조의 step 0을 새 PC에서 실제 계산한 뒤 성능을 판단합니다. 기존 X 성능을 Y 측정값으로 복사하지 않습니다.

## 추적, 일시중단, 재개

- 화면은 프로세스 확인 시각과 마지막 **평가 완료** 시각을 구분합니다. 한 step이 계산 중이면 후자는 이전 시각이 정상입니다.
- worker는 매 업데이트 종료 후 자동 재시작하여 메모리를 회수합니다. Adam/theta를 복원하며 초기화하지 않습니다.
- NaN, gradient 이상, 부족한 저장 공간, 비정상 종료는 오류로 표시하고 무한 재시도하지 않습니다.
- 사용자 일시중단: `runs/y_pass/PAUSE_REQUESTED`라는 빈 파일을 만들면 현재 체크포인트 저장 뒤 멈춥니다.
- 명시적 재개: Windows에서는 `.\.venv\Scripts\python.exe run_y.py --resume --background`.
- 300회로 연장: `run_y.py --resume --steps 300 --background`. 기존 실행이 끝나거나 중단된 후 사용합니다.
- 다른 출력 위치: `run_y.py --out D:\ypol_results --background`.

다른 컴퓨터에서 화면을 보려면 GPU PC에서 다음 서버를 실행합니다.

```text
python serve.py --port 8767
```

SSH를 사용할 수 있는 경우 보는 PC에서 `ssh -L 8767:127.0.0.1:8767 USER@GPU_PC`를 연결하고 `http://127.0.0.1:8767/index.html`을 엽니다. 이 서버는 안전을 위해 localhost만 바인딩하며 인터넷 공개나 방화벽 변경을 자동으로 하지 않습니다. GitHub에 올린 HTML 자체는 GPU PC 결과를 자동 수집하지 않습니다.

## 자원 및 검증 범위

원래 PC는 16 GB GPU를 거의 모두 사용했습니다. 최소 14 GiB의 **빈** VRAM을 확인한 뒤 시작하며, 16 GB 이상 GPU를 단독 사용하는 구성을 권장합니다. 32 GB 이상 시스템 RAM, 여유 디스크 25 GB 이상을 준비하세요. GPU 종류와 드라이버에 따라 실행 가능 여부와 속도는 달라집니다. 자원이 부족해도 격자·파장·차수를 몰래 낮추지 않습니다.

현재 모델의 연결성 검사는 진단값이며 모든 공정 조건을 강제하는 최적화는 아닙니다. Binary라고 해서 제작 가능한 구조로 인증된 것은 아닙니다. 최소 선폭·neck·공기 연결성, 7-point 사이 파장, 높은 Fourier order와 독립 FDTD 검증은 후속 작업입니다. 목표 달성은 보장하지 않으며, 추가 200회 후 결과와 최고 Binary 후보를 보존합니다.

## 파일

- `seed/`: SHA256 검증되는 lossless float64 회전 밀도/Adam 데이터. 한 파일이 GitHub 제한을 넘지 않도록 5층씩 저장.
- `src/`: 현재 실행 코드의 고정 복사본과 Y-pass 수정. 원래 PC의 프로젝트 경로 불필요.
- `third_party/`: 고정된 upstream 코드와 MIT LICENSE 및 TiO2 재료 데이터.
- `tests/`: 목적함수 역할·gradient 방향/크기, 작은 RCWA 회전 대칭, checkpoint 재개, 보고서 테스트.
- `runs/y_pass/`: 실행 후 실제 결과, HTML, PNG, CSV, `REPORT_KO.md`, 체크포인트.

출처: [hyoseokp/information-optimal-color-router](https://github.com/hyoseokp/information-optimal-color-router). 원본 커밋과 내보내기 전 소스 해시는 `seed/manifest.json`에 기록합니다. 이 패키지는 color-router objective가 아닌 X/Y 편광 선택 투과 objective를 사용합니다.
