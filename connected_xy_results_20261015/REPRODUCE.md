# 재계산 방법

이 패키지는 **고정된 기판 연결 구조의 광학 재평가**용입니다. 최적화나 서버 실행을 시작하지 않습니다.

## 결과만 보기

GitHub README에서 그림·표를 보거나 폴더 전체를 내려받아 `index.html`을 여세요. 외부 CDN이나 API가 필요하지 않습니다. `fields/*.npz`에는 복소 Ex/Ey/Ez가 모두 들어 있습니다.

## 동일 계산 엔진 준비

Python 3.12, Git, NVIDIA GPU/CUDA를 사용할 수 있는 환경에서 진행합니다. PyTorch는 기록된 `2.9.0+cu128` 환경을 기준으로 합니다. 다른 GPU·드라이버에서는 비트 단위 동일성을 보증하지 않습니다. 실제 버전은 `environment.json`을 확인하세요.

```powershell
python scripts/fetch_engines.py
```

위 스크립트는 별도 `engine_cache/X`, `engine_cache/Y`에 소스를 받고, 고정 commit으로 이동하여 전자기 계산에 사용한 소스·재료 파일 SHA256을 검사합니다. 기존 폴더가 있으면 덮어쓰지 않습니다. upstream 라이선스는 각 엔진의 `third_party/information-optimal-color-router-latest/LICENSE`에 있습니다.

별도 가상환경에 PyTorch CUDA 빌드 및 `requirements.txt`의 패키지를 설치하세요. 기존 공용 TorchFDTD 설치 폴더나 다른 사람의 환경을 수정하지 마세요.

## 결과를 별도 폴더에 재계산

먼저 GPU 메모리와 사용률을 확인하세요. 한 번에 하나의 평가 프로세스만 실행합니다.

```powershell
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

원본 보존을 위해 이 결과 폴더를 별도 위치에 복사한 후 `--package`에 복사본을 지정하세요. 아래 예시의 `.`은 복사본 폴더입니다.

```powershell
python scripts/evaluate.py --package . --engine-x engine_cache/X/xpol_gpu_continue70_20261015 --engine-y engine_cache/Y/ypol_gpu_20261013
python scripts/report.py
```

특정 케이스만 계산하려면 `--case X --wavelength 400`을 추가합니다. 허용 파장은 400–700 nm, 50 nm 간격입니다. 기존 결과를 덮어쓰므로 반드시 복사본에서 실행하세요. 재계산 결과는 원본 `manifest.json`과 해시가 달라질 수 있으며, 이는 새 산출물이기 때문입니다.

## 저장된 결과 검증

```powershell
python scripts/verify.py
```

원본 패키지의 파일 해시와 구조 연결성, 공극 외부 연결, 필드 배열, 투과율과 지표 수식을 검사합니다. 400/500/600/700 nm는 기존 동일 연결 구조의 44×44 모니터 평가와 새 128×128 모니터의 총 투과율이 일치하는지도 확인했습니다.

## 해석 시 주의

- 소스는 진폭 1의 독립 X/Y 단색 평면파입니다. 실제 실험 장비나 광원의 절대 파워를 모델링한 값이 아닙니다.
- `TX`, `TY`는 **입사 편광별 전체 투과율**입니다. 출력 편광 분석기 기준의 co/cross 편광 전력과 구별하세요.
- 필드 모니터는 SiO₂ 안으로 출구 이후 1.35 µm에 있습니다. 설정의 `detector_gap_um=0.35`만 보고 0.35 µm로 해석하면 안 됩니다.
- `|E|²` 평균·최댓값·공간 변동계수는 전기장 분포 지표입니다. 전력밀도 W/m²가 아닙니다. 자기장·반사율·흡수율은 이 결과에서 계산하지 않았습니다.
- 각 파장의 결과가 중간 파장 성능이나 RCWA 차수 수렴을 보장하지 않습니다. 특히 450/550/650 nm에서 선택성이 약해질 수 있으므로 전체 표를 확인하세요.
- topology 검사는 원본 10 nm voxel의 면 접촉을 사용합니다. 그림의 3D 표면은 표시를 위해 30 nm로 추출한 것이며 실제 최소 선폭을 보증하지 않습니다.
