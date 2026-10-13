# 실행 패키지 검증 기록

2026-10-13, 기존 X-pass GPU 런을 중단하지 않고 CPU에서 점검했습니다.

| 검사 | 결과 |
| --- | --- |
| Y 목적함수와 X/Y 교환 목적함수의 동일성 | 통과 |
| synthetic transmission objective의 gradient 부호·크기 | 3개 perturbation 크기에서 통과 |
| 실제 X/Y 저장 필드와 Y-pass ER/차분 계산 | 통과 |
| 작은 RCWA 모델의 90도 회전 X/Y 교환 | 500/650 nm, order 2에서 통과 |
| 3회 Adam 업데이트: 연속 실행과 step별 재개 | theta/m/v 완전 일치, 누적 Adam count 126→129 |
| 전체 seed theta/m/v 파일 및 복원 배열 SHA256 | 모두 일치, float64 무손실 |
| 소형 RCWA + Binary STE 실제 업데이트 | CPU 2회 업데이트 및 마지막 평가 완료 |
| HTML/PNG/CSV/한글 보고서 생성 | 완료, report health 오류 없음 |
| 브라우저 레이아웃 | 1440 px / 390 px, JS 오류·가로 overflow 없음, 그래프·3D 이미지 로드 확인 |

검증 환경: Python 3.12, PyTorch 2.9.0.dev20250907+cu128. 패키지 설치 기본값은 재설치 가능한 안정판 2.9.0+cu128입니다. 새 PC의 GPU 실행·실행 시간·4 µm 전체 Y-pass 성능은 아직 측정하지 않았습니다. 작은 모델 결과를 본 구조의 효율로 주장하지 않습니다.

소형 RCWA의 테스트 목적은 실행 경로·역할 전환·역전파·저장 동작 확인입니다. 실제 100층 전체 구조의 gradient 정확도, Fourier order 수렴, 공정 가능성 또는 광대역 목표 달성을 증명한 것은 아닙니다.
