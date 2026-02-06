import json         #{"text": "..."}\n 형태로 JSONL을 쓰기 위해 사용. 여러 개의 독립적인 객체를 줄 바꿈 \n으로 구분하여 한 파일에 나열한 형식. open AI파인튜닝, Hugging face 등에서 사용
import numpy as np  
from pathlib import Path #윈도우 경로를 안전하게 다루기 위한 객체 (문자열보다 실수 적음)

# ===== 경로 설정 (사용자 환경에 맞게 유지) =====
BASE_DIR = Path(r"C:\Deep learning\results_norm")
X_PATH = BASE_DIR / "structures_processed.npy"
Y_PATH = BASE_DIR / "spectrum_processed.npy"

# 저장 경로
OUT_DIR = BASE_DIR / "qwen_dataset_hex_int"  # 폴더명 구분 위해 변경 권장
OUT_DIR.mkdir(parents=True, exist_ok=True) # parents=True : 중간 폴더가 없어도 같이 만들어줌.  exist_ok=True : 이미 있어도 에러 안 남.

TRAIN_JSONL = OUT_DIR / "train.jsonl"  
VAL_JSONL = OUT_DIR / "val.jsonl"

VAL_RATIO = 0.01      #검증 데이터 비율을 1%로 설정
# [수정 포인트 1] 소수점 대신 스케일 팩터 정의
# 0.712 -> 712로 변환하기 위해 1000을 곱합니다.
# 0.6~0.8 구간이 600~800이 되어 변별력이 생깁니다.
SCALE_FACTOR = 1000
#DECIMALS = 3          #스펙트럼 값은 소수점 3자리로 반올림해서 텍스트화

# 태그 / 확실히 구분 가능, 학습 및 추론시 파싱하기 쉬움
INPUT_START  = "### INPUT_GRID_START"
INPUT_END    = "### INPUT_GRID_END"
OUTPUT_START = "### OUTPUT_SPECTRUM_START"
OUTPUT_END   = "### OUTPUT_SPECTRUM_END"

def bits_row_to_hex(row_bits_0_1: np.ndarray) -> str:    
    row_bits_0_1 = row_bits_0_1.astype(np.uint8)      # 비트 패킹은 정수형이 안전. bool/float 섞이면 문제 생길 수 있음
    packed = np.packbits(row_bits_0_1)                # 8개 비트를 1바이트로 압축
    return packed.tobytes().hex()                     # 바이트열을 저장하기 쉬운 16진수 문자열로 변환. 결과적으로 매우 짦은 텍스트로 바뀜(hex)

def grid_to_text_hex(grid_50x50: np.ndarray) -> str:    #50*50을 uint8로 맞춘 뒤
    g = grid_50x50.astype(np.uint8)
    lines = []
    for r in range(50):
        hx = bits_row_to_hex(g[r])                      # 각 행을 변환해서 줄바꿈으로 이어붙임
        lines.append(hx)
    return "\n".join(lines)                             # 줄바꿈으로 이어붙임

def spectrum_to_text(spec_100: np.ndarray) -> str:
    spec = np.asarray(spec_100, dtype=np.float32)       # 입력이 리스트 및 다른 dtype이어도 float32로 통일
    spec = np.clip(spec, 0.0, 1.0)                      # 이미 데이터가 정규화된 것이라면 범위를 강제(학습 안정)
    #spec = np.round(spec, DECIMALS)                     # 소수점 3자리로 반올림
    spec_int = (spec * SCALE_FACTOR).astype(int)
    
    # [수정 1] 콤마 뒤에 공백 추가! (매우 중요)
    # LLM이 숫자 하나하나를 독립된 토큰으로 인식하게 돕습니다.
    return ", ".join(str(v) for v in spec_int.tolist()) # 콤마 뒤에 공백을 추가하면 토큰 분리가 용이

def make_example(grid, spec) -> str:
    inp = grid_to_text_hex(grid)
    out = spectrum_to_text(spec)

    # [수정 2] 잡다한 영어 설명(Instruction) 전부 제거
    # 기계적인 포맷만 남겨서 모델이 수치 패턴에만 집중하도록 유도
    return (
        f"{INPUT_START}\n"
        f"{inp}\n"
        f"{INPUT_END}\n"   # 사이 공백 줄임
        f"{OUTPUT_START}\n"
        f"{out}\n"
        f"{OUTPUT_END}"
    )

def main():
    if not X_PATH.exists() or not Y_PATH.exists():
        print("파일 없음 확인 필요")
        return

    X = np.load(X_PATH) 
    Y = np.load(Y_PATH) 

    # 채널 차원 제거 및 전처리
    if X.ndim == 4: X = X[:, 0, :, :]
    if Y.ndim == 3: Y = Y[:, 0, :]

    N = X.shape[0]
    print(f"Total samples: {N}")    # 샘플 개수 출력

    # 셔플
    rng = np.random.default_rng(42) # 시드 42로 고정. 시드는 난수를 뽑는 출발점 번호. 시드가 같으면 섞이는 순서도 항상 동일. 즉 언제 돌려도 같은 샘플이 val에 들어감. 비교가 정당해짐.
    idx = rng.permutation(N)

    n_val = max(1, int(N * VAL_RATIO))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    def write_jsonl(path, indices):
        with open(path, "w", encoding="utf-8") as f:
            for i in indices:
                text = make_example(X[i], Y[i])
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"Writing train to: {TRAIN_JSONL}")
    write_jsonl(TRAIN_JSONL, train_idx)
    
    print(f"Writing val to:   {VAL_JSONL}")
    write_jsonl(VAL_JSONL, val_idx)

    print("\n[Done] 정수 스케일링 데이터 생성 완료.")


if __name__ == "__main__":
    main()