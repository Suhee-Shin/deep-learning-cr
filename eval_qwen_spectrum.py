import os  #파일 폴더 경로 다루기
import re  #정규표현식, 문자열 패턴 매칭
import json  #json 파일읽고 쓰기. 설정, 평가결과, 데이터셋로그를 저장
import numpy as np #수치계산 기본 라이브러리
import torch #Pytorch 모델 추론/텐서연산/GPU사용
import matplotlib.pyplot as plt #그래프 그리기

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig 
#모델에 맞는 토크나이저 자동로드(텍스트를 토큰으로 바꿔서 모델 입력으로 넣고 출력 토큰을 다시 텍스트로 복원) 
#Casual language model (LLM) 자동 로드
#양자화 로드 설정 큰 모델을 GPU메모리에 올리기 쉽게 INT8/4bit 로드 같은 옵션을 지정할 때 씀
from peft import PeftModel # PEFT(파라미터 효율 미세조정) 모델 래퍼. 베이스 모델(Qwen)위에 LoRA 어댑터를 얹어서 로드. 즉 Qwen을 LoRA로 튜닝한 버전을 평가하려고 필요함

# LoRA 어댑터 : 거대한 LLM은 건드리지 않고, 아주 적은 추가 파라미터만 학습해서 모델을 전공 맞춤으로 만드는 장치 

# ===== 경로 고정 =====
MODEL_DIR = r"C:\LLM\models\Qwen2.5-7B-hf"
ADAPTER_DIR = r"C:\Deep learning\qwen2p5_spectrum_lora_out_hex_int\checkpoint-1000"
VAL_JSONL = r"C:\Deep learning\results_norm\qwen_dataset_hex_int\val.jsonl"

# 결과 저장 폴더
OUT_DIR = r"C:\Deep learning\eval_results_hex_ckpt1000"

# Format tokens in data
OUTPUT_START = "### OUTPUT_SPECTRUM_START"
OUTPUT_END   = "### OUTPUT_SPECTRUM_END"

# ===== 스케일 설정 =====
SCALE_MODE = "norm"   # "norm" (0~1), "db" (-40~0 등), "linear" 등
CLIP_NORM = True      # norm일 때만 0~1 clip할지

# [추가] 정수 변환을 복원하기 위한 나누기 값 (데이터 만들 때 곱했던 1000)
SCALE_FACTOR = 1000.0

# 평가 개수(처음엔 200~1000 추천. 전체 1000도 OK) #평가 데이터(샘플) 몇 개만 골라서 성능을 보겠다는 뜻
N_EVAL = 200 

# 생성 파라미터
MAX_NEW_TOKENS = 2000     # 100개 float를 출력하려면 넉넉히. 숫자 하나가 토큰 몇 개가 될지 몰라서 그런 것. -8, 0.123 등
#TEMPERATURE = 0.0        # 결정적으로 출력(규칙성↑) / 모델이 다음 토큰을 고를 때 얼마나 모험적으로 랜덤하게 고를지를 조절하는 값. 출력의 랜덤성을 조절하는 노이즈 노브
                         # 0에 가까우면 가장 확률 높은 토큰에 더 몰림, 반복 실행해도 거의 같음. 숫자 크면 확률 분포 평평해짐, 출력이 다양, 매번 달라질 수 있음
DO_SAMPLE = False        # false면 사실상 랜덤 거의 없음. True면 temperature가 본격적으로 랜덤성 조절 역할을 함. 나는 100개만 딱 원하니까 False면 됨

# 출력 파싱 설정
# 모델이 생성한 텍스트에서 숫자 100개를 제대로 뽑아냈는지 확인(파싱 성공 예:Parsed OK 287/300) 
# 모델이 뱉은 텍스트를 코드 쓰기 좋은 구조화된 데이터로 바꾸는 과정. 숫자만 정확히 기계가 읽을 수 있게 뽑아내는 것
DECIMALS_EXPECTED = None  # None이면 자유롭게 파싱(소수점 자리 상관X)
# FORMAT_INSTR = (
#     "Return exactly 100 comma-separated floats on ONE line.\n"
#     "Do not include any prefix like 'Example:', '### OUTPUT_SPECTRUM_END', or any text.\n"
#     "Only the 100 numbers.\n"
# )

def extract_prompt_and_gt(text: str):
    # (Ground Truth)GT 블록 추출. 정답 스펙트럼
    m = re.search(
        re.escape(OUTPUT_START) + r"(.*?)" + re.escape(OUTPUT_END),
        text,
        flags=re.S
    )
    if m is None:
        raise ValueError("No OUTPUT block found in val example.")

    block = m.group(1).strip().splitlines()  # GT 블록 추출
    # 숫자 CSV 라인 선택 (콤마 가장 많은 줄)
    block = [ln.strip() for ln in block if ln.strip()]   # 빈 줄 제거
    block.sort(key=lambda x: x.count(","), reverse=True)   # 가장 그럴듯한 숫자 줄 선택
    gt_line = block[0]   # 정답 스펙트럼 

    # prompt는 OUTPUT_START까지만 주고, 그 뒤는 모델이 생성. GT는 절대 모델에게 보여주면 안 됨
    prefix = text.split(OUTPUT_START)[0].rstrip()
    prompt = (
        prefix
        + "\n"
        + OUTPUT_START
        + "\n"
        #+ FORMAT_INSTR
    )
    return prompt, gt_line

def _build_num_pattern(decimals_expected=None) -> str:
    """
    Returns regex pattern for a single number token.
    - If decimals_expected is None: allow int or float, optional sign, optional decimals.
    - If decimals_expected is int: enforce exactly that many decimals when decimal point exists.
      (allows integers too; if you want to force decimals always, remove the integer alternative)
    """
    if decimals_expected is None:
        # int or float, optional sign, no exponent
        return r"-?\d+(?:\.\d+)?"
    else:
        d = int(decimals_expected)
        # either integer OR fixed-decimal float
        return rf"-?\d+(?:\.\d{{{d}}})?"

#LLM이 생성한 텍스트 전체에서 숫자 100개가 콤마로 정확히 구분된 한 줄을 찾아서 numpy 배열로 변환한다. 못 찾으면 None. 
def parse_spectrum_csv_strict(s: str, expected_n: int = 100, decimals_expected=None):  # s: 모델이 생성한 원본 텍스트 전체, expected_n: 기대하는 숫자 개수(100걔)
    """
    실전용 strict:
    - 전체 텍스트에서 '숫자,숫자,숫자,...' 패턴이 있는 라인을 찾고
    - 콤마로 split한 뒤 빈 토큰 제거 후 expected_n개면 성공
    """
    s = s.replace("[", "").replace("]", "") # 대괄호 혹시 있으면 제거
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()] # 줄 단위로 나눠 전처리

    num_pat = r"-?\d+(?:\.\d+)?"  #-?: 음수 가능, \d+: 정수부, (?:\.\d+): 소수부는 선택

    best = None
    best_commas = -1

    for ln in lines:
        # 숫자+콤마 패턴이 충분히 있는 라인만 후보
        if ln.count(",") < expected_n - 1:   #숫자 개수보다 콤마는 1개 적음
            continue

        parts = [p.strip() for p in ln.split(",")]
        parts = [p for p in parts if p != ""]  # trailing comma 방지, 즉 마지막 빈 토큰 제거

        if len(parts) != expected_n: #100개가 아니면 전부 탈락
            continue

        vals = [] #각 토큰이 진짜 숫자인지 확인
        ok = True
        for p in parts:
            if re.fullmatch(num_pat, p) is None:
                ok = False
                break
            try:
                raw_val = float(p)
                restored_val = raw_val / SCALE_FACTOR  # <--- 여기가 핵심. 여기서 1000으로 나누어줌.
                vals.append(restored_val)
            except ValueError:
                ok = False
                break

        if ok:
            # 가장 콤마 많은 라인을 best로 즉 가장 숫자 CSV에 가까운 라인을 최종 채택
            if ln.count(",") > best_commas:
                best = np.array(vals, dtype=np.float32)
                best_commas = ln.count(",")

    return best  #성공하면 np.ndarray(shape=(100,), dtype+float32)로 뜨고, 실패하면 None으로 뜸

def _apply_scale_postprocess(arr: np.ndarray) -> np.ndarray:
    """Only does what your current code actually supports: clip for norm."""
    if SCALE_MODE == "norm" and CLIP_NORM:
        return np.clip(arr, 0.0, 1.0)
    return arr


def _plot_limits_and_ylabel():
    """Make plotting consistent with SCALE_MODE."""
    if SCALE_MODE == "norm":
        return (-0.05, 1.05), "Normalized coupling efficiency"
    elif SCALE_MODE == "db":
        # You can adjust these to your dataset conventions
        return (None, None), "Coupling efficiency (dB)"
    else:
        return (None, None), f"Coupling efficiency ({SCALE_MODE})"

############### MAIN ####################
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # ----- 4bit 로드 (inference도 메모리 절약) -----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    # ----- LoRA adapter 로드 -----
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval() # 드롭 아웃 같은 학습용 동작 끄고 평가/추론 모드로 전환

    # ----- val jsonl 로드 -----
    examples = []
    with open(VAL_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            examples.append(obj["text"])

    n = min(N_EVAL, len(examples))
    print(f"Loaded val examples: {len(examples)}. Evaluating first {n} samples.")

    gts = []  # 정답 스펙트럼 배열
    preds = []  # 모델 예측 스펙트럼 배열
    raw_outputs = []  # 모델이 생성한 원문 텍스트(디버깅/실패분석용)

    # ----- 평가 루프 -----
    for i in range(n):
        prompt, gt_line = extract_prompt_and_gt(examples[i])  # 정답 라인을 gt_line으로 뽑고 모델에게 줄 프롬프트는 GT 빼고 형식 지시문(Format_instr) 붙여서 만듦.
        # if i == 0:
        #     print("\n=== DEBUG GT_LINE ===")
        #     print(gt_line)
        #     print("=== END GT_LINE ===\n")

        # Ground truth 파싱
        gt = parse_spectrum_csv_strict(gt_line, expected_n=100, decimals_expected=DECIMALS_EXPECTED)  # GT 파싱. 정답 문자열->숫자배열

        if gt is None:
            # GT가 파싱이 안되면 데이터셋 문제이므로 건너뛰거나 에러
            print(f"Skipping sample {i}: GT parsing failed.")
            continue
        
        gt = _apply_scale_postprocess(gt)
        # 토크나이징
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():    # 생성(generate)로 예측 출력 만들기. 평가에서는 gradient 필요 없으니 메모리 속도 절약
            out_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.15, #  1.1 ~ 1.2 사이 추천. 반복 방지.
            )

        # 프롬프트를 제외하고 새로 생성된 부분만 디코드
        # 보통 [프롬프트 토큰들]+[새로 생긴 토큰들] 이렇게 나오므로 앞에 0으로 잘라줌
        gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()  # 모델이 실제로 예측 스펙트럼으로 뱉은 텍스트

        # prefix noise 제거 (markdown / 예시 / 라벨 등)
        # [수정 4] Raw Output 확인을 위해 전처리 전 텍스트 저장
        raw_outputs.append(gen_text) 

        # 노이즈 제거 (정규식)
        cleaned_text = re.sub(
            r"^\s*(?:>|-|\*|\d+\)|Example:|Output:|###\s*OUTPUT_SPECTRUM_END)\s*",
            "",
            gen_text
        )

        pred = parse_spectrum_csv_strict(cleaned_text, expected_n=100, decimals_expected=None)
        raw_outputs.append(gen_text)

        # 파싱 실패한 경우 디버그 출력. 숫자 100개를 못 뽑으면 초반 5개까지만 원인을 보겠다는 의미
        if pred is None and i < 5:
            print(f"\n--- FAIL DEBUG Sample {i} ---")
            print("Prompt end:", prompt[-50:]) # 프롬프트 끝부분 확인
            print("Generated (first 300 chars):", gen_text[:300])
            print("-----------------------------")

        # 실패하면 NaN 100개로 채움 / 성공이면 clip(선택)
        if pred is None:
            pred = np.full((100,), np.nan, dtype=np.float32)
        else:
                pred = np.clip(pred, 0.0, 1.0)
        
        # GT와 pred를 리스트에 저장
        gts.append(gt)
        preds.append(pred)
        
        # 숫자가 제대로 나오는지 눈으로 확인 (앞 5개 숫자 + 평균값 출력)
        print(f"Sample {i:03d} | Pred[:5]: {np.round(pred[:5], 4)} ... | Mean: {np.mean(pred):.4f}")
       
        # 20개마다 진행 상황 출력
        if (i + 1) % 20 == 0:
            ok_cnt = np.sum([np.isfinite(p).all() for p in preds])
            print(f"[{i+1}/{n}] Processed. Success rate: {ok_cnt}/{len(preds)}")

    # 단, gts와 preds 개수가 다를 수 있으므로(GT 파싱 실패 등) stack 전에 체크 필요
    if len(gts) == 0:
        print("No valid samples.")
        return
    
    # 리스트 -> (n,100) 행렬로 변환
    gts = np.stack(gts, axis=0)     # (n,100)
    preds = np.stack(preds, axis=0) # (n,100)

    # ----- 실패 샘플 제외하고 metric 계산 -----
    good_mask = np.isfinite(preds).all(axis=1)
    gts_good = gts[good_mask]
    preds_good = preds[good_mask]
    

    if len(gts_good) == 0:   # 전부 파싱 실패
        print("No successful predictions (all failed to parse 100 numbers).")
        # raw output 저장만 하고 종료
        with open(os.path.join(OUT_DIR, "raw_outputs.txt"), "w", encoding="utf-8") as f:
            for i, t in enumerate(raw_outputs):
                f.write(f"--- sample {i} ---\n{t}\n\n")
        return

    # ---- 예측이 상수로 붕괴했는지 진단 (여기서부터 안전) ----
    # 표준편차(std)를 두 가지 방식으로. 파장별로 샘플들 사이의 표준편차./ 샘플별로 100개 파장축에서의 표준편차. 평평하면 std 작음, 피크나 딥이 많으면 큼
    mean_std_wl = float(np.mean(np.std(preds_good, axis=0)))  # 파장별(인덱스별) 샘플 분산 평균
    mean_std_sample = float(np.mean(np.std(preds_good, axis=1)))  # 샘플별 스펙트럼 내부 분산 평균
    print("Mean std over wavelength =", mean_std_wl)
    print("Mean std over samples    =", mean_std_sample)

    # MAE / RMSE (전체 100포인트 평균)
    # GT(정답100개) vs Pred(모델이 생성한 100개)의 평균오차. 실제 스펙트럼 모양이 맞는지 확인
    mae = float(np.mean(np.abs(preds_good - gts_good)))
    rmse = float(np.sqrt(np.mean((preds_good - gts_good) ** 2)))

    print(f"\n Parsed OK: {len(gts_good)}/{n}")
    print(f"MAE  = {mae:.6f}")
    print(f"RMSE = {rmse:.6f}")

    # ----- 결과 저장 -----
    np.save(os.path.join(OUT_DIR, "gts.npy"), gts)
    np.save(os.path.join(OUT_DIR, "preds.npy"), preds)
    with open(os.path.join(OUT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Parsed OK: {len(gts_good)}/{n}\n")
        f.write(f"MAE: {mae:.8f}\n")
        f.write(f"RMSE: {rmse:.8f}\n")
        f.write(f"MeanStd(wavelength): {mean_std_wl:.8f}\n")
        f.write(f"MeanStd(samples): {mean_std_sample:.8f}\n")

    # 실패한 raw 생성도 저장
    with open(os.path.join(OUT_DIR, "raw_outputs.txt"), "w", encoding="utf-8") as f:
        for i, t in enumerate(raw_outputs):
            f.write(f"--- sample {i} ---\n{t}\n\n")

    # ----- GT vs Pred plot (예시 몇 개) -----
    # 성공 샘플 중 앞에서 5개 그리기
    idxs = np.where(good_mask)[0][:5]
    x = np.arange(100)

    (ymin, ymax), ylab = _plot_limits_and_ylabel()

    for k, idx in enumerate(idxs):
        plt.figure(figsize=(7, 4))
        plt.plot(x, gts[idx], label="GT")
        plt.plot(x, preds[idx], label="Pred")
        if ymin is not None and ymax is not None:
            plt.ylim(ymin, ymax)
        plt.xlabel("Wavelength index (0..99)")
        plt.ylabel(ylab)
        plt.title(f"Validation sample #{idx} (parsed OK)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        out_png = os.path.join(OUT_DIR, f"gt_vs_pred_{k}_sample{idx}.png")
        plt.savefig(out_png, dpi=150)
        plt.show()

    print("\nSaved outputs to:", OUT_DIR)

# 이 파일을 직접 실행하면 main()을 돌린다는 뜻.
if __name__ == "__main__":
    main()
