import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataclasses import dataclass
from typing import List, Dict, Any


FOUND = 0
TOTAL = 0


# ===== 경로 고정 =====
MODEL_DIR = r"C:\LLM\models\Qwen2.5-7B-hf"
DATA_DIR  = r"C:\Deep learning\results_norm\qwen_dataset_hex_int"
OUT_DIR   = r"C:\Deep learning\qwen2p5_spectrum_lora_out_hex_int"

TRAIN_JSONL = os.path.join(DATA_DIR, "train.jsonl")
VAL_JSONL   = os.path.join(DATA_DIR, "val.jsonl")

# 태그 (make_jsonl과 반드시 동일)
OUTPUT_START = "### OUTPUT_SPECTRUM_START"

# ===== 4-bit QLoRA =====
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

# ===== LoRA =====
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False

# ===== 데이터 로드 =====
dataset = load_dataset("json", data_files={
    "train": TRAIN_JSONL,
    "validation": VAL_JSONL,
})

MAX_LEN = 2048

OUTPUT_START = "### OUTPUT_SPECTRUM_START"

# 토크나이저가 개행/공백과 합쳐 토큰화할 수 있어서 여러 패턴을 준비
OUTPUT_START_PATTERNS = [
    OUTPUT_START,
    "\n" + OUTPUT_START,
    OUTPUT_START + "\n",
    "\n" + OUTPUT_START + "\n",
    "\n\n" + OUTPUT_START + "\n",
]

OUTPUT_START_IDS_LIST = [
    tokenizer(p, add_special_tokens=False)["input_ids"] for p in OUTPUT_START_PATTERNS
]


def find_sublist(haystack, needle):
    """Return first index where needle starts in haystack, or None."""
    n = len(needle)
    if n == 0:
        return None
    for i in range(len(haystack) - n + 1):
        if haystack[i:i+n] == needle:
            return i
    return None

OUTPUT_START = "### OUTPUT_SPECTRUM_START"

def tokenize_and_mask(batch):
    input_ids_list, attention_mask_list, labels_list = [], [], []

    for text in batch["text"]:
        text = text + tokenizer.eos_token
        tok = tokenizer(text, truncation=True, max_length=MAX_LEN, padding=False)
        input_ids = tok["input_ids"]
        attention_mask = tok["attention_mask"]
        labels = input_ids.copy()

        # 토큰에서 OUTPUT_START 패턴 찾기
        start_pos = None
        for needle in OUTPUT_START_IDS_LIST:
            p = find_sublist(input_ids, needle)
            if p is not None:
                # OUTPUT_START 토큰들 "포함해서" 마스킹하려면 p+len(needle)까지
                start_pos = p + len(needle)
                break

        if start_pos is None:
            # 못 찾으면 전체 마스킹
            labels = [-100] * len(labels)
        else:
            for j in range(min(start_pos, len(labels))):
                labels[j] = -100

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {"input_ids": input_ids_list, "attention_mask": attention_mask_list, "labels": labels_list}



tokenized = dataset.map(tokenize_and_mask, batched=True, remove_columns=["text"])

# tokenized 생성 직후 확인 코드 추가
num_ignored = 0
for label in tokenized["train"]["labels"][:100]: # 앞 100개만 확인
    if all(l == -100 for l in label):
        num_ignored += 1

print(f"Checking first 100 samples: {num_ignored} samples were fully ignored (tag not found).")
if num_ignored > 10:
    print("WARNING: 태그를 못 찾는 데이터가 너무 많습니다. OUTPUT_START_PATTERNS를 확인하세요.")

@dataclass
class DataCollatorForCausalLMWithLabelsPadding:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # input_ids/attention_mask는 tokenizer.pad로 동적 패딩
        batch = self.tokenizer.pad(
            {
                "input_ids": [f["input_ids"] for f in features],
                "attention_mask": [f["attention_mask"] for f in features],
            },
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]

        # labels는 -100으로 max_len까지 패딩
        labels = []
        for f in features:
            lab = f["labels"]
            if len(lab) < max_len:
                lab = lab + [-100] * (max_len - len(lab))
            else:
                lab = lab[:max_len]
            labels.append(lab)

        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


# 중요한 변경: LM collator 말고 default_data_collator
data_collator = DataCollatorForCausalLMWithLabelsPadding(tokenizer=tokenizer)



common_kwargs = dict(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,    # batch_size=1, grad_accum=32이면, 32번의 step이 돌아야 1번의 update가 일어남
    learning_rate=1e-4,
    num_train_epochs=2,          # 일단 3~5 epoch부터 추천 (빠르게 정상화됨)
    logging_steps=5,            # logging_steps=10으로 설정하면 로그가 너무 뜸하게 찍힐 수 있음(실제 320 샘플마다 1번)
    eval_steps=200,
    save_steps=200,
    save_total_limit=2,          # 2 or 3
    fp16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    report_to="none",
)

sig = TrainingArguments.__init__.__code__.co_varnames
if "eval_strategy" in sig:
    args = TrainingArguments(eval_strategy="steps", **common_kwargs)
else:
    args = TrainingArguments(evaluation_strategy="steps", **common_kwargs)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
)

# 처음부터 학습
trainer.train()
trainer.save_model(OUT_DIR)
print("Saved to:", OUT_DIR)
