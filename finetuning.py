import torch
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

from peft import LoraConfig

lora_config = LoraConfig(
    r=32,                     # ← 47GB라서 16보다 32 추천
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="qwen2p5_lora_out",
    per_device_train_batch_size=2,     # ← 여유 있음
    gradient_accumulation_steps=8,     # effective batch = 16
    max_steps=2000,                    # or num_train_epochs
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=3,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    report_to="none",
)
