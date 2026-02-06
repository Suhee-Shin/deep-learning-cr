import json
import matplotlib.pyplot as plt

STATE_PATH = r"C:\Deep learning\qwen2p5_spectrum_lora_out_hex\checkpoint-1000\trainer_state.json"

with open(STATE_PATH, "r") as f:
    state = json.load(f)

steps = []
losses = []

for log in state["log_history"]:
    if "loss" in log:
        steps.append(log["step"])
        losses.append(log["loss"])

plt.figure(figsize=(6,4))
plt.plot(steps, losses, marker="o", linewidth=1)
plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.tight_layout()
plt.show()