"""Render the frozen baseline only; no optimizer or electromagnetic solve."""
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
from portable_report import read, publish, save_history_plots, write_summary

out = ROOT/'baseline'
history, best = read(out/'history.json'), read(out/'best.json')
save_history_plots(out, history, best)
write_summary(out, [], history, best, SimpleNamespace(max_updates=270))
publish(out, dict(status='baseline', updates_completed=200))
print(out/'index.html')
