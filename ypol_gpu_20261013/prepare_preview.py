"""Publish a truthful pre-run dashboard without evaluating or inventing a spectrum."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
from portable_report import json_save, publish, read

out = ROOT/'preview'
out.mkdir(exist_ok=True)
manifest = read(ROOT/'seed/manifest.json')
json_save(out/'seed_provenance.json', {k: manifest[k] for k in
    ('source_evaluated_step', 'source_updates_completed', 'beta', 'rotation', 'adam_updates')})
json_save(out/'supervisor.json', dict(status='ready', detail='Launch on the separate GPU computer.'))
publish(out)
print(out/'index.html')
