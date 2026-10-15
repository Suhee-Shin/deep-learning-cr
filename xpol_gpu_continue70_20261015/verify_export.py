"""Verify the downloaded bundle and optionally a freshly restored checkpoint."""
import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from run_x import digest, read, verified_updates, verify_code

ap = argparse.ArgumentParser()
ap.add_argument('--restored', type=Path)
options = ap.parse_args()
verify_code()
manifest = read(ROOT/'seed/manifest.json')
for item in manifest['parts']:
    path = ROOT/'seed'/item['file']
    assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256'], item['file']
if options.restored:
    out = options.restored.resolve()
    for name in ('state.pt', 'best_state.pt', 'best_binary.npz', 'history.json'):
        assert digest(out/name) == manifest['files'][name]['sha256'], name
    assert verified_updates(out) == 200
    state = torch.load(out/'state.pt', map_location='cpu', weights_only=False)
    import hashlib
    for key in ('theta', 'm', 'v'):
        assert hashlib.sha256(state[key].numpy().tobytes()).hexdigest() == manifest['tensors'][key]['sha256'], key
print('PASS: frozen source, checkpoint parts' + (' and exact restored state/Adam/history' if options.restored else ''))
