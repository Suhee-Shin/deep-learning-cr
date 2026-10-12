"""Verify published source/data hashes and the saved baseline trace contract."""
from pathlib import Path
import hashlib
import json
import numpy as np

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'MANIFEST.json').read_text())
for relative,expected in manifest.items():
    actual=hashlib.sha256((root/relative).read_bytes()).hexdigest()
    if actual!=expected: raise ValueError(f'Hash mismatch: {relative}')
meta=json.loads((root/'results/baseline_0V/run.json').read_text())
for relative,expected in meta['code_hashes'].items():
    if hashlib.sha256((root/'vendor/TorchFDTD'/relative).read_bytes()).hexdigest()!=expected:
        raise ValueError(f'Original baseline code mismatch: {relative}')
with np.load(root/'results/baseline_0V/trace_000.npz',allow_pickle=False) as data:
    assert data['signals'].shape==(80000,12)
    assert np.isfinite(data['signals']).all()
    assert float(data['voltage_v'])==0.
    assert str(data['fingerprint'])==meta['fingerprint']
    assert float(data['time_step_s'])==meta['time_step_s']
print(f'PASS: {len(manifest)} file hashes; exact baseline source; finite 80000 x 12 trace, 0 V.')
print('This verifies data integrity, not mode identity, convergence or tuning sensitivity.')
