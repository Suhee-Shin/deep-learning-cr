"""Read-only validation of all published payloads, topology, fields and metrics."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.ndimage import label,generate_binary_structure
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'manifest.json').read_text())
for name,expected in manifest['sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,'Hash mismatch: '+name
config=json.loads((ROOT/'config.json').read_text());s=generate_binary_structure(3,1)
for key in ('X','Y'):
    p=ROOT/'structures'/f'{key}_binary.npz'
    a=np.load(p)['stack'].astype(bool);assert a.shape==(100,400,400)
    _,n=label(a,s);assert n==1 and a[-1].any()
    air,n=label(~a,s);open_ids=np.unique(air[0]);open_ids=open_ids[open_ids>0]
    assert not ((~a)&~np.isin(air,open_ids)).any()
    digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==config['structure_sha256'][key]
    for wl in range(400,701,50):
        f=ROOT/'fields'/f'{key}_{wl}nm.npz';d=json.loads(f.with_suffix('.json').read_text())
        with np.load(f) as z:
            assert z['E'].shape==(3,128,128,2) and np.isfinite(z['E']).all()
            assert str(z['structure_sha256'])==digest==d['structure_sha256']
            assert np.allclose(z['total_E2'],(abs(z['E'])**2).sum(0))
            assert np.allclose(z['transmission'],[d['tx'],d['ty']],atol=1e-12)
            assert np.isclose(z['output_propagation_um'],1.35)
        tp=d['tx'] if key=='X' else d['ty'];tb=d['ty'] if key=='X' else d['tx']
        assert np.isclose(d['extinction_ratio_db'],10*np.log10(tp/tb))
        assert np.isclose(d['insertion_loss_db'],-10*np.log10(tp))
        if wl%100==0:assert d['reference_check']['passed']
    print(key,'PASS: single connected solid, top-open air, 7 field datasets, metrics')
assert json.loads((ROOT/'COMPLETE.json').read_text())['completed']
print('PASS:',len(manifest['sha256']),'file hashes; 14 cases / 28 input-polarization fields')
