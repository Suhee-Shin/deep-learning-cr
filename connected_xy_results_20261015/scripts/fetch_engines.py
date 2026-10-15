"""Fetch pinned source packages only; never start an optimizer or install packages."""
from pathlib import Path
import json,hashlib,subprocess,argparse
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--destination',default=str(ROOT/'engine_cache'));args=p.parse_args()
dest=Path(args.destination).resolve();dest.mkdir(parents=True,exist_ok=True)
info=json.loads((ROOT/'engine_sources.json').read_text())
for key,cfg in info.items():
    repo=dest/key
    if repo.exists():raise RuntimeError(f'Use a new empty destination; already exists: {repo}')
    subprocess.run(['git','clone','--filter=blob:none','--no-checkout','--single-branch','--branch',cfg['branch'],cfg['repository'],str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'config','core.autocrlf','false'],check=True)
    subprocess.run(['git','-C',str(repo),'sparse-checkout','init','--cone'],check=True)
    subprocess.run(['git','-C',str(repo),'sparse-checkout','set',cfg['package_folder']+'/src',cfg['package_folder']+'/third_party'],check=True)
    subprocess.run(['git','-C',str(repo),'checkout','--detach',cfg['commit']],check=True)
    engine=repo/cfg['package_folder']
    for name,expected in cfg['forward_source_sha256'].items():
        assert hashlib.sha256((engine/name).read_bytes()).hexdigest()==expected,f'Source mismatch: {key}/{name}'
    print(key,'verified engine:',engine)
