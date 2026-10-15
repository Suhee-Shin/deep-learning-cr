"""Serve local results for viewing over an SSH tunnel; never bind publicly."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'runs/x_pass_270')
ap.add_argument('--port', type=int, default=8768)
args = ap.parse_args()
if not args.out.is_dir():
    ap.error('Results do not exist yet. Start run_x.py first.')
handler = partial(SimpleHTTPRequestHandler, directory=str(args.out.resolve()))
print(f'http://127.0.0.1:{args.port}/index.html', flush=True)
ThreadingHTTPServer(('127.0.0.1', args.port), handler).serve_forever()
