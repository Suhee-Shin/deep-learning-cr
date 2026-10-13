"""Read-only local dashboard server. Use an SSH tunnel for remote viewing."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--out', type=Path, default=Path(__file__).parent/'runs/y_pass')
ap.add_argument('--port', type=int, default=8767)
args = ap.parse_args()
if not (args.out/'index.html').exists():
    ap.error('Start or prepare the run before opening the dashboard server.')
handler = partial(SimpleHTTPRequestHandler, directory=str(args.out.resolve()))
server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
print(f'http://127.0.0.1:{args.port}/index.html', flush=True)
server.serve_forever()
