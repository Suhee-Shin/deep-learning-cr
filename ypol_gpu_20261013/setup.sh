#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
printf '%s\n' 'Ready: .venv/bin/python run_y.py --background'
