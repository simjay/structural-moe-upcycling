#!/usr/bin/env bash
set -euo pipefail

pip install --upgrade pip Pillow
pip install -e ".[train]"
