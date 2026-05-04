#!/usr/bin/env bash
set -euo pipefail

pip install --upgrade pip Pillow numpy scipy
pip install -e ".[train]"
