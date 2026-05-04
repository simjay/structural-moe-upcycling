#!/usr/bin/env bash
set -euo pipefail

pip install --upgrade pip Pillow numpy
pip install -e ".[train]"
