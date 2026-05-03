#!/usr/bin/env bash
set -euo pipefail

pip install --upgrade pip
pip install -e ".[train]"
