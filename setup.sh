#!/usr/bin/env bash
set -euo pipefail

pip install --upgrade pip

pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo

pip install -e ".[train]"
