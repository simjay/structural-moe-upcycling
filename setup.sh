#!/usr/bin/env bash
set -euo pipefail

pip install --upgrade pip

# Ensure CUDA toolkit is visible for building flash-attn
if [ -z "${CUDA_HOME:-}" ]; then
    for d in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-12.8; do
        if [ -d "$d" ]; then export CUDA_HOME="$d"; break; fi
    done
fi

if [ -z "${CUDA_HOME:-}" ]; then
    echo "⚠ CUDA_HOME not found — installing CUDA toolkit..."
    sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-8
    export CUDA_HOME=/usr/local/cuda-12.8
fi

export PATH="$CUDA_HOME/bin:$PATH"

pip install flash-attn --no-build-isolation

pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo

pip install -e ".[train]"
