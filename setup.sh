#!/usr/bin/env bash
set -euo pipefail

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate sentencepiece protobuf
