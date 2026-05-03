#!/usr/bin/env bash
set -euo pipefail

# Full pipeline for Mixtral experiment: upcycle → train → eval → cleanup
# for all three initialization methods (direct, gaussian, svd).
# Results are logged to wandb automatically during training.
# Requires 4x H100-80GB.

WORK_DIR="/tmp/mixtral"
CKPT_DIR="/tmp/mixtral-checkpoints"
METHODS=("direct" "gaussian" "svd")

echo "============================================"
echo " Mixtral Experiment: full pipeline"
echo "============================================"

for method in "${METHODS[@]}"; do
    echo ""
    echo "--------------------------------------------"
    echo " Method: ${method}"
    echo "--------------------------------------------"

    model_dir="${WORK_DIR}-${method}"
    output_dir="${CKPT_DIR}/mixtral-${method}"

    # 1. Upcycle
    echo "[1/3] Upcycling (${method})..."
    python3 -m src.mixtral.upcycle --method "${method}" --output "${model_dir}"

    # 2. Train + eval (logs train/loss, eval/loss, router/mean_entropy, gsm8k/accuracy to wandb)
    echo "[2/3] Training + GSM8K eval (${method})..."
    python3 -m src.mixtral.train \
        --model "${model_dir}" \
        --run-name "mixtral-${method}" \
        --output "${CKPT_DIR}"

    # 3. Cleanup
    echo "[3/3] Cleaning up model and checkpoint files (${method})..."
    rm -rf "${model_dir}"
    rm -rf "${output_dir}"

    echo "Done with ${method}."
done

echo ""
echo "============================================"
echo " All Mixtral experiments complete."
echo " Results are on wandb: mixtral-direct, mixtral-gaussian, mixtral-svd"
echo "============================================"
