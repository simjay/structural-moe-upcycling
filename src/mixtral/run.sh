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
    echo "[1/4] Upcycling (${method})..."
    python3 -m src.mixtral.upcycle --method "${method}" --output "${model_dir}"

    # 2. Train (logs train/loss, eval/loss, router/mean_entropy to wandb)
    echo "[2/4] Training (${method})..."
    python3 -m src.mixtral.train \
        --model "${model_dir}" \
        --run-name "mixtral-${method}" \
        --output "${CKPT_DIR}"

    # 3. Eval (GSM8K accuracy on final checkpoint)
    echo "[3/4] Evaluating GSM8K (${method})..."
    python3 -m src.eval.gsm8k --model "${output_dir}/final" --max-samples 200

    # 4. Cleanup
    echo "[4/4] Cleaning up model and checkpoint files (${method})..."
    rm -rf "${model_dir}"
    rm -rf "${output_dir}"

    echo "Done with ${method}."
done

echo ""
echo "============================================"
echo " All Mixtral experiments complete."
echo " Results are on wandb: mixtral-direct, mixtral-gaussian, mixtral-svd"
echo "============================================"
