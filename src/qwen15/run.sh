#!/usr/bin/env bash
set -euo pipefail

# Full pipeline for Qwen1.5 experiment: upcycle → train → eval → cleanup
# for direct, gaussian, and multiple SVD configurations.
# Results are logged to wandb and saved as JSON locally.

WORK_DIR="/tmp/qwen-moe"
CKPT_DIR="/tmp/moe-checkpoints"
RESULTS_DIR="results/qwen15"

mkdir -p "${RESULTS_DIR}"

run_experiment() {
    local method="$1"
    local run_name="$2"
    local upcycle_extra="${3:-}"

    echo ""
    echo "--------------------------------------------"
    echo " Run: ${run_name}"
    echo "--------------------------------------------"

    local model_dir="${WORK_DIR}-${run_name}"
    local output_dir="${CKPT_DIR}/${run_name}"

    echo "[1/3] Upcycling (${method})..."
    python3 -m src.qwen15.upcycle --method "${method}" ${upcycle_extra} --output "${model_dir}"

    echo "[2/3] Training + GSM8K eval..."
    python3 -m src.qwen15.train \
        --model "${model_dir}" \
        --run-name "${run_name}" \
        --output "${CKPT_DIR}"

    echo "[3/3] Saving results and cleaning up..."
    cp "${output_dir}/results.json" "${RESULTS_DIR}/${run_name}.json" 2>/dev/null || true
    rm -rf "${model_dir}"
    rm -rf "${output_dir}"

    echo "Done with ${run_name}."
}

echo "============================================"
echo " Qwen1.5 Experiment: full pipeline"
echo "============================================"

# Baselines
run_experiment direct  "qwen-direct"
run_experiment gaussian "qwen-gaussian" "--sigma 0.5"

# SVD sweep: k,scale pairs
SVD_CONFIGS=("64,0.5" "128,0.5" "256,0.5")

for config in "${SVD_CONFIGS[@]}"; do
    IFS=',' read -r k scale <<< "${config}"
    run_experiment svd "qwen-svd-k${k}-s${scale}" "--k ${k} --svd-scale ${scale}"
done

echo ""
echo "============================================"
echo " All Qwen1.5 experiments complete."
echo " Results: ${RESULTS_DIR}/"
echo "============================================"
