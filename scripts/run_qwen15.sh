#!/usr/bin/env bash
set -euo pipefail

# Final experiment pipeline for the paper:
#   Part A: Step-0 analysis (9 configs, no training, outputs JSON)
#   Part B: Training dynamics (3 configs, 300 steps, logs to wandb)

STEP0_DIR="results/qwen15/step0"
CKPT_DIR="/tmp/moe-checkpoints"
RESULTS_DIR="results/qwen15"

mkdir -p "${STEP0_DIR}" "${RESULTS_DIR}"

# ============================================================
#  Part A: Step-0 sweep (quality vs diversity, no training)
# ============================================================

echo "============================================"
echo " Part A: Step-0 Analysis"
echo "============================================"

# Direct (baseline)
echo "[step0] direct"
python3 -m src.qwen15.init_analysis \
    --method direct \
    --output "${STEP0_DIR}/direct.json"

# Gaussian sweep (sigma = 0.1, 0.3, 0.5)
for sigma in 0.1 0.3 0.5; do
    echo "[step0] gaussian sigma=${sigma}"
    python3 -m src.qwen15.init_analysis \
        --method gaussian --sigma "${sigma}" \
        --output "${STEP0_DIR}/gaussian-s${sigma}.json"
done

# SVD sweep: vary k with fixed scale=0.5
for k in 64 128 256; do
    echo "[step0] svd k=${k} scale=0.5"
    python3 -m src.qwen15.init_analysis \
        --method svd --k "${k}" --svd-scale 0.5 \
        --output "${STEP0_DIR}/svd-k${k}-s0.5.json"
done

# SVD sweep: vary scale with fixed k=128
for scale in 0.1 0.3; do
    echo "[step0] svd k=128 scale=${scale}"
    python3 -m src.qwen15.init_analysis \
        --method svd --k 128 --svd-scale "${scale}" \
        --output "${STEP0_DIR}/svd-k128-s${scale}.json"
done

echo ""
echo "============================================"
echo " Part A complete. Results in ${STEP0_DIR}/"
echo "============================================"
echo ""

# ============================================================
#  Part B: Training dynamics (3 runs, 300 steps each)
# ============================================================

echo "============================================"
echo " Part B: Training Dynamics (300 steps each)"
echo "============================================"

WORK_DIR="/tmp/qwen-moe"

run_training() {
    local method="$1"
    local run_name="$2"
    local upcycle_extra="${3:-}"

    echo ""
    echo "--------------------------------------------"
    echo " Training: ${run_name}"
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

run_training direct  "qwen-direct-final"
run_training gaussian "qwen-gaussian-final" "--sigma 0.5"
run_training svd     "qwen-svd-final"      "--k 256 --svd-scale 0.5"

echo ""
echo "============================================"
echo " All experiments complete."
echo " Step-0 results: ${STEP0_DIR}/"
echo " Training results: ${RESULTS_DIR}/"
echo " wandb: qwen-direct-final, qwen-gaussian-final, qwen-svd-final"
echo "============================================"
