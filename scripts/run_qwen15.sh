#!/usr/bin/env bash
set -euo pipefail

# Qwen1.5 experiment pipeline.
# Step 0: Evaluate the dense model (no upcycling) as a reference baseline.
# Then for each config: upcycle → train (step-0 metrics + 400 steps + GSM8K) → cleanup.
# Everything is logged to wandb in a single run per config.
#
# Configs:
#   0. dense baseline                   (no upcycling, eval only)
#   1. direct                          (baseline)
#   2. gaussian  sigma=0.5             (moderate random noise)
#   3. gaussian  sigma=1.0             (heavy random noise)
#   4. svd       k=512  svd_scale=0.1  (light structured perturbation)
#   5. svd       k=512  svd_scale=0.5  (heavy structured perturbation)

CKPT_DIR="/tmp/moe-checkpoints"
RESULTS_DIR="results/qwen15"
WORK_DIR="/tmp/qwen-moe"

mkdir -p "${RESULTS_DIR}"

CONFIGS=(
    "direct|qwen-direct||"
    "gaussian|qwen-gaussian-s0.5|--sigma 0.5|"
    "gaussian|qwen-gaussian-s1.0|--sigma 1.0|"
    "svd|qwen-svd-k512-s0.1|--k 512 --svd-scale 0.1|"
    "svd|qwen-svd-k512-s0.5|--k 512 --svd-scale 0.5|"
)

echo "============================================"
echo " Qwen1.5 Experiment (5 configs + dense baseline)"
echo "============================================"

echo ""
echo "--------------------------------------------"
echo " Dense baseline (no upcycling, eval only)"
echo "--------------------------------------------"
python3 -m src.eval.gsm8k --model Qwen/Qwen1.5-1.8B --max-samples 1319 --run-name qwen-dense-baseline

echo ""
for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r method run_name upcycle_args _ <<< "${entry}"

    echo ""
    echo "--------------------------------------------"
    echo " ${run_name}"
    echo "--------------------------------------------"

    model_dir="${WORK_DIR}-${run_name}"
    output_dir="${CKPT_DIR}/${run_name}"

    echo "[1/3] Upcycling (${method})..."
    python3 -m src.qwen15.upcycle --method "${method}" ${upcycle_args} --output "${model_dir}"

    echo "[2/3] Training (step-0 + 400 steps + GSM8K)..."
    python3 -m src.qwen15.train \
        --model "${model_dir}" \
        --run-name "${run_name}" \
        --output "${CKPT_DIR}"

    echo "[3/3] Saving results and cleaning up..."
    cp "${output_dir}/results.json" "${RESULTS_DIR}/${run_name}.json" 2>/dev/null || true
    rm -rf "${model_dir}"
    rm -rf "${output_dir}"

    echo "Done with ${run_name}."
done

echo ""
echo "============================================"
echo " All experiments complete."
echo " Results: ${RESULTS_DIR}/"
echo "============================================"
