#!/usr/bin/env bash
# Paraphrase K=3/4 train GT NL decompositions → train_nl_enhance.jsonl
#
# Usage:
#   ./run_enhance_decompose_train.sh
#   LIMIT=20 NUM_VARIANTS=1 CUDA_VISIBLE_DEVICES=1 ./run_enhance_decompose_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_VARIANTS="${NUM_VARIANTS:-1}"
LIMIT="${LIMIT:-0}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEM="${GPU_MEM:-0.90}"

ARGS=(
  --model "${MODEL}"
  --batch-size "${BATCH_SIZE}"
  --num-variants "${NUM_VARIANTS}"
  --tensor-parallel-size "${TP_SIZE}"
  --gpu-memory-utilization "${GPU_MEM}"
)

if [[ "${LIMIT}" -gt 0 ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

python enhance_decompose_nl.py "${ARGS[@]}"
