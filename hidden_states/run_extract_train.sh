#!/usr/bin/env bash
# Extract hidden states from merged traces (batched full forward per prefix).
#
# Usage:
#   ./run_extract_train.sh
#   LIMIT=100 DEVICE=cuda:1 BATCH_SIZE=32 ./run_extract_train.sh
#   DEVICE=cuda:1 SPLIT=train ./run_extract_train.sh && DEVICE=cuda:1 SPLIT=dev ./run_extract_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SPLIT="${SPLIT:-train}"
DATASET="${DATASET:-musique}"
DEVICE="${DEVICE:-cuda:0}"
LIMIT="${LIMIT:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TRACE_CHUNK_SIZE="${TRACE_CHUNK_SIZE:-32}"
LENGTH_BUCKET_TOKENS="${LENGTH_BUCKET_TOKENS:-256}"

ARGS=(
  --split "${SPLIT}"
  --dataset "${DATASET}"
  --trace-source merged
  --device "${DEVICE}"
  --dtype bfloat16
  --batch-size "${BATCH_SIZE}"
  --trace-chunk-size "${TRACE_CHUNK_SIZE}"
  --length-bucket-tokens "${LENGTH_BUCKET_TOKENS}"
  --resume
)

if [[ "${LIMIT}" -gt 0 ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

python extract_hidden_states.py "${ARGS[@]}"
