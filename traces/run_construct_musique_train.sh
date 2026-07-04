#!/usr/bin/env bash
# One-pass trace construction: gold + counterfactual + merged (balanced).
#
# Usage:
#   ./run_construct_musique_train.sh
#   LIMIT_EXAMPLES=100 ./run_construct_musique_train.sh   # smoke test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SPLIT="${SPLIT:-train}"
COS_TOPK="${COS_TOPK:-10}"
MAX_WRONG_PER_HOP="${MAX_WRONG_PER_HOP:-3}"
BALANCE_RATIO="${BALANCE_RATIO:-1.0}"
LIMIT_EXAMPLES="${LIMIT_EXAMPLES:-0}"
DEVICE="${DEVICE:-}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-128}"
EXAMPLE_BATCH_SIZE="${EXAMPLE_BATCH_SIZE:-128}"

ARGS=(
  --dataset musique
  --split "${SPLIT}"
  --cos-topk "${COS_TOPK}"
  --max-wrong-per-hop "${MAX_WRONG_PER_HOP}"
  --balance-ratio "${BALANCE_RATIO}"
  --encode-batch-size "${ENCODE_BATCH_SIZE}"
  --example-batch-size "${EXAMPLE_BATCH_SIZE}"
)

if [[ -n "${DEVICE}" ]]; then
  ARGS+=(--device "${DEVICE}")
fi

if [[ "${LIMIT_EXAMPLES}" -gt 0 ]]; then
  ARGS+=(--limit-examples "${LIMIT_EXAMPLES}")
fi

python construct_balanced_traces.py "${ARGS[@]}"
