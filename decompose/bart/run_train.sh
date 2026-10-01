#!/usr/bin/env bash
# Train BART decomposer on MuSiQue + 2Wiki (+ optionally HotpotQA) GPT-mixed data.
#
# Usage:
#   ./run_train.sh
#   LEARNING_RATE=1e-5 BATCH_SIZE=8 ./run_train.sh
#   TRAIN_MUSIQUE_ONLY=1 ./run_train.sh
#   INCLUDE_HOTPOT=1 ./run_train.sh   # also needs run_annotate_hotpot.sh run first

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA="${SCRIPT_DIR}/data"
if [[ "${INCLUDE_HOTPOT:-0}" == "1" ]]; then
  OUT_DIR="${OUT_DIR:-outputs/bart_decomposer_musique_2wiki_hotpot}"
else
  OUT_DIR="${OUT_DIR:-outputs/bart_decomposer_musique_2wiki_repro}"
fi
MUSIQUE_TRAIN="${DATA}/musique_raw/musique_ans_gold_context_version_train.jsonl"
MUSIQUE_DEV="${DATA}/musique_raw/musique_ans_gold_context_version_dev.jsonl"
TWOWIKI_GPT_TRAIN="${DATA}/2wiki_gpt_mixed_train.jsonl"
HOTPOT_GPT_TRAIN="${DATA}/hotpot_gpt_mixed_train.jsonl"

LEARNING_RATE="${LEARNING_RATE:-3e-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-100}"
BF16="${BF16:-1}"

if [[ "${TRAIN_MUSIQUE_ONLY:-0}" == "1" ]]; then
  TRAIN_FILES=(--train-file "${MUSIQUE_TRAIN}")
else
  TRAIN_FILES=(--train-file "${MUSIQUE_TRAIN}" "${TWOWIKI_GPT_TRAIN}")
  if [[ "${INCLUDE_HOTPOT:-0}" == "1" ]]; then
    TRAIN_FILES+=("${HOTPOT_GPT_TRAIN}")
  fi
fi

TRAIN_ARGS=(
  "${TRAIN_FILES[@]}"
  --dev-file "${MUSIQUE_DEV}"
  --output-dir "${OUT_DIR}"
  --learning-rate "${LEARNING_RATE}"
  --batch-size "${BATCH_SIZE}"
  --max-target-length "${MAX_TARGET_LENGTH}"
)

if [[ "${BF16}" == "1" ]]; then
  TRAIN_ARGS+=(--bf16)
fi

echo "Training BART decomposer → ${OUT_DIR}"
python train.py "${TRAIN_ARGS[@]}"
