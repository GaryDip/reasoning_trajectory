#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

CKPT="outputs/bart_decomposer_musique_2wiki_repro/checkpoint-1425"
OUT_DIR="outputs/bart_decomposer_musique_2wiki_repro"
RAW_DIR="$OUT_DIR/raw_predictions"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_BEAMS="${NUM_BEAMS:-10}"

MUSIQUE_DEV="${PROJECT_ROOT}/data/raw/musique/musique_ans_v1.0_dev.jsonl"
TWOWIKI_DEV="${PROJECT_ROOT}/data/raw/2wikimultihopqa/dev.json"
HOTPOT_DEV="${PROJECT_ROOT}/data/raw/hotpotqa/hotpot_dev_distractor_v1.json"

mkdir -p "$RAW_DIR"

python predict.py \
  --model-dir "$CKPT" \
  --dataset musique \
  --dev-file "$MUSIQUE_DEV" \
  --output "$RAW_DIR/predict_musique_dev.jsonl" \
  --fp16 \
  --batch-size "$BATCH_SIZE" \
  --num-beams "$NUM_BEAMS"

python predict.py \
  --model-dir "$CKPT" \
  --dataset 2wiki \
  --dev-file "$TWOWIKI_DEV" \
  --output "$RAW_DIR/predict_2wiki_dev.jsonl" \
  --fp16 \
  --batch-size "$BATCH_SIZE" \
  --num-beams "$NUM_BEAMS"

python predict.py \
  --model-dir "$CKPT" \
  --dataset hotpot \
  --dev-file "$HOTPOT_DEV" \
  --output "$RAW_DIR/predict_hotpot_dev.jsonl" \
  --fp16 \
  --batch-size "$BATCH_SIZE" \
  --num-beams "$NUM_BEAMS"
