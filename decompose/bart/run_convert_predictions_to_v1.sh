#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OUT_DIR="outputs/bart_decomposer_musique_2wiki_repro"
RAW_DIR="$OUT_DIR/raw_predictions"
V1_DIR="$OUT_DIR/v1_predictions"

mkdir -p "$V1_DIR"

python convert_predictions_to_v1.py \
  --input "$RAW_DIR/predict_musique_dev.jsonl" \
  --output "$V1_DIR/predict_musique_dev_v1.jsonl" \
  --include-gold

python convert_predictions_to_v1.py \
  --input "$RAW_DIR/predict_2wiki_dev.jsonl" \
  --output "$V1_DIR/predict_2wiki_dev_v1.jsonl"

python convert_predictions_to_v1.py \
  --input "$RAW_DIR/predict_hotpot_dev.jsonl" \
  --output "$V1_DIR/predict_hotpot_dev_v1.jsonl"
