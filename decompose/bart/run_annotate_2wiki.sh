#!/usr/bin/env bash
# Generate 2Wiki GPT decomposition annotations for BART training.
#
# Usage:
#   ./run_annotate_2wiki.sh
#   PRESET=mixed_dev ./run_annotate_2wiki.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

PRESET="${PRESET:-mixed_train}"
INPUT="${PROJECT_ROOT}/data/raw/2wikimultihopqa/train.json"
if [[ "${PRESET}" == "mixed_dev" ]]; then
  INPUT="${PROJECT_ROOT}/data/raw/2wikimultihopqa/dev.json"
fi

OUTPUT="${SCRIPT_DIR}/data/2wiki_gpt_${PRESET}.jsonl"

python annotate_2wiki_gpt_decompose.py \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --preset "${PRESET}" \
  --model "${OPENAI_MODEL:-gpt-4.1-mini}"

echo "Wrote ${OUTPUT}"
