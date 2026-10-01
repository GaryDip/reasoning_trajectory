#!/usr/bin/env bash
# Generate HotpotQA GPT decomposition annotations for BART training.
#
# Usage:
#   OPENAI_API_KEY=sk-... ./run_annotate_hotpot.sh
#   PRESET=mixed_dev OPENAI_API_KEY=sk-... ./run_annotate_hotpot.sh
#   PRESET=smoke OPENAI_API_KEY=sk-... WORKERS=1 ./run_annotate_hotpot.sh   # ~10 records, cheap sanity check

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

PRESET="${PRESET:-mixed_train}"
INPUT="${PROJECT_ROOT}/data/raw/hotpotqa/hotpot_train_v1.1.json"
if [[ "${PRESET}" == "mixed_dev" ]]; then
  INPUT="${PROJECT_ROOT}/data/raw/hotpotqa/hotpot_dev_distractor_v1.json"
fi

OUTPUT="${SCRIPT_DIR}/data/hotpot_gpt_${PRESET}.jsonl"

python annotate_hotpot_gpt_decompose.py \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --preset "${PRESET}" \
  --model "${OPENAI_MODEL:-gpt-5-mini}" \
  --workers "${WORKERS:-4}"

echo "Wrote ${OUTPUT}"
