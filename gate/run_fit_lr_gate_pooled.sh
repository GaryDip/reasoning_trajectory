#!/usr/bin/env bash
# Train pooled LR gate on train hidden states; evaluate on dev.
#
# Usage:
#   ./run_fit_lr_gate_pooled.sh
#   TARGET_FPR=0.15 SPLIT_EVAL=dev ./run_fit_lr_gate_pooled.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

HS_ROOT="${HS_ROOT:-${SCRIPT_DIR}/../hidden_states}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-${SCRIPT_DIR}/artifacts_pooled}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results_pooled}"
TARGET_FPR="${TARGET_FPR:-0.15}"
SPLIT_EVAL="${SPLIT_EVAL:-dev}"
PCA_DIM="${PCA_DIM:-64}"
C="${C:-1.0}"
MAX_J="${MAX_J:-3}"

python fit_lr_gate_pooled.py \
  --hs-root "${HS_ROOT}" \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  --out-dir "${RESULTS_DIR}" \
  --split-eval "${SPLIT_EVAL}" \
  --target-fpr "${TARGET_FPR}" \
  --pca-dim "${PCA_DIM}" \
  --C "${C}" \
  --max-j "${MAX_J}"
