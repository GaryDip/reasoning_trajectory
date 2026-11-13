#!/usr/bin/env bash
# Refit all three gate variants with a UNIFORM layer 15, the choice that wins when layers are
# selected on a train-internal held-out split instead of on dev
# (see gate/select_layers_train_holdout.py and gate/pilot_layer_compare/layer_selection_train_holdout.json:
#  uniform 15 beats the production 15/15/23/23 by +0.0034 AUC at j=2 and +0.0168 at j=3).
#
# All three variants must move together: the A/B/C/D three-signal comparison is only a clean
# single-variable ablation if v2 (delta), h-only and v3 share the same layer policy.
#
# CPU only, a few minutes each. Writes to NEW directories -- the production artifacts under
# gate/artifacts_pooled_v2, gate/gate_h_only/... and gate/gate_v3/artifacts_pooled_v3 are untouched.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LAYERS="${LAYERS:-0:15,1:15,2:15,3:15}"
TAG="${TAG:-layer15_uniform}"
OUT_ROOT="${SCRIPT_DIR}/${TAG}"
mkdir -p "${OUT_ROOT}"

echo "=== [1/3] gate v3 (h + delta) — 主方法"
python gate_v3/fit_lr_gate_pooled_v3.py \
  --per-j-layers "${LAYERS}" \
  --artifacts-dir "${OUT_ROOT}/v3_artifacts" \
  --out-dir "${OUT_ROOT}/v3_results" 2>&1 | tail -25

echo "=== [2/3] gate v2 (delta only) — B 组"
python fit_lr_gate_pooled.py \
  --per-j-layers "${LAYERS}" \
  --hs-root "${SCRIPT_DIR}/../hidden_states/pilot_multilayer" \
  --train-split-name train_pilot \
  --artifacts-dir "${OUT_ROOT}/v2_artifacts" \
  --out-dir "${OUT_ROOT}/v2_results" 2>&1 | tail -25

echo "=== [3/3] h-only — C 组"
python gate_h_only/fit_lr_gate_pooled_h_only.py \
  --per-j-layers "${LAYERS}" \
  --artifacts-dir "${OUT_ROOT}/h_only_artifacts" \
  --out-dir "${OUT_ROOT}/h_only_results" 2>&1 | tail -25

echo
echo "=== 三个 artifact 的 per-j 层与阈值"
for d in v3 v2 h_only; do
  python - "${OUT_ROOT}/${d}_artifacts/meta.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
name = sys.argv[1].split("/")[-2]
print(f"{name:<18} layers={m.get('per_j_layers')} "
      f"thresholds={[round(r['threshold'], 4) for r in m['models']]} "
      f"train_auc={[r.get('train_auc') for r in m['models']]}")
PY
done
echo "完成。artifact 在 ${OUT_ROOT}/"
