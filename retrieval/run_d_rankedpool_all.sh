#!/usr/bin/env bash
# Full-dev D-group runs (gate v3, λ=0.60, production BART decomposer) on all three datasets, with the
# per-hop reranked candidate pool logged (`hop_ranked_para_ids` in the case log) so precision@3 can
# be computed against GRITHopper / ChainRAG. Config is identical to the 0831 report's D-group row;
# afterwards runs compute_evidence_precision.py on the new runs.
#
# Datasets already finished (a summary json exists for the run tag) are skipped, so reruns resume.
#
# Usage:
#   ./run_d_rankedpool_all.sh
#   GPU_A=1 GPU_B=2 UTIL_A=0.25 UTIL_B=0.3 ./run_d_rankedpool_all.sh
#   DATASETS="musique" ./run_d_rankedpool_all.sh
#   GPU_A=2 GPU_B=2 UTIL_A=0.3 SEQUENTIAL=1 ./run_d_rankedpool_all.sh   # all on one GPU, one at a time
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

GPU_A="${GPU_A:-1}"        # musique + hotpot
GPU_B="${GPU_B:-$GPU_A}"   # 2wiki; default: same GPU as the others
UTIL_A="${UTIL_A:-0.3}"    # same as the 0831 D-group runs
UTIL_B="${UTIL_B:-0.3}"
STAGGER_SEC="${STAGGER_SEC:-90}"
SEQUENTIAL="${SEQUENTIAL:-1}"   # default: datasets one after another; 0 = all at once
read -r -a DATASETS <<< "${DATASETS:-musique hotpot 2wiki}"
TAG_SUFFIX="lambda060_dev_rankedpool"

mkdir -p logs
pids=()
for ds in "${DATASETS[@]}"; do
  tag="gate_v3_rawprefix_${ds}_${TAG_SUFFIX}"
  if compgen -G "results/*_${tag}/retrieval_exp_*.json" > /dev/null; then
    echo "[skip] $ds already done ($tag)"
    continue
  fi
  if [[ "$ds" == "2wiki" ]]; then gpu="$GPU_B"; util="$UTIL_B"; else gpu="$GPU_A"; util="$UTIL_A"; fi
  echo "[launch] $ds on GPU $gpu (util $util) -> logs/${tag}.log"
  CUDA_VISIBLE_DEVICES="$gpu" python run_retrieval_exp_wavefront_gate_v3_rawprefix.py \
    --dataset "$ds" --decompose-mode bart_decompose \
    --decompose-file "../data/decompose/${ds}/bart/dev_nl.jsonl" \
    --artifacts-dir ../gate/gate_v3/artifacts_pooled_v3 \
    --lambda-gate 0.60 --beam-width 1 --retrieve-k 10 --retriever cosine \
    --final-reader-prompt comparison_hint --short-answer-prompt default \
    --limit 0 --gpu-memory-utilization "$util" --max-model-len 8192 --gate-device cuda:0 \
    --run-tag "$tag" > "logs/${tag}.log" 2>&1 &
  pids+=($!)
  if [[ "$SEQUENTIAL" == "1" ]]; then
    wait "${pids[-1]}" || { echo "[error] $ds failed -- see logs/${tag}.log"; exit 1; }
    echo "[done] $ds"
  else
    sleep "$STAGGER_SEC"  # stagger vLLM startups so engines sharing a GPU don't profile memory at once
  fi
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [[ $fail != 0 ]]; then
  echo "[error] at least one run failed -- check logs/gate_v3_rawprefix_*_${TAG_SUFFIX}.log, then rerun (finished datasets are skipped)"
  exit 1
fi

d_runs=()
for ds in musique 2wiki hotpot; do
  run=$(ls -d results/*_gate_v3_rawprefix_${ds}_${TAG_SUFFIX} 2>/dev/null | tail -1 || true)
  [[ -n "$run" ]] && d_runs+=("${ds}=$(realpath "$run")")
done
echo "[precision] ${d_runs[*]}"
python compute_evidence_precision.py --d-runs "${d_runs[@]}" \
  --out results/evidence_precision_3way_rankedpool.json
