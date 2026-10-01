#!/usr/bin/env bash
# Ablation: end-to-end D-group (gate v3, λ=0.60) with the MuSiQue-only BART decomposer
# (outputs/bart_decomposer_old/checkpoint-936) instead of the production MuSiQue+2Wiki one.
# Everything except --decompose-file matches the 0831 report's D-group row (0828
# *_lambda060_dev runs), which serves as the baseline -- it is NOT rerun.
#
# Stages (each skips datasets whose output already exists, so reruns resume):
#   decompose : BART predict -> v1 -> Llama NL sub-questions        (GPU_A)
#   e2e       : run_retrieval_exp_wavefront_gate_v3_rawprefix.py    (musique+hotpot on GPU_A, 2wiki on GPU_B)
#   compare   : table vs the 0828 baseline runs                     (CPU)
#
# Usage:
#   ./run_musique_only_decomposer_e2e.sh                       # all stages
#   STAGES="decompose" ./run_musique_only_decomposer_e2e.sh     # e.g. in the BART/transformers env
#   STAGES="e2e compare" ./run_musique_only_decomposer_e2e.sh   # e.g. in the vLLM env
#   GPU_A=1 GPU_B=2 ./run_musique_only_decomposer_e2e.sh
#   GPU_A=1 GPU_B=1 SEQUENTIAL=1 ./run_musique_only_decomposer_e2e.sh   # everything on one GPU, e2e one at a time
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

STAGES="${STAGES:-decompose e2e compare}"
GPU_A="${GPU_A:-1}"
GPU_B="${GPU_B:-$GPU_A}"   # default: everything on GPU_A
SEQUENTIAL="${SEQUENTIAL:-1}"   # default: e2e datasets one after another; 0 = all at once
CKPT="outputs/bart_decomposer_old/checkpoint-936"
OUT="outputs/bart_decomposer_old"
RAW="${PROJECT_ROOT}/data/raw"
TAG_SUFFIX="musiqueonly_decomposer"

declare -A DEV_FILE=(
  # MuSiQue goes through predict.py's BART-format reader (needs composed_question_text), not the
  # raw v1.0 file -- same input the production decomposer's predict_musique_dev.jsonl was made from
  [musique]="${SCRIPT_DIR}/data/musique_raw/musique_ans_gold_context_version_dev.jsonl"
  [2wiki]="${RAW}/2wikimultihopqa/dev.json"
  [hotpot]="${RAW}/hotpotqa/hotpot_dev_distractor_v1.json"
)
DATASETS=(musique 2wiki hotpot)

has_stage() { [[ " ${STAGES} " == *" $1 "* ]]; }

if has_stage decompose; then
  mkdir -p "$OUT/raw_predictions" "$OUT/v1_predictions" "$OUT/nl_predictions"
  for ds in "${DATASETS[@]}"; do
    raw="$OUT/raw_predictions/predict_${ds}_dev.jsonl"
    v1="$OUT/v1_predictions/predict_${ds}_dev_v1.jsonl"
    nl="$OUT/nl_predictions/predict_${ds}_dev_nl.jsonl"
    # "done" = the .summary.json each tool writes only after finishing the whole file; a partial
    # output (decompose_to_nl resumes from its .checkpoint.json) must not be mistaken for done
    if [[ ! -s "${raw%.jsonl}.summary.json" ]]; then
      echo "[decompose] BART predict: $ds"
      CUDA_VISIBLE_DEVICES="$GPU_A" python predict.py --model-dir "$CKPT" --dataset "$ds" \
        --dev-file "${DEV_FILE[$ds]}" --output "$raw" --fp16 --batch-size 16 --num-beams 10
    fi
    if [[ ! -s "$v1" ]]; then
      echo "[decompose] convert to v1: $ds"
      extra=(); [[ "$ds" == "musique" ]] && extra=(--include-gold)
      python convert_predictions_to_v1.py --input "$raw" --output "$v1" "${extra[@]}"
    fi
    if [[ ! -s "${nl%.jsonl}.summary.json" ]]; then
      echo "[decompose] Llama NL rewrite: $ds"
      if [[ -f "${nl%.jsonl}.checkpoint.json" && -f "$nl" ]]; then
        # resuming after an interrupt: decompose_to_nl appends a record and only then rewrites its
        # checkpoint, so a kill in between can leave a truncated last line or a record the
        # checkpoint doesn't know about (it would be redone and duplicated). Keep only parseable
        # lines whose id is in the checkpoint, first occurrence.
        python - "$nl" "${nl%.jsonl}.checkpoint.json" <<'PY'
import json, os, sys
nl, ck = sys.argv[1], sys.argv[2]
done = set(json.load(open(ck, encoding="utf-8")))
kept, seen, dropped = [], set(), 0
for line in open(nl, encoding="utf-8"):
    if not line.strip():
        continue
    try:
        rid = str(json.loads(line)["id"])
    except (json.JSONDecodeError, KeyError):
        dropped += 1
        continue
    if rid in done and rid not in seen:
        seen.add(rid)
        kept.append(line.rstrip("\n"))
    else:
        dropped += 1
tmp = nl + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    f.write("\n".join(kept) + ("\n" if kept else ""))
os.replace(tmp, nl)
with open(ck, "w", encoding="utf-8") as f:
    json.dump(sorted(seen), f, ensure_ascii=False)
print(f"[decompose] resume cleanup: kept {len(kept)} records, dropped {dropped}")
PY
      fi
      # same settings as run_decompose_to_nl_all.sh defaults (what produced the production files)
      CUDA_VISIBLE_DEVICES="$GPU_A" python decompose_to_nl.py --input "$v1" --output "$nl" \
        --backend transformers --model meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 \
        --batch-size 4 --max-new-tokens 512 --device-map cuda:0 --attn-implementation sdpa
    fi
  done
fi

if has_stage e2e; then
  cd "${PROJECT_ROOT}/retrieval"
  mkdir -p logs
  pids=()
  for ds in "${DATASETS[@]}"; do
    tag="gate_v3_rawprefix_${ds}_lambda060_dev_${TAG_SUFFIX}"
    if compgen -G "results/*_${tag}/retrieval_exp_*.json" > /dev/null; then
      echo "[e2e] $ds already done ($tag), skipping"
      continue
    fi
    gpu="$GPU_A"; [[ "$ds" == "2wiki" ]] && gpu="$GPU_B"
    echo "[e2e] launching $ds on GPU $gpu -> logs/${tag}.log"
    CUDA_VISIBLE_DEVICES="$gpu" python run_retrieval_exp_wavefront_gate_v3_rawprefix.py \
      --dataset "$ds" --decompose-mode bart_decompose \
      --decompose-file "${SCRIPT_DIR}/${OUT}/nl_predictions/predict_${ds}_dev_nl.jsonl" \
      --artifacts-dir ../gate/gate_v3/artifacts_pooled_v3 \
      --lambda-gate 0.60 --beam-width 1 --retrieve-k 10 --retriever cosine \
      --final-reader-prompt comparison_hint --short-answer-prompt default \
      --limit 0 --gpu-memory-utilization 0.3 --max-model-len 8192 --gate-device cuda:0 \
      --run-tag "$tag" > "logs/${tag}.log" 2>&1 &
    pids+=($!)
    if [[ "$SEQUENTIAL" == "1" ]]; then
      wait "${pids[-1]}" || { echo "[e2e] $ds failed -- see retrieval/logs/${tag}.log"; exit 1; }
      echo "[e2e] $ds done"
    else
      sleep 90  # stagger vLLM startups so two engines on GPU_A don't profile memory at the same time
    fi
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [[ $fail == 0 ]] || { echo "[e2e] at least one run failed -- see retrieval/logs/"; exit 1; }
  cd "${SCRIPT_DIR}"
fi

if has_stage compare; then
  python "${PROJECT_ROOT}/retrieval/compare_decomposer_ablation.py" --tag-suffix "$TAG_SUFFIX"
fi
