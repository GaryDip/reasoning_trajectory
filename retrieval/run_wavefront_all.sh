#!/usr/bin/env bash
# End-to-end retrieval: BART decompose + BGE + pooled LR gate + vLLM wavefront.
#
# Usage:
#   GPUS=0 ./run_wavefront_all.sh
#   GPUS=0 DATASETS="musique 2wiki hotpot" METHODS=gated_rule_a ./run_wavefront_all.sh
#   GPUS=0 LIMIT=20 DATASETS=musique ./run_wavefront_all.sh
#   GPUS="0 1 2" LAMBDA_LR=0.25 ./run_wavefront_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python3}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
GPUS="${GPUS:-0}"
read -r -a GPU_LIST <<< "${GPUS}"
NUM_SHARDS="${#GPU_LIST[@]}"
if [[ "${NUM_SHARDS}" -lt 1 ]]; then
  echo "GPUS must contain at least one GPU id" >&2
  exit 1
fi

DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
SPLIT="${SPLIT:-dev}"
LIMIT="${LIMIT:-0}"
TOPK="${TOPK:-10}"
EXPAND_TOPK="${EXPAND_TOPK:-10}"
LAMBDA_LR="${LAMBDA_LR:-0.25}"
COS_MODEL="${COS_MODEL:-BAAI/bge-base-en-v1.5}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GATE_DEVICE="${GATE_DEVICE:-cuda:0}"
HIDDEN_BATCH_SIZE="${HIDDEN_BATCH_SIZE:-8}"
DECOMPOSE_MODE="${DECOMPOSE_MODE:-bart_decompose}"
GATE_ARTIFACTS="${GATE_ARTIFACTS:-${PROJECT_ROOT}/gate/artifacts_pooled}"
DECOMPOSE_ROOT="${DECOMPOSE_ROOT:-${PROJECT_ROOT}/data/decompose}"
RAW_ROOT="${RAW_ROOT:-${PROJECT_ROOT}/data/raw}"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRIPT_DIR}/results}"
DATASETS=(${DATASETS:-musique 2wiki hotpot})
METHODS=(${METHODS:-gated_rule_a})

MUSIQUE_DIR="${MUSIQUE_DIR:-${RAW_ROOT}/musique}"
TWOWIKI_FILE="${TWOWIKI_FILE:-${RAW_ROOT}/2wikimultihopqa/dev.json}"
HOTPOT_FILE="${HOTPOT_FILE:-${RAW_ROOT}/hotpotqa/hotpot_dev_distractor_v1.json}"

WAVEFRONT="${SCRIPT_DIR}/run_retrieval_exp_wavefront.py"
if [[ ! -f "${WAVEFRONT}" ]]; then
  echo "ERROR: missing ${WAVEFRONT}" >&2
  exit 1
fi
if [[ ! -f "${GATE_ARTIFACTS}/meta.json" ]]; then
  echo "ERROR: gate artifacts not found at ${GATE_ARTIFACTS}" >&2
  echo "Run: cd ${PROJECT_ROOT}/gate && ./run_fit_lr_gate_pooled.sh" >&2
  exit 1
fi

LAMBDA_TAG="$(printf '%s' "${LAMBDA_LR}" | tr '.' 'p' | tr '-' 'm')"
SESSION_NAME="${SESSION_NAME:-wavefront_pooled_lam${LAMBDA_TAG}}"
SESSION="${SESSION:-${RESULTS_ROOT}/$(date +%Y%m%d_%H%M%S)_${SESSION_NAME}}"
LOG_DIR="${SESSION}/logs"
mkdir -p "${SESSION}" "${LOG_DIR}"

LIMIT_ARGS=()
if [[ "${LIMIT}" -gt 0 ]]; then
  LIMIT_ARGS+=(--limit "${LIMIT}")
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

resolve_decompose_file() {
  local dataset="$1"
  if [[ "${DECOMPOSE_MODE}" == "bart_decompose" ]]; then
    echo "${DECOMPOSE_ROOT}/${dataset}/bart/dev_nl.jsonl"
  elif [[ "${DECOMPOSE_MODE}" == "gt" ]]; then
    if [[ "${dataset}" != "musique" ]]; then
      echo "ERROR: GT decompose only available for musique" >&2
      exit 1
    fi
    echo "${DECOMPOSE_ROOT}/musique/gt/${SPLIT}_nl.jsonl"
  else
    echo "ERROR: unknown DECOMPOSE_MODE=${DECOMPOSE_MODE}" >&2
    exit 1
  fi
}

export SESSION SPLIT MODEL GPUS NUM_SHARDS DTYPE ATTN_IMPLEMENTATION LIMIT TOPK EXPAND_TOPK
export LAMBDA_LR COS_MODEL GPU_MEMORY_UTILIZATION MAX_MODEL_LEN GATE_DEVICE HIDDEN_BATCH_SIZE
export METHODS_STR="${METHODS[*]}"
export DATASETS_STR="${DATASETS[*]}"
export DECOMPOSE_MODE GATE_ARTIFACTS PROJECT_ROOT

"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

session = Path(os.environ["SESSION"])
datasets = os.environ["DATASETS_STR"].split()
num_shards = int(os.environ["NUM_SHARDS"])
runs = []
for dataset in datasets:
    for shard in range(num_shards):
        out_dir = f"{dataset}/shard{shard}" if num_shards > 1 else dataset
        runs.append({
            "dataset": dataset,
            "shard_index": shard,
            "num_shards": num_shards,
            "dir": out_dir,
        })

manifest = {
    "session": str(session),
    "project_root": os.environ["PROJECT_ROOT"],
    "script": "run_retrieval_exp_wavefront.py",
    "split": os.environ["SPLIT"],
    "decompose_mode": os.environ["DECOMPOSE_MODE"],
    "gate_artifacts": os.environ["GATE_ARTIFACTS"],
    "gate_artifact_mode": "pooled",
    "methods": os.environ["METHODS_STR"].split(),
    "model": os.environ["MODEL"],
    "devices": os.environ["GPUS"].split(),
    "num_shards": num_shards,
    "dtype": os.environ["DTYPE"],
    "attn_implementation": os.environ["ATTN_IMPLEMENTATION"],
    "limit": int(os.environ["LIMIT"]),
    "lambda_lr": float(os.environ["LAMBDA_LR"]),
    "topk": int(os.environ["TOPK"]),
    "expand_topk": int(os.environ["EXPAND_TOPK"]),
    "cos_model": os.environ["COS_MODEL"],
    "gpu_memory_utilization": float(os.environ["GPU_MEMORY_UTILIZATION"]),
    "max_model_len": int(os.environ["MAX_MODEL_LEN"]),
    "gate_device": os.environ["GATE_DEVICE"],
    "hidden_batch_size": int(os.environ["HIDDEN_BATCH_SIZE"]),
    "runs": runs,
}
(session / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Manifest: {session / 'manifest.json'} ({len(runs)} runs)")
PY

echo "=== multihop_trace wavefront retrieval ==="
echo "Session        : ${SESSION}"
echo "Split          : ${SPLIT}"
echo "Datasets       : ${DATASETS[*]}"
echo "Methods        : ${METHODS[*]}"
echo "Decompose mode : ${DECOMPOSE_MODE}"
echo "Gate artifacts : ${GATE_ARTIFACTS}"
echo "Devices        : ${GPUS} (${NUM_SHARDS} shard(s))"
echo "Limit          : ${LIMIT} (0 = full split)"
echo "Lambda LR      : ${LAMBDA_LR}"
echo

run_shard() {
  local dataset="$1"
  local shard_index="$2"
  local gpu="$3"
  local decompose_file
  local out_dir
  local run_tag
  local log_file

  decompose_file="$(resolve_decompose_file "${dataset}")"
  if [[ ! -f "${decompose_file}" ]]; then
    echo "ERROR: missing decompose file: ${decompose_file}" >&2
    exit 1
  fi

  if [[ "${NUM_SHARDS}" -gt 1 ]]; then
    out_dir="${SESSION}/${dataset}/shard${shard_index}"
    run_tag="${dataset}_wavefront_${LAMBDA_TAG}_s${shard_index}"
    log_file="${LOG_DIR}/${dataset}_shard${shard_index}.log"
  else
    out_dir="${SESSION}/${dataset}"
    run_tag="${dataset}_wavefront_${LAMBDA_TAG}"
    log_file="${LOG_DIR}/${dataset}.log"
  fi

  mkdir -p "${out_dir}"

  echo ">>> [${dataset}] gpu=${gpu} shard=${shard_index}/${NUM_SHARDS}"
  echo "    decompose: ${decompose_file}"
  echo "    out: ${out_dir}"
  echo "    log: ${log_file}"

  (
    set -x
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${WAVEFRONT}" \
      --split "${SPLIT}" \
      --dataset "${dataset}" \
      --musique-dir "${MUSIQUE_DIR}" \
      --twowiki-file "${TWOWIKI_FILE}" \
      --hotpot-file "${HOTPOT_FILE}" \
      --decompose-mode "${DECOMPOSE_MODE}" \
      --decompose-file "${decompose_file}" \
      --gate-artifact-mode pooled \
      --artifacts-dir "${GATE_ARTIFACTS}" \
      --out-dir "${out_dir}" \
      --methods "${METHODS[@]}" \
      --topk "${TOPK}" \
      --expand-topk "${EXPAND_TOPK}" \
      --lambda-lr "${LAMBDA_LR}" \
      --cos-model "${COS_MODEL}" \
      --model "${MODEL}" \
      --dtype "${DTYPE}" \
      --attn-implementation "${ATTN_IMPLEMENTATION}" \
      --gate-device "${GATE_DEVICE}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --hidden-batch-size "${HIDDEN_BATCH_SIZE}" \
      --answerable-only \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard_index}" \
      --run-tag "${run_tag}" \
      "${LIMIT_ARGS[@]}"
  ) 2>&1 | tee "${log_file}"
}

merge_and_eval_dataset() {
  local dataset="$1"
  local dataset_dir="${SESSION}/${dataset}"
  local merged_cases="${dataset_dir}/retrieval_cases_${SPLIT}_${dataset}_wavefront_${LAMBDA_TAG}_merged.jsonl"
  local merged_metrics="${dataset_dir}/retrieval_exp_${SPLIT}_${dataset}_wavefront_${LAMBDA_TAG}_merged.json"

  echo ">>> [${dataset}] merge shards + recompute metrics"
  "${PYTHON}" - "${dataset}" "${dataset_dir}" "${merged_cases}" "${merged_metrics}" <<'PY'
import json
import os
import sys
from pathlib import Path

dataset = sys.argv[1]
dataset_dir = Path(sys.argv[2])
merged_cases = Path(sys.argv[3])
merged_metrics = Path(sys.argv[4])
split = os.environ["SPLIT"]
lambda_tag = os.environ["LAMBDA_TAG"]
num_shards = int(os.environ["NUM_SHARDS"])
methods = os.environ["METHODS_STR"].split()
ranker_methods = [m for m in methods if m != "oracle"]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


shards: list[list[dict]] = []
for shard_index in range(num_shards):
    if num_shards > 1:
        path = (
            dataset_dir
            / f"shard{shard_index}"
            / f"retrieval_cases_{split}_{dataset}_wavefront_{lambda_tag}_s{shard_index}.jsonl"
        )
    else:
        path = (
            dataset_dir
            / f"retrieval_cases_{split}_{dataset}_wavefront_{lambda_tag}.jsonl"
        )
    if not path.exists():
        raise SystemExit(f"Missing cases: {path}")
    shards.append(read_jsonl(path))

merged: list[dict] = []
max_len = max((len(rows) for rows in shards), default=0)
for i in range(max_len):
    for rows in shards:
        if i < len(rows):
            merged.append(rows[i])

seen: set[str] = set()
deduped: list[dict] = []
for row in merged:
    rid = str(row.get("id", ""))
    if rid in seen:
        raise SystemExit(f"Duplicate id after merge: {rid}")
    seen.add(rid)
    deduped.append(row)

if num_shards == 1:
    print(f"Single shard, skip rewrite: {dataset_dir / f'retrieval_cases_{split}_{dataset}_wavefront_{lambda_tag}.jsonl'}")
    sys.exit(0)

merged_cases.parent.mkdir(parents=True, exist_ok=True)
with merged_cases.open("w", encoding="utf-8") as f:
    for row in deduped:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pct(num: float, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def empty_method_stats() -> dict:
    return {
        "hop_total": 0, "r1": 0, "r3": 0, "rr_sum": 0.0, "rank_sum": 0,
        "selection_total": 0, "selection_correct": 0,
        "chain_total": 0, "chain_r1": 0, "chain_r3": 0,
        "case_total": 0, "case_r1": 0, "case_r3": 0,
        "chain_k_match_total": 0, "chain_k_match_r1": 0, "chain_k_match_r3": 0,
        "chain_selection_total": 0, "chain_selection_correct": 0,
        "chain_selection_k_match_total": 0, "chain_selection_k_match_correct": 0,
        "answer_total": 0, "answer_em": 0, "answer_f1": 0.0,
        "gate_hops": 0, "gate_triggered_hops": 0,
        "gate_examples": 0, "gate_triggered_examples": 0,
    }


def result_method(stats: dict) -> dict:
    return {
        "overall": {
            "n_hops": stats["hop_total"],
            "recall@1": pct(stats["r1"], stats["hop_total"]),
            "recall@3": pct(stats["r3"], stats["hop_total"]),
            "mrr": pct(stats["rr_sum"], stats["hop_total"]),
            "avg_rank": round(stats["rank_sum"] / stats["hop_total"], 2) if stats["hop_total"] else None,
        },
        "selection_overall": {
            "n_hops": stats["selection_total"],
            "selection_acc": pct(stats["selection_correct"], stats["selection_total"]),
        },
        "chain_overall": {
            "n_examples": stats["chain_total"],
            "chain_recall@1": pct(stats["chain_r1"], stats["chain_total"]),
            "chain_recall@3": pct(stats["chain_r3"], stats["chain_total"]),
        },
        "case_recall_overall": {
            "n_examples": stats["case_total"],
            "case_recall@1": pct(stats["case_r1"], stats["case_total"]),
            "case_recall@3": pct(stats["case_r3"], stats["case_total"]),
        },
        "chain_overall_k_match": {
            "n_examples": stats["chain_k_match_total"],
            "chain_recall@1": pct(stats["chain_k_match_r1"], stats["chain_k_match_total"]),
            "chain_recall@3": pct(stats["chain_k_match_r3"], stats["chain_k_match_total"]),
        },
        "chain_selection_overall": {
            "n_examples": stats["chain_selection_total"],
            "chain_selection_acc": pct(stats["chain_selection_correct"], stats["chain_selection_total"]),
        },
        "chain_selection_overall_k_match": {
            "n_examples": stats["chain_selection_k_match_total"],
            "chain_selection_acc": pct(
                stats["chain_selection_k_match_correct"],
                stats["chain_selection_k_match_total"],
            ),
        },
        "answer_overall": {
            "n_examples": stats["answer_total"],
            "answer_em": pct(stats["answer_em"], stats["answer_total"]),
            "answer_f1": pct(stats["answer_f1"], stats["answer_total"]),
        },
    }


stats_by_method = {m: empty_method_stats() for m in ranker_methods}
by_k = {m: {k: empty_method_stats() for k in (1, 2, 3, 4)} for m in ranker_methods}

for row in deduped:
    k_pred = int(row.get("K_pred") or 0)
    k_gold = int(row.get("K_gold") or 0)
    k_match = bool(row.get("K_match"))
    hop_results = row.get("hop_results") or []
    for method in ranker_methods:
        targets = [stats_by_method[method]]
        if k_pred in by_k[method]:
            targets.append(by_k[method][k_pred])

        for target in targets:
            chain_r1 = (row.get("chain_recall1") or {}).get(method)
            chain_r3 = (row.get("chain_recall3") or {}).get(method)
            if chain_r1 is not None or chain_r3 is not None:
                target["chain_total"] += 1
                target["chain_r1"] += int(chain_r1 is True)
                target["chain_r3"] += int(chain_r3 is True)

            case_r1 = (row.get("case_recall1") or {}).get(method)
            case_r3 = (row.get("case_recall3") or {}).get(method)
            if case_r1 is not None or case_r3 is not None:
                target["case_total"] += 1
                target["case_r1"] += int(case_r1 is True)
                target["case_r3"] += int(case_r3 is True)

            if k_match:
                km_r1 = (row.get("chain_recall1_k_match") or {}).get(method)
                km_r3 = (row.get("chain_recall3_k_match") or {}).get(method)
                if km_r1 is not None or km_r3 is not None:
                    target["chain_k_match_total"] += 1
                    target["chain_k_match_r1"] += int(km_r1 is True)
                    target["chain_k_match_r3"] += int(km_r3 is True)

            chain_sel = (row.get("chain_selection") or {}).get(method)
            if chain_sel is not None:
                target["chain_selection_total"] += 1
                target["chain_selection_correct"] += int(chain_sel is True)

            chain_sel_km = (row.get("chain_selection_k_match") or {}).get(method)
            if chain_sel_km is not None:
                target["chain_selection_k_match_total"] += 1
                target["chain_selection_k_match_correct"] += int(chain_sel_km is True)

            ans_em = (row.get("answer_em") or {}).get(method)
            ans_f1 = (row.get("answer_f1") or {}).get(method)
            if ans_em is not None or ans_f1 is not None:
                target["answer_total"] += 1
                target["answer_em"] += int(ans_em or 0)
                target["answer_f1"] += float(ans_f1 or 0.0)

            gate_any = (row.get("gate_fired_any") or {}).get(method)
            if gate_any is not None:
                target["gate_examples"] += 1
                target["gate_triggered_examples"] += int(gate_any is True)

        for hop_index, hop in enumerate(hop_results):
            method_row = hop.get(method)
            if not isinstance(method_row, dict):
                continue
            if hop_index >= k_gold:
                continue
            gold_rank = method_row.get("gold_rank")
            sel = (method_row.get("selection") or {}).get("correct")
            for target in targets:
                target["hop_total"] += 1
                if gold_rank is not None:
                    target["r1"] += int(gold_rank == 1)
                    target["r3"] += int(gold_rank <= 3)
                    target["rr_sum"] += 1.0 / gold_rank
                    target["rank_sum"] += int(gold_rank)
                target["selection_total"] += 1
                target["selection_correct"] += int(sel is True)
                gate_fired = method_row.get("gate_fired")
                if gate_fired is not None:
                    target["gate_hops"] += 1
                    target["gate_triggered_hops"] += int(gate_fired is True)

results = {
    "config": {
        "dataset": dataset,
        "split": split,
        "num_shards": num_shards,
        "source": "merged shard retrieval_cases",
        "methods": methods,
    },
    "cases_file": str(merged_cases),
    "n_examples": len(deduped),
    "n_k_match": sum(1 for row in deduped if row.get("K_match")),
}
results["k_match_rate"] = pct(results["n_k_match"], results["n_examples"])

for method, stats in stats_by_method.items():
    method_result = result_method(stats)
    method_result["by_K"] = {
        str(k): result_method(k_stats)["overall"]
        for k, k_stats in by_k[method].items()
    }
    method_result["selection_by_K"] = {
        str(k): result_method(k_stats)["selection_overall"]
        for k, k_stats in by_k[method].items()
    }
    method_result["answer_by_K"] = {
        str(k): result_method(k_stats)["answer_overall"]
        for k, k_stats in by_k[method].items()
    }
    if stats["gate_hops"] or stats["gate_examples"]:
        method_result["gate_stats"] = {
            "n_hops": stats["gate_hops"],
            "n_triggered_hops": stats["gate_triggered_hops"],
            "hop_trigger_rate": pct(stats["gate_triggered_hops"], stats["gate_hops"]),
            "n_examples": stats["gate_examples"],
            "n_examples_any_triggered": stats["gate_triggered_examples"],
            "example_trigger_rate": pct(
                stats["gate_triggered_examples"], stats["gate_examples"]
            ),
        }
    results[method] = method_result

merged_metrics.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Merged cases:   {merged_cases} ({len(deduped)} rows)")
print(f"Merged metrics: {merged_metrics}")
PY
}

run_dataset() {
  local dataset="$1"
  echo "==> dataset=${dataset}"

  if [[ "${NUM_SHARDS}" -gt 1 ]]; then
    local pids=()
    local shard_index
    for shard_index in "${!GPU_LIST[@]}"; do
      run_shard "${dataset}" "${shard_index}" "${GPU_LIST[$shard_index]}" &
      pids+=("$!")
    done
    local pid
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
    merge_and_eval_dataset "${dataset}"
  else
    run_shard "${dataset}" 0 "${GPU_LIST[0]}"
  fi

  echo "<<< dataset=${dataset} done"
  echo
}

export LAMBDA_TAG

for dataset in "${DATASETS[@]}"; do
  run_dataset "${dataset}"
done

echo "All done: ${SESSION}"
