#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OUT_DIR="outputs/bart_decomposer_musique_2wiki_repro"
V1_DIR="$OUT_DIR/v1_predictions"
NL_DIR="$OUT_DIR/nl_predictions"
V1_SHARD_DIR="$V1_DIR/shards"
NL_SHARD_DIR="$NL_DIR/shards"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
BACKEND="${BACKEND:-transformers}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DTYPE="${DTYPE:-bfloat16}"
NUM_SHARDS="${NUM_SHARDS:-2}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
VERIFY_ARGS=()

if [[ "${VERIFY:-1}" == "0" ]]; then
  VERIFY_ARGS=("--no-verify")
fi

if [[ "$NUM_SHARDS" != "1" && "$NUM_SHARDS" != "2" ]]; then
  echo "NUM_SHARDS must be 1 or 2, got: $NUM_SHARDS" >&2
  exit 1
fi

mkdir -p "$V1_SHARD_DIR" "$NL_SHARD_DIR"

COMMON_ARGS=(
  --backend "$BACKEND"
  --model "$MODEL"
  --dtype "$DTYPE"
  --batch-size "$BATCH_SIZE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  "${VERIFY_ARGS[@]}"
)

if [[ "$BACKEND" == "vllm" ]]; then
  COMMON_ARGS+=(
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
  )
else
  COMMON_ARGS+=(
    --device-map "$DEVICE_MAP"
    --attn-implementation "$ATTN_IMPLEMENTATION"
  )
fi

split_jsonl() {
  local input="$1"
  local shard0="$2"
  local shard1="$3"
  python - "$input" "$shard0" "$shard1" <<'PY'
import sys
from pathlib import Path

input_path, shard0_path, shard1_path = map(Path, sys.argv[1:])
lines = [line for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
shard0 = lines[0::2]
shard1 = lines[1::2]
shard0_path.write_text("\n".join(shard0) + ("\n" if shard0 else ""), encoding="utf-8")
shard1_path.write_text("\n".join(shard1) + ("\n" if shard1 else ""), encoding="utf-8")
print(f"round-robin split {input_path}: {len(shard0)} + {len(shard1)} rows")
PY
}

merge_shards() {
  local output="$1"
  local shard0="$2"
  local shard1="$3"
  python - "$output" "$shard0" "$shard1" <<'PY'
import sys
from pathlib import Path

output_path, shard0_path, shard1_path = map(Path, sys.argv[1:])
shard0 = [line for line in shard0_path.read_text(encoding="utf-8").splitlines() if line.strip()]
shard1 = [line for line in shard1_path.read_text(encoding="utf-8").splitlines() if line.strip()]
merged: list[str] = []
for index in range(max(len(shard0), len(shard1))):
    if index < len(shard0):
        merged.append(shard0[index])
    if index < len(shard1):
        merged.append(shard1[index])
output_path.write_text("\n".join(merged) + ("\n" if merged else ""), encoding="utf-8")
print(f"merged {output_path}: {len(merged)} rows")
PY
}

run_dataset() {
  local name="$1"
  local input="$V1_DIR/predict_${name}_dev_v1.jsonl"
  local output="$NL_DIR/predict_${name}_dev_nl.jsonl"
  local shard0="$V1_SHARD_DIR/predict_${name}_dev_v1.shard0.jsonl"
  local shard1="$V1_SHARD_DIR/predict_${name}_dev_v1.shard1.jsonl"
  local out0="$NL_SHARD_DIR/predict_${name}_dev_nl.shard0.jsonl"
  local out1="$NL_SHARD_DIR/predict_${name}_dev_nl.shard1.jsonl"

  if [[ "$NUM_SHARDS" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU0" python decompose_to_nl.py \
      --input "$input" \
      --output "$output" \
      "${COMMON_ARGS[@]}"
    return
  fi

  split_jsonl "$input" "$shard0" "$shard1"

  CUDA_VISIBLE_DEVICES="$GPU0" python decompose_to_nl.py \
    --input "$shard0" \
    --output "$out0" \
    "${COMMON_ARGS[@]}" &
  local pid0=$!

  CUDA_VISIBLE_DEVICES="$GPU1" python decompose_to_nl.py \
    --input "$shard1" \
    --output "$out1" \
    "${COMMON_ARGS[@]}" &
  local pid1=$!

  wait "$pid0"
  wait "$pid1"

  merge_shards "$output" "$out0" "$out1"
}

run_dataset musique
run_dataset 2wiki
run_dataset hotpot
