#!/usr/bin/env bash
# Re-run the whole end-to-end experiment matrix against a gate retrained with a UNIFORM layer 15
# (the layer policy that wins when layers are selected on a train-internal held-out split rather
# than on dev -- see gate/select_layers_train_holdout.py).
#
# Phases, in dependency order. Everything after `lambda` needs LAMBDA to be set to the value that
# sweep picked, so run `lambda` first, read its table, then export LAMBDA=<value>.
#
#   lambda      lambda sweep on the MuSiQue TRAIN subset (dev never used for tuning)   ~3.4h
#   main        D-group: 3 datasets, dev full, at $LAMBDA                              ~3.1h
#   bc          B (delta-only) and C (h-only) on 3 datasets at $LAMBDA                 ~6.9h
#   decomposer  MuSiQue-only-decomposer ablation on 3 datasets                          ~2.6h
#   global10    global setting, k=10, 3 datasets                                        ~5.6h
#   colbert     ColBERT retriever + gate, 3 datasets                                    ~5.4h
#   global50    global setting, k=50, 3 datasets                                       ~15.7h
#
# Not re-run on purpose (results are independent of the gate artifact):
#   - the lambda=0 no-gate baseline: gate_score is multiplied by 0
#   - ColBERT without gate (--lambda-gate 0.0)
#   - the reader-prompt ablation: it compares prompt vs prompt with the gate held fixed
#   - ChainRAG / GRITHopper: different systems entirely
#
# Each dataset is skipped if a finished run with the same tag already exists, so re-running the
# same command resumes. One phase per tmux window; phases after `lambda` are independent of each
# other and can run concurrently on different GPUs.
#
# Usage:
#   GPU=1 ./run_layer15_pipeline.sh lambda
#   GPU=1 LAMBDA=0.60 ./run_layer15_pipeline.sh main
#   GPU=2 LAMBDA=0.60 ./run_layer15_pipeline.sh global10 colbert
#   GPU=1 GATE_GPU=3 LAMBDA=0.60 ./run_layer15_pipeline.sh main      # vLLM on GPU1, gate model on GPU3
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

GPU="${GPU:-0}"
GATE_GPU="${GATE_GPU:-}"          # empty = gate model shares GPU with vLLM
UTIL="${UTIL:-0.3}"
SCRIPT_DEFAULT="run_retrieval_exp_wavefront_gate_v3_rawprefix.py"
SCRIPT="${SCRIPT_DEFAULT}"   # B/C 两组要换成各自的打分脚本，见 bc phase
HB="${HB:-8}"                      # --hidden-batch-size；卡紧时调小（gate 预热按 HB×4096 token 占显存）
TAG="${TAG:-l15}"
ART_ROOT="${ART_ROOT:-${SCRIPT_DIR}/../gate/layer15_uniform}"
ART_V3="${ART_V3:-${ART_ROOT}/v3_artifacts}"
ART_V2="${ART_V2:-${ART_ROOT}/v2_artifacts}"
ART_HONLY="${ART_HONLY:-${ART_ROOT}/h_only_artifacts}"
LAMBDA="${LAMBDA:-}"
LAMBDA_LIST="${LAMBDA_LIST:-0.30 0.40 0.50 0.60 0.70 0.80}"
read -r -a DATASETS <<< "${DATASETS:-musique hotpot 2wiki}"
TRAIN_SUBSET="${TRAIN_SUBSET:-../decompose/bart/data/musique_train_lambda_subset_big_bart_nl.jsonl}"
mkdir -p logs

if [[ "$#" -eq 0 ]]; then
  echo "用法: GPU=<n> [LAMBDA=<x>] $0 <phase> [phase...]   可选 phase: lambda main bc decomposer global10 colbert global50" >&2
  exit 1
fi

for a in "${ART_V3}" "${ART_V2}" "${ART_HONLY}"; do
  [[ -f "$a/meta.json" ]] || { echo "缺少 artifact: $a/meta.json —— 先跑 gate/refit_all_layer15_uniform.sh" >&2; exit 1; }
done

if [[ -n "${GATE_GPU}" && "${GATE_GPU}" != "${GPU}" ]]; then
  VISIBLE="${GPU},${GATE_GPU}"; GATE_DEV="cuda:1"
else
  VISIBLE="${GPU}"; GATE_DEV="cuda:0"
fi

done_already() { compgen -G "results/*_$1/retrieval_exp_*.json" > /dev/null; }

# run <run_tag> <extra args...>
run() {
  local tag="$1"; shift
  if done_already "$tag"; then echo "[skip] $tag 已完成"; return 0; fi
  echo "[run ] $tag  (${SCRIPT}, GPU ${VISIBLE}, gate ${GATE_DEV}, util ${UTIL}, hb ${HB})  -> logs/${tag}.log"
  local t0=$SECONDS
  CUDA_VISIBLE_DEVICES="${VISIBLE}" VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}" \
    python "${SCRIPT}" \
    --beam-width 1 --limit 0 \
    --gpu-memory-utilization "${UTIL}" --max-model-len 8192 --gate-device "${GATE_DEV}" \
    --hidden-batch-size "${HB}" \
    --short-answer-prompt default \
    --run-tag "$tag" "$@" > "logs/${tag}.log" 2>&1
  echo "[done] $tag  用时 $(( (SECONDS-t0)/60 )) 分钟"
}

need_lambda() { [[ -n "${LAMBDA}" ]] || { echo "该 phase 需要 LAMBDA=<扫描选出的值>" >&2; exit 1; }; }

for phase in "$@"; do
case "$phase" in

lambda)   # λ 扫描：MuSiQue train 子集，dev 不参与
  for lam in ${LAMBDA_LIST}; do
    run "${TAG}_trainsubset_lambda${lam/./p}" \
      --dataset musique --split train --decompose-mode bart_decompose \
      --decompose-file "${TRAIN_SUBSET}" --artifacts-dir "${ART_V3}" \
      --lambda-gate "$lam" --retrieve-k 10 --retriever cosine --final-reader-prompt comparison_hint
  done
  echo; echo "=== λ 扫描结果（按 EM 排序）"
  python - "${TAG}" <<'PY'
import glob, json, sys
tag = sys.argv[1]
rows = []
for d in sorted(glob.glob(f"results/*_{tag}_trainsubset_lambda*")):
    f = glob.glob(d + "/retrieval_exp_*.json")
    if not f: continue
    s = json.load(open(f[0])); c = s["config"]; p = s["full_pool_gold_rank_overall"]
    rows.append((c["lambda_gate"], p["recall@1"], s["final_beam_chain_match_rate_overall"],
                 s["answer_overall"]["answer_em"], s["answer_overall"]["answer_f1"]))
print(f"{'λ':>6}{'recall@1':>10}{'chain@1':>9}{'EM':>8}{'F1':>8}")
for r in sorted(rows):
    print(f"{r[0]:>6}{r[1]:>10}{r[2]:>9}{r[3]:>8}{r[4]:>8}")
if rows:
    best = max(rows, key=lambda r: r[3])
    print(f"\nEM 最高: λ={best[0]}  →  后续 phase 用 LAMBDA={best[0]}")
PY
  ;;

main)     # D 组主结果（这些 run 自带每跳候选池记录，precision@3 分析可直接用）
  need_lambda
  for ds in "${DATASETS[@]}"; do
    run "${TAG}_${ds}_lambda${LAMBDA/./p}_dev" \
      --dataset "$ds" --decompose-mode bart_decompose \
      --decompose-file "../data/decompose/${ds}/bart/dev_nl.jsonl" --artifacts-dir "${ART_V3}" \
      --lambda-gate "${LAMBDA}" --retrieve-k 10 --retriever cosine --final-reader-prompt comparison_hint
  done ;;

bc)       # 三信号对照的 B（仅 Δ）与 C（仅 h）；D 由 main phase 提供
  # 每组必须用自己的打分脚本：v3 的 score_gate_v3 要求 artifact 里有 pca_h + pca_delta，
  # 而 v2 artifact 只有 pca、h_only artifact 只有 pca_h —— 用 v3 脚本跑会 KeyError。
  # 这两个脚本不支持 --retriever / --retrieval-scope（它们只有 cosine + distractor 这一条路径）。
  need_lambda
  for ds in "${DATASETS[@]}"; do
    SCRIPT="run_retrieval_exp_wavefront_gate_v2_rawprefix.py"
    run "${TAG}_B_deltaonly_${ds}_dev" \
      --dataset "$ds" --decompose-mode bart_decompose \
      --decompose-file "../data/decompose/${ds}/bart/dev_nl.jsonl" --artifacts-dir "${ART_V2}" \
      --lambda-gate "${LAMBDA}" --retrieve-k 10 --final-reader-prompt comparison_hint
    SCRIPT="run_retrieval_exp_wavefront_gate_h_only_rawprefix.py"
    run "${TAG}_C_honly_${ds}_dev" \
      --dataset "$ds" --decompose-mode bart_decompose \
      --decompose-file "../data/decompose/${ds}/bart/dev_nl.jsonl" --artifacts-dir "${ART_HONLY}" \
      --lambda-gate "${LAMBDA}" --retrieve-k 10 --final-reader-prompt comparison_hint
  done
  SCRIPT="${SCRIPT_DEFAULT}" ;;

decomposer)  # 只用 MuSiQue 训练的分解器
  need_lambda
  for ds in "${DATASETS[@]}"; do
    run "${TAG}_${ds}_musiqueonly_decomposer" \
      --dataset "$ds" --decompose-mode bart_decompose \
      --decompose-file "../decompose/bart/outputs/bart_decomposer_old/nl_predictions/predict_${ds}_dev_nl.jsonl" \
      --artifacts-dir "${ART_V3}" \
      --lambda-gate "${LAMBDA}" --retrieve-k 10 --retriever cosine --final-reader-prompt comparison_hint
  done ;;

global10|global50)
  need_lambda
  k=10; [[ "$phase" == "global50" ]] && k=50
  for ds in "${DATASETS[@]}"; do
    run "${TAG}_global_${ds}_k${k}" \
      --dataset "$ds" --decompose-mode bart_decompose \
      --decompose-file "../data/decompose/${ds}/bart/dev_nl.jsonl" --artifacts-dir "${ART_V3}" \
      --retrieval-scope global --corpus-dir "global_corpus/${ds}_dev" \
      --lambda-gate "${LAMBDA}" --retrieve-k "$k" --retriever cosine --final-reader-prompt comparison_hint
  done ;;

colbert)  # 可迁移性：换检索器，gate 不变（无 gate 那三组与 artifact 无关，沿用旧 run）
  need_lambda
  for ds in "${DATASETS[@]}"; do
    run "${TAG}_colbert_${ds}_gate" \
      --dataset "$ds" --decompose-mode bart_decompose \
      --decompose-file "../data/decompose/${ds}/bart/dev_nl.jsonl" --artifacts-dir "${ART_V3}" \
      --retriever colbert --lambda-gate "${LAMBDA}" --retrieve-k 10 \
      --final-reader-prompt comparison_hint
  done ;;

*) echo "未知 phase: $phase" >&2; exit 1 ;;
esac
done
echo "全部指定的 phase 完成。"
