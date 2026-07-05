# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A multi-hop retrieval "gate" pipeline: it trains a small classifier that decides, mid-reasoning, whether
the retrieval for the current hop looks wrong and should trigger an intervention (re-retrieve / expand).
The pipeline runs end-to-end: raw QA datasets → question decomposition → reasoning-trace construction →
LLM hidden-state extraction → LR gate training → retrieval evaluation.

READMEs are in Chinese and are kept up to date — read the one in the directory you're touching before
making changes; this file only covers what you'd otherwise have to piece together across directories.

## Pipeline (5 stages, in dependency order)

| # | Dir | Key script | Input → Output |
|---|-----|------------|-----------------|
| 0 | `data/raw/` | — | MuSiQue / 2WikiMultihopQA / HotpotQA raw json/jsonl |
| 1 | `decompose/gt/`, `decompose/bart/` | `gt_decompose_to_nl.py`, `decompose_to_nl.py` | question → NL sub-questions, two independent methods (see below) |
| 2 | `traces/` | `construct_balanced_traces.py` | GT decompose + gold/counterfactual evidence → trace jsonl |
| 3 | `hidden_states/` | `extract_hidden_states.py` | trace → per-hop Llama hidden states (`.npz`, not committed) |
| 4 | `gate/` | `fit_lr_gate_pooled.py` | hidden-state Δ vectors → PCA+LogReg joblib artifacts |
| 5 | `retrieval/` | `run_wavefront_all.sh` | BART decompose + gate scores → end-to-end retrieval metrics |

`config/default.yaml` centralizes paths/hyperparams referenced across stages (model names, layer index,
PCA dim, target FPR, retrieval λ, etc.) — check it before hardcoding a path or hyperparameter in a script.

### GT vs BART decomposition — don't mix these up

There are **two independent decomposers**, used for different purposes:

- **GT** (`decompose/gt/`): rewrites MuSiQue's own `question_decomposition` field into NL via GPT.
  MuSiQue-only. Used to build training traces (stage 2) and exact-GT hidden states — this is "ground truth"
  in the sense of coming from the dataset's own annotations, not a trained model.
- **BART** (`decompose/bart/`): a self-trained BART-large decomposer (`train.py`) that runs on all three
  datasets (MuSiQue / 2Wiki / Hotpot dev). Used only for stage-5 end-to-end retrieval evaluation, since
  at inference time there is no ground-truth decomposition to lean on.

Both funnel into `data/decompose/{dataset}/{gt,bart}/*.jsonl`, which is the only path downstream code
(`traces/`, `retrieval/`) actually reads — regenerate the source in `decompose/` first, then sync into
`data/decompose/`. Path selection is centralized in `traces/dataset_loaders.py::get_decompose_path(dataset, split, mode="gt"|"bart")`.

### Trace construction (stage 2) — label protocol

`traces/` builds two trace types deterministically (no LLM mistakes involved):

- **gold** (`trace_type=correct`): every hop uses gold evidence, `wrong_hops=[]`.
- **counterfactual** (`trace_type=error`): for a single injected hop `h`, evidence is swapped for a
  cosine-top-k non-gold passage; `wrong_hops=[h]`. All other hops stay gold. The sub-question's
  `[Answer N]` placeholders are expanded using gold hop answers (matches production behavior).

`construct_balanced_traces.py` is the one to run — it writes `gold/`, `counterfactual/`, and a `merged/`
directory where negatives are up/downsampled per `(K, hop)` bucket to balance against positives (rare
4-hop buckets get upsampled). **Hidden-state extraction (stage 3) always reads `merged/`, not `gold/` or
`counterfactual/` directly.**

Trace text assembly is shared via `traces/trace_format.py::assemble_trace()` — don't reimplement the
"Question: ... Step i: ... Evidence: \"...\" Final Answer: ..." format elsewhere.

### Hidden states → gate: what "transition j" means

For a K-hop trace, hidden states are extracted per cumulative prefix (`h_0..h_{K+1}`, last-token, layer
31 of Llama-3.1-8B-Instruct by default). The gate operates on **Δ features**: `Δ_j = h_j - h_{j-1}`,
computed in `gate/fit_lr_gate_pooled.py`, not stored in the `.npz` files themselves.

Transitions are pooled across K by semantic role (`gate/hop_labels.py` is the single source of truth for
this, referenced from both `gate/` and `traces/`):

| j | transition | used for K |
|---|------------|------------|
| 0 | Q→E1 | 2,3,4 |
| 1 | E1→E2 | 2,3,4 |
| 2 | E2→E3 | 3,4 |
| 3 | E3→E4 | 4 |

There is no transition for the final answer hop (no evidence-selection decision to gate there). Label:
`should_intervene(j) = (j + 1) in wrong_hops`, and only applies to `trace_type=error` traces (`neg/`);
gold traces (`pos/`) are all-negative by construction.

`fit_lr_gate.py` (per-`(K,j)`) is the older, non-pooled variant — `fit_lr_gate_pooled.py` is the
recommended path and what `retrieval/` actually loads (`gate/lr_artifacts.py::load_artifacts(mode="pooled")`).

### Retrieval (stage 5) — method matrix

`run_retrieval_exp_wavefront.py` (batched vLLM, preferred over the older `run_retrieval_exp.py`) composes:
BART decompose → BGE (`bge-base-en-v1.5`) cosine retrieval → optional LR-gate-informed rerank → vLLM
select/answer → final reader. Methods, selected via CLI/env, differ only in the rerank step:

- `baseline` — cosine top-k, no gate.
- `lr_rerank` — cosine score mixed with gate score via `λ · LR_score` (`LAMBDA_LR` env, default from config).
- `gated_rule_a` — only expands the rerank pool when the top-1 gate score exceeds a threshold.
- `oracle` — upper bound, assumes the gold passage is already in the candidate pool.

`run_wavefront_all.sh` runs all three datasets; use `GPUS=`, `LIMIT=`, `DATASETS=` env vars for a quick
smoke test on one dataset before a full run (all three datasets, all methods, is slow/GPU-heavy).

## Data size / git-ignored artifacts

Large binary artifacts are intentionally excluded from git (see root `.gitignore`) and must be
regenerated locally rather than fetched:

- `**/*.npz` — hidden-state activations (`hidden_states/{train,dev}/activations/`, ~92.7k files / 2.4GB).
  Regenerate via `hidden_states/extract_hidden_states.py` (needs the trace jsonl + a local
  Llama-3.1-8B-Instruct checkpoint). `manifest.jsonl` and `run_meta.json` *are* committed, so you can
  inspect what was extracted without regenerating.
- `**/*.pt`, `**/*.safetensors` — model checkpoints (e.g. `decompose/bart/outputs/*/checkpoint-*/`).

## Environment

No project-level `pyproject.toml`/`requirements.txt` — only `decompose/bart/requirements.txt` covers the
BART training/inference stack (`torch>=2.0`, `transformers>=4.40`, `accelerate`, `sacrebleu`). Other
stages (`hidden_states/`, `gate/`, `retrieval/`) assume an ambient environment with torch/transformers,
`scikit-learn`/`joblib` (gate), and `vllm` + BGE embeddings (retrieval) already available — check the
relevant script's imports rather than assuming a shared requirements file covers them.

BART training pins `transformers==4.57.3` specifically (see `decompose/bart/TRAINING_NOTES.md`) — CUDA/
PyTorch versions must match what's in the existing checkpoint's `config.json` if reusing it.

There is no test suite in this repo.
