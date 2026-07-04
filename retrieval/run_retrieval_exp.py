#!/usr/bin/env python3
"""
Retrieval experiment: compare embedding baseline vs LR trajectory rerank.

Pipeline (fully consistent with run_musique_pipeline_gt.py)
------------------------------------------------------------
For every hop j of every example:
  1. Expand sub-question with [Answer N] placeholders filled from prior answers.
  2. Retrieve candidates with embedding similarity.
  3. (Method-specific) rerank / gate logic.
  4. LLM selects one passage from the final top-3  (--selection-mode select, default)
     OR reads all top-3 concatenated and returns answer + passage index
     (--selection-mode concat_answer).
  5. LLM generates a short answer (select mode) or uses the combined JSON answer.
  6. Append answer to prior → used for next hop's question expansion.

Methods
-------
  baseline       Cosine embedding top-3 (no rerank)
  colbert        ColBERTv2 top-3 (no rerank; same downstream as baseline)
  oracle         Oracle coverage check (adds per-method oracle stats)
  lr_rerank      Cosine top-k + LR gate rerank → top-3
  gated_rule_a   Cosine top-3 → if top-1 abnormal > threshold:
                   expand to expand-topk, LR rerank → top-3
  gated_rule_b   Cosine top-3 → if ≥2 of top-3 abnormal > threshold:
                   expand to expand-topk, LR rerank → top-3

Data sources
------------
  --musique-dir      musique_ans_v1.0_dev.jsonl  (paragraphs + gold)
  --decompose-mode   gt (default) | bart_decompose
  --decompose-file   override path; default depends on --decompose-mode:
                       gt             → musique_gt_nl_{split}.jsonl
                       bart_decompose → ../decompose/outputs/musique_pred_nl_{split}.jsonl

BART decompose notes
--------------------
  - sub_questions capped at 4 hops (extra hops discarded).
  - Pipeline runs all pred hops; LR gate artifacts use pred K (not gold hop count).
  - Hops beyond gold decomposition still run (select/answer) but skip gold retrieval metrics.
  - Short answers: LLM outputs NA when passage has no answer (avoids long filler in next hop).
  --artifacts-dir    delta_gate_retrieval/artifacts/  (PCA+LR models)

Metrics
-------
  Retrieval:    Recall@1, Recall@3, MRR, Avg_rank  (per hop / K / chain)
  Selection:    selection_acc, chain_selection_acc
  Case recall:  case_recall@1/3  (all gold hops in top-1/3 for the example)
  Chain (K-match): chain_* / chain_selection_* only when K_pred == K_gold
  Answer:       answer_em, answer_f1               (final answer per example)
  Gate:         hop_trigger_rate, example_trigger_rate (gated methods only)
  Oracle:       P(gold in top-k pool)              (per method, when oracle in --methods)

Usage
-----
  # baseline only (loads LLaMA for selection + answer, same pipeline as LR methods)
  python run_retrieval_exp.py --methods baseline --topk 10 --device cuda:0

  # compare cosine vs ColBERTv2 retrieval (uses colbert-ai; no ragatouille needed)
  python run_retrieval_exp.py \
      --methods baseline colbert --device cuda:2 \
      --run-tag colbert_cmp

  # full experiment (needs GPU for LLM selection + answer gen + LR gate)
  python run_retrieval_exp.py \
      --methods baseline oracle lr_rerank gated_rule_a gated_rule_b \
      --topk 10 --expand-topk 10 --lambda-lr 1.0 \
      --device cuda:2

  # BART-predicted NL decompositions (from decompose/decompose_to_nl.py)
  python run_retrieval_exp.py \
      --decompose-mode bart_decompose \
      --methods baseline --device cuda:0 \
      --run-tag bart_baseline
  # → results/20250607_153045_bart_baseline/retrieval_exp_dev_bart_baseline.json

  # GT × baseline × gated_rule_a × BART (4 runs, one session folder)
  ./run_decompose_cmp.sh
  # → results/20250607_153045_decompose_cmp/{gt_baseline,gt_gated_rule_a,...}/

  # lambda sweep on stratified subset (see sweep_lambda_lr.py)
  python sweep_lambda_lr.py --split dev --limit-per-k 70 --device cuda:0

  # skip separate selection: LLM reads top-3 passages and picks source passage
  python run_retrieval_exp.py --methods baseline --selection-mode concat_answer --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
REPO_ROOT = PROJECT_ROOT.parent
MUSIQUE_DIR = PROJECT_ROOT / "data" / "raw" / "musique"
DECOMPOSE_ROOT = PROJECT_ROOT / "data" / "decompose"
DECOMPOSE_GT_TEMPLATE = DECOMPOSE_ROOT / "musique" / "gt" / "{split}_nl.jsonl"
DECOMPOSE_BART_TEMPLATE = DECOMPOSE_ROOT / "musique" / "bart" / "{split}_nl.jsonl"
DECOMPOSE_BART_DATASET_TEMPLATES = {
    "musique": DECOMPOSE_ROOT / "musique" / "bart" / "{split}_nl.jsonl",
    "2wiki": DECOMPOSE_ROOT / "2wiki" / "bart" / "{split}_nl.jsonl",
    "hotpot": DECOMPOSE_ROOT / "hotpot" / "bart" / "{split}_nl.jsonl",
}
TWOWIKI_DEV_FILE = PROJECT_ROOT / "data" / "raw" / "2wikimultihopqa" / "dev.json"
HOTPOT_DEV_FILE = PROJECT_ROOT / "data" / "raw" / "hotpotqa" / "hotpot_dev_distractor_v1.json"
BART_MAX_HOPS = 4
K_BUCKETS = (1, 2, 3, 4)
ARTIFACTS_DIR = PROJECT_ROOT / "gate" / "artifacts"
POOLED_ARTIFACTS_DIR = PROJECT_ROOT / "gate" / "artifacts_pooled"
HS_EXTRACT = REPO_ROOT / "multihop_trajectory" / "hidden_state_extract"
BASE_INFER = REPO_ROOT / "multihop_trajectory" / "llama_infer_reasoning"


# ── String helpers ─────────────────────────────────────────────────────────────

def escape_double_quotes(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def sanitize_run_tag(tag: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", tag).strip("_") or "run"


def resolve_output_dir(
    *,
    out_dir: Path | None,
    results_root: Path,
    run_tag: str,
    decompose_mode: str,
    methods: list[str],
) -> Path:
    """Default: results_root/{YYYYMMDD_HHMMSS}_{run_tag}/"""
    if out_dir is not None:
        return out_dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = run_tag or f"{decompose_mode}_{'_'.join(methods)}"
    return results_root / f"{ts}_{sanitize_run_tag(tag)}"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_musique(musique_dir: Path, split: str = "dev") -> dict[str, dict]:
    path = musique_dir / f"musique_ans_v1.0_{split}.jsonl"
    if not path.exists():
        sys.exit(f"Missing: {path}")
    rows = load_jsonl(path)
    return {str(r["id"]): r for r in rows}


def context_to_paragraphs(context: list[Any], support_titles: set[str]) -> list[dict]:
    paragraphs: list[dict] = []
    for idx, item in enumerate(context):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        title = str(item[0])
        sentences = item[1]
        if isinstance(sentences, list):
            paragraph_text = " ".join(str(s).strip() for s in sentences if str(s).strip())
        else:
            paragraph_text = str(sentences)
        paragraphs.append(
            {
                "idx": idx,
                "title": title,
                "paragraph_text": paragraph_text,
                "is_supporting": title in support_titles,
            }
        )
    return paragraphs


def unique_support_titles(row: dict) -> list[str]:
    titles: list[str] = []
    for fact in row.get("supporting_facts") or []:
        if not isinstance(fact, (list, tuple)) or not fact:
            continue
        title = str(fact[0])
        if title not in titles:
            titles.append(title)
    return titles


def normalize_open_qa_record(row: dict, dataset: str) -> dict:
    row_id = str(row.get("id") or row.get("_id") or "")
    support_titles = unique_support_titles(row)
    paragraphs = context_to_paragraphs(row.get("context") or [], set(support_titles))
    title_to_idx = {p["title"]: int(p["idx"]) for p in paragraphs}
    gold_decomp = []
    for title in support_titles[:BART_MAX_HOPS]:
        if title in title_to_idx:
            gold_decomp.append(
                {
                    "question": "",
                    "answer": row.get("answer", ""),
                    "paragraph_support_idx": title_to_idx[title],
                }
            )
    return {
        "id": row_id,
        "dataset": dataset,
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "answerable": True,
        "paragraphs": paragraphs,
        "question_decomposition": gold_decomp,
    }


def load_2wiki(path: Path) -> dict[str, dict]:
    if not path.exists():
        sys.exit(f"Missing: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(r.get("_id") or r.get("id")): normalize_open_qa_record(r, "2wiki")
        for r in rows
    }


def load_hotpot(path: Path) -> dict[str, dict]:
    if not path.exists():
        sys.exit(f"Missing: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(r.get("_id") or r.get("id")): normalize_open_qa_record(r, "hotpot")
        for r in rows
    }


def load_dataset_records(args: argparse.Namespace) -> dict[str, dict]:
    if args.dataset == "musique":
        return load_musique(args.musique_dir, args.split)
    if args.dataset == "2wiki":
        return load_2wiki(args.twowiki_file)
    if args.dataset == "hotpot":
        return load_hotpot(args.hotpot_file)
    raise ValueError(f"unknown dataset: {args.dataset}")


def load_decompose_index(path: Path) -> dict[str, list[str]]:
    """Return {id: [raw_sub_question, ...]} from a NL decompose JSONL."""
    index: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rid = str(obj.get("id", "")).strip()
            sqs = obj.get("sub_questions") or []
            if rid and isinstance(sqs, list):
                index[rid] = [str(s) for s in sqs if str(s).strip()]
    return index


def prepare_sub_questions(
    sub_questions: list[str],
    *,
    decompose_mode: str,
) -> list[str]:
    """BART decompose: cap at BART_MAX_HOPS; LR gate uses this hop count."""
    if decompose_mode == "bart_decompose" and len(sub_questions) > BART_MAX_HOPS:
        return sub_questions[:BART_MAX_HOPS]
    return sub_questions


def normalize_short_answer(raw: str) -> str:
    """First line only; empty or explanatory non-answers → NA for next-hop expand."""
    ans = raw.split("\n")[0].strip().rstrip(".")
    if not ans:
        return "NA"
    lower = ans.lower()
    no_info_markers = (
        "no information",
        "not mentioned",
        "not specified",
        "cannot be determined",
        "does not contain",
        "doesn't contain",
        "not found in the passage",
        "unknown",
    )
    if len(ans) > 40 and any(m in lower for m in no_info_markers):
        return "NA"
    return ans


def resolve_decompose_file(
    mode: str,
    split: str,
    override: Path | None,
    dataset: str = "musique",
) -> Path:
    if override is not None:
        return override
    if mode == "bart_decompose":
        if dataset in DECOMPOSE_BART_DATASET_TEMPLATES:
            return Path(str(DECOMPOSE_BART_DATASET_TEMPLATES[dataset]).format(split=split))
        return Path(str(DECOMPOSE_BART_TEMPLATE).format(split=split))
    if mode == "gt":
        if dataset != "musique":
            raise ValueError("--decompose-mode gt is only available for MuSiQue in this script")
        return Path(str(DECOMPOSE_GT_TEMPLATE).format(split=split))
    raise ValueError(f"unknown decompose mode: {mode}")


# ── Lazy imports ───────────────────────────────────────────────────────────────

def _get_hs_helpers():
    """encode_user_only + last_token_hidden for LR gate scoring."""
    sys.path.insert(0, str(HS_EXTRACT))
    from extract_reasoning_trace_hiddenstates import encode_user_only, last_token_hidden
    return encode_user_only, last_token_hidden


def _get_pipeline_helpers():
    """Helpers from base pipeline: expand, select prompt, parse, short answer, judge."""
    sys.path.insert(0, str(BASE_INFER))
    from run_musique_pipeline import (
        expand_hop_template,
        prompt_select_passage,
        parse_json_choice,
        prompt_short_answer,
        prompt_concat_answer,
        parse_json_answer_choice,
        judge_answer_official,
    )
    return (expand_hop_template, prompt_select_passage, parse_json_choice,
            prompt_short_answer, prompt_concat_answer, parse_json_answer_choice,
            judge_answer_official)


# ── First-stage retrieval ──────────────────────────────────────────────────────

COSINE_METHODS = frozenset({"baseline", "lr_rerank", "gated_rule_a", "gated_rule_b"})
COLBERT_METHODS = frozenset({"colbert"})


def retriever_for_method(method: str) -> str:
    if method in COLBERT_METHODS:
        return "colbert"
    if method in COSINE_METHODS:
        return "cosine"
    raise ValueError(f"unknown method for retriever mapping: {method}")


_ST_CACHE: dict[str, Any] = {}
_COLBERT_CKPT: dict[str, Any] = {}


def _get_colbert_checkpoint(model_name: str):
    if model_name not in _COLBERT_CKPT:
        import warnings
        from colbert.infra import ColBERTConfig
        from colbert.modeling.checkpoint import Checkpoint

        # colbert-ai uses deprecated torch.cuda.amp APIs on PyTorch 2.x; harmless noise.
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module=r"colbert\.utils\.amp",
        )

        print(f"  [colbert] loading {model_name} …", file=sys.stderr, flush=True)
        config = ColBERTConfig()
        _COLBERT_CKPT[model_name] = Checkpoint(model_name, colbert_config=config)
    return _COLBERT_CKPT[model_name]


def embed_retrieval(
    query: str,
    paragraphs: list[dict],
    k: int,
    model_name: str = "BAAI/bge-base-en-v1.5",
) -> list[tuple[dict, float]]:
    """Return top-k (paragraph, cosine_score) sorted descending."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    if model_name not in _ST_CACHE:
        print(f"  [emb] loading {model_name} …", file=sys.stderr, flush=True)
        _ST_CACHE[model_name] = SentenceTransformer(model_name)
    st = _ST_CACHE[model_name]

    docs = [
        f"{p.get('title') or ''}\n{(p.get('paragraph_text') or '').strip()}"
        for p in paragraphs
    ]
    qv = st.encode([query], normalize_embeddings=True)
    dv = st.encode(docs, normalize_embeddings=True)
    sims = (qv @ dv.T)[0]
    order = np.argsort(-sims)[:k].tolist()
    return [(paragraphs[i], float(sims[i])) for i in order]


def colbert_retrieval(
    query: str,
    paragraphs: list[dict],
    k: int,
    model_name: str = "colbert-ir/colbertv2.0",
) -> list[tuple[dict, float]]:
    """ColBERTv2 via colbert-ai (in-memory MaxSim over example paragraphs)."""
    import torch
    from colbert.modeling.colbert import colbert_score

    docs = [
        f"{p.get('title') or ''}\n{(p.get('paragraph_text') or '').strip()}".strip()
        for p in paragraphs
    ]
    if not docs:
        return []

    try:
        ckpt = _get_colbert_checkpoint(model_name)
    except ImportError as e:
        raise ImportError(
            "method=colbert requires: pip install colbert-ai\n" + str(e)
        ) from e

    Q = ckpt.queryFromText([query], bsize=1)
    D = ckpt.docFromText(docs, keep_dims=True)
    if isinstance(D, tuple):
        D = D[0]
    mask = torch.ones(D.shape[:2], dtype=torch.bool)
    scores = colbert_score(Q, D, mask, config=ckpt.colbert_config)
    order = scores.argsort(descending=True).tolist()[:k]
    return [(paragraphs[i], float(scores[i].item())) for i in order]


def retrieve_candidates(
    query: str,
    paragraphs: list[dict],
    k: int,
    *,
    retriever: str,
    cos_model: str,
    colbert_model: str,
) -> list[tuple[dict, float]]:
    if retriever == "cosine":
        return embed_retrieval(query, paragraphs, k, cos_model)
    if retriever == "colbert":
        return colbert_retrieval(query, paragraphs, k, colbert_model)
    raise ValueError(f"unknown retriever: {retriever}")


# ── LR gate scoring ────────────────────────────────────────────────────────────

def patch_legacy_logistic_regression(lr: Any) -> None:
    """Older sklearn LR pickles may miss attrs read by newer predict_proba."""
    if not hasattr(lr, "multi_class"):
        lr.multi_class = "auto"


def patch_gate_artifact(artifact: dict) -> dict:
    lr = artifact.get("lr") if isinstance(artifact, dict) else None
    if lr is not None:
        patch_legacy_logistic_regression(lr)
    return artifact


def load_artifacts(artifacts_dir: Path, mode: str = "nonpooled") -> dict:
    try:
        import joblib
    except ImportError:
        sys.exit("Install joblib: pip install joblib")
    if mode == "pooled":
        arts: dict[int, dict] = {}
        for f in sorted(artifacts_dir.glob("j*.joblib")):
            m = re.match(r"j(\d+)\.joblib", f.name)
            if m:
                arts[int(m.group(1))] = patch_gate_artifact(joblib.load(f))
    else:
        arts: dict[tuple[int, int], dict] = {}
        for f in sorted(artifacts_dir.glob("K*_j*.joblib")):
            m = re.match(r"K(\d+)_j(\d+)\.joblib", f.name)
            if m:
                K, j = int(m.group(1)), int(m.group(2))
                arts[(K, j)] = patch_gate_artifact(joblib.load(f))
    print(
        f"  [gate] loaded {len(arts)} {mode} artifacts from {artifacts_dir}",
        file=sys.stderr,
    )
    return arts


def get_gate_artifact(artifacts: dict, mode: str, K: int, j_art: int) -> dict | None:
    if mode == "pooled":
        return artifacts.get(j_art)
    return artifacts.get((K, j_art))


def build_trace_prefix(q_main: str, hop_steps: list[tuple[str, str]]) -> str:
    """
    Build cumulative trace prefix matching the format used during LR gate training.
    hop_steps: [(expanded_q_1, evidence_text_1), ...] for completed hops.
    """
    s = f"Question: {q_main}"
    for i, (eq, ev) in enumerate(hop_steps, start=1):
        s += f' Step {i}: {eq} Evidence: "{escape_double_quotes(ev)}"'
    return s


def compute_abnormal_scores(
    *,
    candidates: list[tuple[dict, float]],
    prefix_before_hop: str,
    hop_j: int,
    expanded_qj: str,
    K: int,
    artifacts: dict,
    gate_artifact_mode: str,
    model,
    tokenizer,
    layer: int,
    hidden_cache: dict[tuple[int, str], Any] | None = None,
) -> list[tuple[dict, float, float]]:
    """
    Returns list of (para, emb_score, abnormal_score).
    abnormal_score = 0.0 if no artifact exists for this hop.
    """
    import torch
    import numpy as np

    j_art = hop_j - 1
    art = get_gate_artifact(artifacts, gate_artifact_mode, K, j_art)
    if art is None:
        return [(p, s, 0.0) for p, s in candidates]

    pca = art["pca"]
    lr  = art["lr"]
    dev = next(model.parameters()).device

    if hidden_cache is None:
        hidden_cache = {}

    def render_chat_text(text: str) -> str:
        messages = [{"role": "user", "content": text}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=False,
            )
        except Exception:
            return text

    def cached_last_hidden_batch(texts: list[str]) -> list[np.ndarray]:
        missing = [text for text in texts if (layer, text) not in hidden_cache]
        if missing:
            rendered = [render_chat_text(text) for text in missing]
            batch = tokenizer(rendered, return_tensors="pt", padding=True)
            input_ids = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            with torch.inference_mode():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
            hs = out.hidden_states
            if hs is None or layer + 1 >= len(hs):
                raise ValueError(
                    f"bad hidden_states len={len(hs) if hs else 0} for layer={layer}"
                )
            layer_h = hs[layer + 1]
            last_indices = (
                attention_mask.size(1)
                - 1
                - torch.flip(attention_mask, dims=[1]).argmax(dim=1)
            )
            batch_indices = torch.arange(layer_h.size(0), device=dev)
            hidden = layer_h[batch_indices, last_indices].float().cpu().numpy()
            hidden = hidden.astype(np.float32)
            for text, vec in zip(missing, hidden):
                hidden_cache[(layer, text)] = vec
        return [hidden_cache[(layer, text)] for text in texts]

    h_prev = cached_last_hidden_batch([prefix_before_hop])[0]

    candidate_texts: list[str] = []
    for para, _ in candidates:
        para_text = (para.get("paragraph_text") or "").strip()
        candidate_texts.append(
            f'{prefix_before_hop} Step {hop_j}: {expanded_qj}'
            f' Evidence: "{escape_double_quotes(para_text)}"'
        )
    candidate_hiddens = cached_last_hidden_batch(candidate_texts)
    result: list[tuple[dict, float, float]] = []
    for (para, emb_score), h_j in zip(candidates, candidate_hiddens):
        delta = (h_j - h_prev).astype("float64").reshape(1, -1)
        z = pca.transform(delta)
        abnormal = float(lr.predict_proba(z)[0, 1])
        result.append((para, emb_score, abnormal))

    return result


def rerank_by_final_score(
    scored: list[tuple[dict, float, float]],
    lam: float,
) -> list[tuple[dict, float]]:
    """Rerank by emb_score - lam * abnormal_score."""
    final = [(p, es - lam * ab) for p, es, ab in scored]
    final.sort(key=lambda x: -x[1])
    return final


# ── LLM generation helpers ─────────────────────────────────────────────────────

class VllmGenerator:
    def __init__(
        self,
        model: str,
        *,
        dtype: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int | None,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            sys.exit("Install vLLM or use --llm-backend transformers")
        kwargs: dict[str, Any] = {
            "model": model,
            "dtype": dtype,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        if max_model_len:
            kwargs["max_model_len"] = max_model_len
        self.llm = LLM(**kwargs)
        self.SamplingParams = SamplingParams

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        sampling_params = self.SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
        )
        outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()


def _llm_generate(prompt: str, model, tokenizer, max_new_tokens: int) -> str:
    """Run one LLM generation call; return decoded new tokens."""
    if isinstance(model, VllmGenerator):
        return model.generate(prompt, max_new_tokens)

    import torch

    messages = [{"role": "user", "content": prompt}]
    dev = next(model.parameters()).device
    try:
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(dev)
    except Exception:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(dev)

    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def llm_select_passage(
    candidates: list[dict],
    expanded_q: str,
    model,
    tokenizer,
    max_new_tokens: int = 64,
) -> int:
    """LLM selects best passage from candidates; returns 0-indexed choice."""
    _, prompt_select_passage, parse_json_choice, _, _, _, _ = _get_pipeline_helpers()
    raw = _llm_generate(prompt_select_passage(expanded_q, candidates), model, tokenizer, max_new_tokens)
    choice = parse_json_choice(raw)
    if choice is None:
        choice = 0
    return max(0, min(len(candidates) - 1, choice))


def llm_short_answer(
    expanded_q: str,
    passage: dict,
    model,
    tokenizer,
    max_new_tokens: int = 64,
) -> str:
    """LLM generates a short answer from the selected passage."""
    _, _, _, prompt_short_answer, _, _, _ = _get_pipeline_helpers()
    raw = _llm_generate(prompt_short_answer(expanded_q, passage), model, tokenizer, max_new_tokens)
    return normalize_short_answer(raw)


def llm_concat_answer(
    expanded_q: str,
    candidates: list[dict],
    model,
    tokenizer,
    max_new_tokens: int = 128,
) -> tuple[str, int]:
    """LLM reads all top-3 passages; returns (short_answer, 0-indexed passage choice)."""
    _, _, _, _, prompt_concat_answer, parse_json_answer_choice, _ = _get_pipeline_helpers()
    raw = _llm_generate(
        prompt_concat_answer(expanded_q, candidates), model, tokenizer, max_new_tokens,
    )
    ans, choice = parse_json_answer_choice(raw)
    if ans is None:
        ans = normalize_short_answer(raw)
    else:
        ans = normalize_short_answer(ans)
    if choice is None:
        choice = 0
    choice = max(0, min(len(candidates) - 1, choice))
    return ans, choice


# ── Metrics ────────────────────────────────────────────────────────────────────

class MetricAccum:
    def __init__(self):
        self.total = 0; self.r1 = 0; self.r3 = 0
        self.rr_sum = 0.0; self.rank_sum = 0

    def update(self, gold_rank: int | None):
        self.total += 1
        if gold_rank is None:
            return
        self.r1 += int(gold_rank == 1)
        self.r3 += int(gold_rank <= 3)
        self.rr_sum += 1.0 / gold_rank
        self.rank_sum += gold_rank

    def result(self) -> dict:
        n = self.total
        return {
            "n_hops":   n,
            "recall@1": round(self.r1 / n, 4) if n else 0.0,
            "recall@3": round(self.r3 / n, 4) if n else 0.0,
            "mrr":      round(self.rr_sum / n, 4) if n else 0.0,
            "avg_rank": round(self.rank_sum / n, 2) if n else None,
        }


class ChainAccum:
    def __init__(self):
        self.n_examples = 0; self.chain_r1 = 0; self.chain_r3 = 0

    def update(self, hop_ranks: list[int | None]):
        self.n_examples += 1
        if all(r is not None and r == 1 for r in hop_ranks):
            self.chain_r1 += 1
        if all(r is not None and r <= 3 for r in hop_ranks):
            self.chain_r3 += 1

    def result(self) -> dict:
        n = self.n_examples
        return {
            "n_examples":     n,
            "chain_recall@1": round(self.chain_r1 / n, 4) if n else 0.0,
            "chain_recall@3": round(self.chain_r3 / n, 4) if n else 0.0,
        }


class CaseRecallAccum:
    """Example-level: success if every gold hop's support is in top-1 / top-3."""

    def __init__(self):
        self.n_examples = 0
        self.case_r1 = 0
        self.case_r3 = 0

    def update(self, gold_ranks: list[int | None]):
        if not gold_ranks:
            return
        self.n_examples += 1
        if all(r is not None and r == 1 for r in gold_ranks):
            self.case_r1 += 1
        if all(r is not None and r <= 3 for r in gold_ranks):
            self.case_r3 += 1

    def result(self) -> dict:
        n = self.n_examples
        return {
            "n_examples":    n,
            "case_recall@1": round(self.case_r1 / n, 4) if n else 0.0,
            "case_recall@3": round(self.case_r3 / n, 4) if n else 0.0,
        }


class SelectionAccum:
    def __init__(self):
        self.total = 0; self.correct = 0

    def update(self, correct: bool | None):
        self.total += 1
        if correct:
            self.correct += 1

    def result(self) -> dict:
        n = self.total
        return {"n_hops": n, "selection_acc": round(self.correct / n, 4) if n else 0.0}


class SelectionChainAccum:
    def __init__(self):
        self.n_examples = 0; self.chain_correct = 0

    def update(self, hop_corrects: list[bool | None]):
        self.n_examples += 1
        if all(c is True for c in hop_corrects):
            self.chain_correct += 1

    def result(self) -> dict:
        n = self.n_examples
        return {
            "n_examples":          n,
            "chain_selection_acc": round(self.chain_correct / n, 4) if n else 0.0,
        }


class MethodTimingAccum:
    """Per-method cumulative wall time (seconds) across all hops/examples."""

    def __init__(self):
        self.n_hops = 0
        self.n_examples = 0
        self.total_sec = 0.0
        self.retrieval_sec = 0.0
        self.gate_sec = 0.0
        self.llm_select_sec = 0.0
        self.llm_answer_sec = 0.0

    def add_hop(
        self,
        *,
        total: float,
        retrieval: float = 0.0,
        gate: float = 0.0,
        llm_select: float = 0.0,
        llm_answer: float = 0.0,
    ) -> None:
        self.n_hops += 1
        self.total_sec += total
        self.retrieval_sec += retrieval
        self.gate_sec += gate
        self.llm_select_sec += llm_select
        self.llm_answer_sec += llm_answer

    def add_example(self) -> None:
        self.n_examples += 1

    def result(self) -> dict:
        n_h = self.n_hops
        n_e = self.n_examples
        return {
            "n_hops": n_h,
            "n_examples": n_e,
            "total_sec": round(self.total_sec, 2),
            "total_min": round(self.total_sec / 60, 2),
            "total_hr": round(self.total_sec / 3600, 3),
            "sec_per_hop": round(self.total_sec / n_h, 4) if n_h else None,
            "sec_per_example": round(self.total_sec / n_e, 2) if n_e else None,
            "breakdown": {
                "retrieval_sec": round(self.retrieval_sec, 2),
                "gate_sec": round(self.gate_sec, 2),
                "llm_select_sec": round(self.llm_select_sec, 2),
                "llm_answer_sec": round(self.llm_answer_sec, 2),
            },
        }


class AnswerAccum:
    """Example-level final answer EM / F1."""
    def __init__(self):
        self.n = 0; self.em_sum = 0; self.f1_sum = 0.0

    def update(self, em: int | None, f1: float | None):
        if em is None:
            return
        self.n += 1
        self.em_sum += em
        self.f1_sum += (f1 or 0.0)

    def result(self) -> dict:
        n = self.n
        return {
            "n_examples": n,
            "answer_em":  round(self.em_sum / n, 4) if n else 0.0,
            "answer_f1":  round(self.f1_sum / n, 4) if n else 0.0,
        }


class GateAccum:
    def __init__(self):
        self.n_hops = 0; self.n_triggered_hops = 0
        self.n_examples = 0; self.n_examples_any_triggered = 0

    def update_hop(self, triggered: bool):
        self.n_hops += 1
        if triggered:
            self.n_triggered_hops += 1

    def update_example(self, any_triggered: bool):
        self.n_examples += 1
        if any_triggered:
            self.n_examples_any_triggered += 1

    def result(self) -> dict:
        nh = self.n_hops; ne = self.n_examples
        return {
            "n_hops":                   nh,
            "n_triggered_hops":         self.n_triggered_hops,
            "hop_trigger_rate":         round(self.n_triggered_hops / nh, 4) if nh else 0.0,
            "n_examples":               ne,
            "n_examples_any_triggered": self.n_examples_any_triggered,
            "example_trigger_rate":     round(self.n_examples_any_triggered / ne, 4) if ne else 0.0,
        }


def find_gold_rank(ranked_paras: list[tuple[dict, float]], gold_idx: int) -> int | None:
    for rank, (para, _) in enumerate(ranked_paras, start=1):
        if int(para.get("idx", -1)) == gold_idx:
            return rank
    return None


def hop_count(decompose_idx: dict[str, list[str]], eid: str) -> int:
    return len(decompose_idx.get(eid) or [])


def gold_hop_ranks(
    hop_results: list[dict],
    method: str,
    K_gold: int,
) -> list[int | None]:
    """Gold support recall rank at each gold hop j (0-indexed), aligned to pred hop j."""
    ranks: list[int | None] = []
    for gj in range(K_gold):
        if gj >= len(hop_results):
            ranks.append(None)
            continue
        md = hop_results[gj].get(method)
        if md is None:
            ranks.append(None)
        else:
            ranks.append(md.get("gold_rank"))
    return ranks


def gold_hop_selections(
    hop_results: list[dict],
    method: str,
    K_gold: int,
) -> list[bool | None]:
    """LLM selection correctness at each gold hop j (0-indexed)."""
    sels: list[bool | None] = []
    for gj in range(K_gold):
        if gj >= len(hop_results):
            sels.append(None)
            continue
        md = hop_results[gj].get(method)
        if md is None:
            sels.append(None)
            continue
        sel = md.get("selection") or {}
        sels.append(sel.get("correct"))
    return sels


def case_recall_flags(gold_ranks: list[int | None]) -> tuple[bool, bool]:
    if not gold_ranks:
        return False, False
    r1 = all(r is not None and r == 1 for r in gold_ranks)
    r3 = all(r is not None and r <= 3 for r in gold_ranks)
    return r1, r3


def select_example_ids(
    ids: list[str],
    decompose_idx: dict[str, list[str]],
    *,
    limit: int = 0,
    limit_per_k: int = 0,
    seed: int = 42,
) -> list[str]:
    """
    Choose example ids for the run.

    limit_per_k > 0: up to N examples per hop count K in {1, 2, 3, 4} (stratified).
    limit > 0: first N ids in sorted order (legacy; ignored if limit_per_k > 0).
    """
    if limit_per_k > 0:
        import random

        by_k: dict[int, list[str]] = {K: [] for K in K_BUCKETS}
        other: list[str] = []
        for eid in ids:
            K = hop_count(decompose_idx, eid)
            if K in by_k:
                by_k[K].append(eid)
            else:
                other.append(eid)
        rng = random.Random(seed)
        picked: list[str] = []
        for K in K_BUCKETS:
            pool = sorted(by_k[K])
            rng.shuffle(pool)
            picked.extend(pool[:limit_per_k])
        if other:
            print(f"  [warn] {len(other)} examples with K not in {K_BUCKETS} skipped", flush=True)
        print(
            "  stratified sample (limit-per-k="
            f"{limit_per_k}, seed={seed}): "
            + ", ".join(f"K={K}:{min(limit_per_k, len(by_k[K]))}" for K in K_BUCKETS)
            + f"  total={len(picked)}",
            flush=True,
        )
        return sorted(picked)
    if limit > 0:
        print(f"  limited to first {limit} examples (sorted id order)", flush=True)
        return ids[:limit]
    return ids


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="End-to-end retrieval experiment on MuSiQue dev "
                    "(fully consistent pipeline: expand → retrieve → select → answer)."
    )
    ap.add_argument("--split",           default="dev")
    ap.add_argument("--dataset",         choices=["musique", "2wiki", "hotpot"], default="musique")
    ap.add_argument("--musique-dir",     type=Path, default=MUSIQUE_DIR)
    ap.add_argument("--twowiki-file",    type=Path, default=TWOWIKI_DEV_FILE)
    ap.add_argument("--hotpot-file",     type=Path, default=HOTPOT_DEV_FILE)
    ap.add_argument(
        "--decompose-mode",
        choices=["gt", "bart_decompose"],
        default="gt",
        help="gt: musique_gt_nl_{split}.jsonl; "
             "bart_decompose: decompose/outputs/musique_pred_nl_{split}.jsonl",
    )
    ap.add_argument("--decompose-file",  type=Path, default=None,
                    help="Override decompose JSONL (default from --decompose-mode)")
    ap.add_argument(
        "--gate-artifact-mode",
        choices=["nonpooled", "pooled"],
        default="nonpooled",
        help="nonpooled uses K*_j*.joblib artifacts; pooled uses j*.joblib artifacts.",
    )
    ap.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Gate artifact directory. Default follows --gate-artifact-mode.",
    )
    ap.add_argument(
        "--results-root",
        type=Path,
        default=HERE / "results",
        help="Parent directory for auto timestamp run folders (default: retrieval_exp/results)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Explicit output directory. Default: results-root/{timestamp}_{run_tag}/",
    )
    ap.add_argument("--out-cases",       type=Path, default=None)
    ap.add_argument("--methods",         nargs="+",
                    choices=["baseline", "colbert", "oracle", "lr_rerank",
                             "gated_rule_a", "gated_rule_b"],
                    default=["baseline", "oracle"])
    ap.add_argument("--topk",            type=int,   default=10,
                    help="Candidate pool size for lr_rerank")
    ap.add_argument("--expand-topk",     type=int,   default=10,
                    help="Expanded pool size when gate fires")
    ap.add_argument("--lambda-lr",       type=float, default=1.0)
    ap.add_argument("--abnormal-threshold", type=float, default=0.5,
                    help="Fallback threshold when artifact has no calibrated threshold. "
                         "Normally each (K,j) artifact stores its own train-calibrated threshold "
                         "(FPR ≤ 10%% on train set); this arg is only used as a fallback.")
    ap.add_argument("--cos-model",       default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--colbert-model",   default="colbert-ir/colbertv2.0",
                    help="ColBERT checkpoint for method=colbert (via colbert-ai)")
    ap.add_argument("--answerable-only", action="store_true", default=True)
    ap.add_argument("--limit",           type=int,   default=0,
                    help="Use first N example ids (sorted). Ignored if --limit-per-k > 0.")
    ap.add_argument("--limit-per-k",     type=int,   default=0,
                    help="Stratified cap: up to N examples per K in {2,3,4} (for lambda sweep).")
    ap.add_argument("--sample-seed",       type=int,   default=42,
                    help="RNG seed for --limit-per-k shuffling.")
    ap.add_argument("--run-tag",           type=str,   default="",
                    help="Optional suffix for output files, e.g. lam1.0 → retrieval_exp_dev_lam1.0.json")
    # LLaMA
    ap.add_argument("--model",           default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--llm-backend",     choices=["transformers", "vllm"], default="transformers")
    ap.add_argument("--layer",           type=int,   default=31)
    ap.add_argument("--dtype",           choices=("bfloat16", "float16", "float32"),
                    default="bfloat16")
    ap.add_argument("--device",          default=None)
    ap.add_argument("--attn-implementation", default=None,
                    help="Transformers attention backend, e.g. sdpa or flash_attention_2.")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=0)
    ap.add_argument("--max-new-tokens-select", type=int, default=64)
    ap.add_argument("--max-new-tokens-answer", type=int, default=64)
    ap.add_argument("--num-shards",      type=int, default=1)
    ap.add_argument("--shard-index",     type=int, default=0)
    ap.add_argument(
        "--selection-mode",
        choices=["select", "concat_answer"],
        default="select",
        help="select: LLM picks one of top-3 then extracts short answer (default). "
             "concat_answer: LLM reads all top-3 and returns JSON {answer, choice}; "
             "chosen passage text is used in the trace prefix for the next hop.",
    )
    args = ap.parse_args()

    if args.artifacts_dir is None:
        args.artifacts_dir = (
            POOLED_ARTIFACTS_DIR
            if args.gate_artifact_mode == "pooled"
            else ARTIFACTS_DIR
        )

    args.decompose_file = resolve_decompose_file(
        args.decompose_mode, args.split, args.decompose_file, args.dataset
    )
    if not args.decompose_file.exists():
        sys.exit(f"Missing decompose file: {args.decompose_file}")

    if not args.run_tag and args.decompose_mode == "bart_decompose":
        args.run_tag = "bart_decompose"

    args.out_dir = resolve_output_dir(
        out_dir=args.out_dir,
        results_root=args.results_root,
        run_tag=args.run_tag,
        decompose_mode=args.decompose_mode,
        methods=args.methods,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.out_dir}", flush=True)
    if args.selection_mode != "select":
        print(f"Selection mode: {args.selection_mode}", flush=True)
    print(
        f"Gate artifacts: mode={args.gate_artifact_mode} dir={args.artifacts_dir}",
        flush=True,
    )

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading {args.dataset} data …", flush=True)
    records = load_dataset_records(args)
    print(f"  {len(records)} records")

    print(
        f"Loading decompose file ({args.decompose_mode}): {args.decompose_file} …",
        flush=True,
    )
    decompose_idx = load_decompose_index(args.decompose_file)
    print(f"  {len(decompose_idx)} decompositions")

    ids = sorted(set(records) & set(decompose_idx))
    print(f"  intersect: {len(ids)} examples")

    if args.answerable_only:
        ids = [i for i in ids if records[i].get("answerable", True)]
        print(f"  answerable: {len(ids)} examples")

    ids = select_example_ids(
        ids,
        decompose_idx,
        limit=args.limit,
        limit_per_k=args.limit_per_k,
        seed=args.sample_seed,
    )
    if args.num_shards < 1:
        sys.exit("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        sys.exit("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.num_shards > 1:
        ids = ids[args.shard_index :: args.num_shards]
        print(
            f"  shard {args.shard_index}/{args.num_shards}: {len(ids)} examples",
            flush=True,
        )
    print(f"  using {len(ids)} examples")

    ranker_methods = [m for m in args.methods if m != "oracle"]
    if not ranker_methods and "oracle" not in args.methods:
        sys.exit("No methods selected.")

    # ── Load LR artifacts (needed for lr_rerank / gated; harmless if unused) ──
    artifacts: dict = {}
    if any(m in ("lr_rerank", "gated_rule_a", "gated_rule_b") for m in args.methods):
        artifacts = load_artifacts(args.artifacts_dir, args.gate_artifact_mode)
    else:
        print("  [gate] no LR/gated methods — skipping artifact load", file=sys.stderr)

    # ── Load LLaMA for full pipeline (select + answer; gate methods also score Δ) ─
    # baseline alone uses the same LLM selection/answer path as lr_rerank.
    with_llm = len(ranker_methods) > 0

    llama_model = llama_tokenizer = None
    gate_model = gate_tokenizer = None
    if with_llm:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            sys.exit("Install: pip install torch transformers")

        if args.device is None:
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                     "float32": torch.float32}
        dtype  = dtype_map[args.dtype]
        dev_map = None if args.device == "cpu" else {"": args.device}
        needs_gate_model = any(
            m in ("lr_rerank", "gated_rule_a", "gated_rule_b") for m in args.methods
        )

        if args.llm_backend == "vllm":
            print(f"Loading vLLM generator ({args.model}) …", flush=True)
            llama_model = VllmGenerator(
                args.model,
                dtype=args.dtype,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len or None,
            )
            llama_tokenizer = None
            print("  vLLM ready.", flush=True)
        else:
            print(f"Loading LLaMA ({args.model}) …", flush=True)
            llama_tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
            if llama_tokenizer.pad_token is None:
                llama_tokenizer.pad_token = llama_tokenizer.eos_token
            model_kwargs = {"dtype": dtype, "device_map": dev_map}
            if args.attn_implementation:
                model_kwargs["attn_implementation"] = args.attn_implementation
            llama_model = AutoModelForCausalLM.from_pretrained(
                args.model, **model_kwargs
            )
            llama_model.eval()
            print("  LLaMA ready.", flush=True)

        if needs_gate_model:
            if args.llm_backend == "vllm":
                print(f"Loading gate transformer model ({args.model}) …", flush=True)
                gate_tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
                if gate_tokenizer.pad_token is None:
                    gate_tokenizer.pad_token = gate_tokenizer.eos_token
                gate_model_kwargs = {"dtype": dtype, "device_map": dev_map}
                if args.attn_implementation:
                    gate_model_kwargs["attn_implementation"] = args.attn_implementation
                gate_model = AutoModelForCausalLM.from_pretrained(
                    args.model, **gate_model_kwargs
                )
                gate_model.eval()
                print("  gate model ready.", flush=True)
            else:
                gate_model = llama_model
                gate_tokenizer = llama_tokenizer

    # ── Accumulators ──────────────────────────────────────────────────────────
    GATED = {"gated_rule_a", "gated_rule_b"}

    def _make_accum(cls):
        return {"all": cls(), **{K: cls() for K in K_BUCKETS}}

    accum:          dict[str, dict] = {m: _make_accum(MetricAccum)         for m in ranker_methods}
    chain_accum:    dict[str, dict] = {m: _make_accum(ChainAccum)          for m in ranker_methods}
    chain_accum_k_match: dict[str, dict] = {
        m: _make_accum(ChainAccum) for m in ranker_methods
    }
    case_recall_accum: dict[str, dict] = {
        m: _make_accum(CaseRecallAccum) for m in ranker_methods
    }
    accum_hop:      dict[str, dict] = {m: defaultdict(MetricAccum)         for m in ranker_methods}
    sel_accum:      dict[str, dict] = {m: _make_accum(SelectionAccum)      for m in ranker_methods}
    sel_chain_accum:dict[str, dict] = {m: _make_accum(SelectionChainAccum) for m in ranker_methods}
    sel_chain_accum_k_match: dict[str, dict] = {
        m: _make_accum(SelectionChainAccum) for m in ranker_methods
    }
    sel_accum_hop:  dict[str, dict] = {m: defaultdict(SelectionAccum)      for m in ranker_methods}
    ans_accum:      dict[str, dict] = {m: _make_accum(AnswerAccum)         for m in ranker_methods}
    gate_accum:     dict[str, GateAccum] = {m: GateAccum() for m in ranker_methods if m in GATED}
    timing_accum:   dict[str, MethodTimingAccum] = {m: MethodTimingAccum() for m in ranker_methods}

    # Oracle (per method)
    oracle_ks = [3, 5, 10, 20]
    oracle_accum: dict[str, dict[int, MetricAccum]] = {
        m: {k: MetricAccum() for k in oracle_ks} for m in ranker_methods
    }
    oracle_chain: dict[str, dict[int, ChainAccum]] = {
        m: {k: ChainAccum() for k in oracle_ks} for m in ranker_methods
    }

    # ── Open per-case JSONL ────────────────────────────────────────────────────
    tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
    if args.out_cases is None:
        args.out_cases = args.out_dir / f"retrieval_cases_{args.split}{tag_suffix}.jsonl"
    args.out_cases.parent.mkdir(parents=True, exist_ok=True)
    cases_fp = args.out_cases.open("w", encoding="utf-8")

    # ── Pipeline helpers ───────────────────────────────────────────────────────
    expand_hop_template = None
    judge_answer_official = None
    if with_llm or ranker_methods:  # always load for question expansion
        (expand_hop_template, _, _, _, _, _, judge_answer_official) = _get_pipeline_helpers()

    # ── Main loop ─────────────────────────────────────────────────────────────
    try:
        from tqdm import tqdm
        pbar = tqdm(ids, unit="ex", file=sys.stderr, dynamic_ncols=True)
    except ImportError:
        pbar = iter(ids)

    retrieve_k = max(args.topk, args.expand_topk, 3)

    n_skip = 0
    n_examples = 0
    n_k_match = 0
    run_t0 = time.perf_counter()
    for eid in pbar:
        hidden_cache: dict[tuple[int, str], Any] = {}
        mrow        = records[eid]
        sub_questions = decompose_idx.get(eid)
        if not sub_questions:
            n_skip += 1
            continue
        sub_questions = prepare_sub_questions(
            sub_questions, decompose_mode=args.decompose_mode
        )
        if not sub_questions:
            n_skip += 1
            continue

        paragraphs  = mrow.get("paragraphs") or []
        gold_decomp = mrow.get("question_decomposition") or []
        q_main      = (mrow.get("question") or "").strip()
        K           = len(sub_questions)  # pred hop count (BART cap); LR artifacts keyed by (K, j)
        K_gold      = len(gold_decomp)
        k_match     = K == K_gold
        n_examples += 1
        if k_match:
            n_k_match += 1

        # Per-method running state: prior answers + completed hop steps for prefix
        state: dict[str, dict] = {
            m: {"prior": [], "hop_steps": []}   # hop_steps: [(expanded_q, ev_text)]
            for m in ranker_methods
        }

        ex_ranks:        dict[str, list[int | None]]   = {m: [] for m in ranker_methods}
        ex_sel_correct:  dict[str, list[bool | None]]  = {m: [] for m in ranker_methods}
        ex_gate_fired:   dict[str, list[bool]]         = {m: [] for m in ranker_methods if m in GATED}
        ex_oracle_pool:  dict[str, dict[int, list[bool]]] = {
            m: {k: [] for k in oracle_ks} for m in ranker_methods
        }
        hop_results: list[dict] = []

        # Retrieval cache: key by (retriever, query) so cosine/colbert don't mix
        _retrieval_cache: dict[tuple[str, str], list[tuple[dict, float]]] = {}

        def get_candidates(query: str, retriever: str) -> list[tuple[dict, float]]:
            key = (retriever, query)
            if key not in _retrieval_cache:
                try:
                    _retrieval_cache[key] = retrieve_candidates(
                        query,
                        paragraphs,
                        retrieve_k,
                        retriever=retriever,
                        cos_model=args.cos_model,
                        colbert_model=args.colbert_model,
                    )
                except Exception as e:
                    print(f"  [warn] {retriever} retrieval failed id={eid}: {e}",
                          file=sys.stderr)
                    _retrieval_cache[key] = []
            return _retrieval_cache[key]

        for hop_j, raw_sq in enumerate(sub_questions, start=1):
            gj = hop_j - 1
            has_gold_hop = gj < len(gold_decomp)
            if args.decompose_mode == "gt" and not has_gold_hop:
                continue

            gold_pi: int | None = None
            gold_para = None
            gold_title = ""
            if has_gold_hop:
                raw_gold_pi = gold_decomp[gj].get("paragraph_support_idx")
                if raw_gold_pi is None:
                    if args.decompose_mode == "gt":
                        continue
                    has_gold_hop = False
                else:
                    try:
                        gold_pi = int(raw_gold_pi)
                    except (TypeError, ValueError):
                        if args.decompose_mode == "gt":
                            continue
                        has_gold_hop = False
                        gold_pi = None
                    else:
                        gold_para = next(
                            (p for p in paragraphs if int(p.get("idx", -1)) == gold_pi),
                            None,
                        )
                        gold_title = (gold_para.get("title") or "") if gold_para else ""

            hop_row: dict = {
                "hop":           hop_j,
                "raw_subq":      raw_sq,
                "gold_para_idx": gold_pi,
                "gold_title":    gold_title,
                "has_gold_hop":  has_gold_hop,
            }

            for method in ranker_methods:
                hop_t0 = time.perf_counter()
                hop_retrieval_sec = 0.0
                hop_gate_sec = 0.0
                hop_llm_select_sec = 0.0
                hop_llm_answer_sec = 0.0
                ms = state[method]

                # ── 1. Expand sub-question with prior answers ─────────────────
                expanded_qj = expand_hop_template(raw_sq, ms["prior"])

                # ── 2. Build trace prefix for LR gate ─────────────────────────
                prefix_before = build_trace_prefix(q_main, ms["hop_steps"])

                # ── 3. Retrieve (per retriever; cached within example) ──────
                retriever = retriever_for_method(method)
                t_ret0 = time.perf_counter()
                candidates_all = get_candidates(expanded_qj, retriever)
                hop_retrieval_sec += time.perf_counter() - t_ret0
                if not candidates_all:
                    continue

                # ── 4. Oracle coverage (per method) ───────────────────────────
                if "oracle" in args.methods and has_gold_hop and gold_pi is not None:
                    for ok in oracle_ks:
                        pool    = candidates_all[:ok]
                        in_pool = find_gold_rank(pool, gold_pi) is not None
                        oracle_accum[method][ok].update(1 if in_pool else None)
                        ex_oracle_pool[method][ok].append(in_pool)

                # ── 5. Method-specific top-3 + gate logic ────────────────────
                gate_fired = False
                top3_initial_ab_scores: list[float] | None = None

                if method in ("baseline", "colbert"):
                    top3: list[tuple[dict, float]] = candidates_all[:3]

                elif method == "lr_rerank":
                    t_gate0 = time.perf_counter()
                    try:
                        scored_full = compute_abnormal_scores(
                            candidates=candidates_all[:args.topk],
                            prefix_before_hop=prefix_before,
                            hop_j=hop_j,
                            expanded_qj=expanded_qj,
                            K=K,
                            artifacts=artifacts,
                            gate_artifact_mode=args.gate_artifact_mode,
                            model=gate_model,
                            tokenizer=gate_tokenizer,
                            layer=args.layer,
                            hidden_cache=hidden_cache,
                        )
                        top3 = rerank_by_final_score(scored_full, args.lambda_lr)[:3]
                    except Exception as e:
                        print(f"  [warn] lr_rerank failed id={eid} hop={hop_j}: {e}", file=sys.stderr)
                        top3 = candidates_all[:3]
                    hop_gate_sec += time.perf_counter() - t_gate0

                else:  # gated_rule_a / gated_rule_b
                    t_gate0 = time.perf_counter()
                    try:
                        scored3 = compute_abnormal_scores(
                            candidates=candidates_all[:3],
                            prefix_before_hop=prefix_before,
                            hop_j=hop_j,
                            expanded_qj=expanded_qj,
                            K=K,
                            artifacts=artifacts,
                            gate_artifact_mode=args.gate_artifact_mode,
                            model=gate_model,
                            tokenizer=gate_tokenizer,
                            layer=args.layer,
                            hidden_cache=hidden_cache,
                        )
                    except Exception as e:
                        print(f"  [warn] gate score failed id={eid} hop={hop_j}: {e}", file=sys.stderr)
                        scored3 = [(p, s, 0.0) for p, s in candidates_all[:3]]

                    ab_top3 = [ab for _, _, ab in scored3]
                    # Use per-(K,j) calibrated threshold from artifact (fallback to CLI arg)
                    j_art = hop_j - 1
                    art = get_gate_artifact(artifacts, args.gate_artifact_mode, K, j_art) or {}
                    art_thresh = art.get("threshold", args.abnormal_threshold)
                    if method == "gated_rule_a":
                        gate_fired = ab_top3[0] > art_thresh
                    else:  # gated_rule_b
                        gate_fired = sum(ab > art_thresh for ab in ab_top3) >= 2

                    gate_accum[method].update_hop(gate_fired)
                    ex_gate_fired[method].append(gate_fired)

                    if gate_fired:
                        try:
                            scored_full = compute_abnormal_scores(
                                candidates=candidates_all[:args.expand_topk],
                                prefix_before_hop=prefix_before,
                                hop_j=hop_j,
                                expanded_qj=expanded_qj,
                                K=K,
                                artifacts=artifacts,
                                gate_artifact_mode=args.gate_artifact_mode,
                                model=gate_model,
                                tokenizer=gate_tokenizer,
                                layer=args.layer,
                                hidden_cache=hidden_cache,
                            )
                            top3 = rerank_by_final_score(scored_full, args.lambda_lr)[:3]
                        except Exception as e:
                            print(f"  [warn] expand rerank failed id={eid} hop={hop_j}: {e}",
                                  file=sys.stderr)
                            top3 = candidates_all[:3]
                    else:
                        top3 = [(p, s) for p, s, _ in scored3]
                    top3_initial_ab_scores = [round(ab, 4) for _, _, ab in scored3]
                    hop_gate_sec += time.perf_counter() - t_gate0

                # ── 6. Retrieval metrics ───────────────────────────────────────
                if has_gold_hop and gold_pi is not None:
                    gold_rank = find_gold_rank(top3, gold_pi)
                    accum[method]["all"].update(gold_rank)
                    accum[method][K].update(gold_rank)
                    accum_hop[method][(K, hop_j - 1)].update(gold_rank)
                else:
                    gold_rank = None

                ex_ranks[method].append(gold_rank)

                method_hop_row: dict = {
                    "expanded_subq":  expanded_qj,
                    "top3_para_ids":  [int(p.get("idx", -1)) for p, _ in top3],
                    "top3_scores":    [round(s, 4) for _, s in top3],
                    "gold_rank":      gold_rank,
                    "gold_in_top3":   gold_rank is not None and gold_rank <= 3,
                }
                if method in GATED:
                    method_hop_row["gate_fired"] = gate_fired
                    if top3_initial_ab_scores is not None:
                        method_hop_row["top3_initial_ab_scores"] = top3_initial_ab_scores

                # ── 7–8. LLM selection + short answer (or concat top-3) ─────────
                top3_paras = [p for p, _ in top3]
                concat_mode = args.selection_mode == "concat_answer"

                if with_llm and llama_model is not None and concat_mode:
                    t_ans0 = time.perf_counter()
                    concat_tokens = args.max_new_tokens_select + args.max_new_tokens_answer
                    try:
                        sub_ans, choice = llm_concat_answer(
                            expanded_qj, top3_paras,
                            llama_model, llama_tokenizer,
                            concat_tokens,
                        )
                    except Exception as e:
                        print(f"  [warn] concat_answer failed id={eid} hop={hop_j}: {e}",
                              file=sys.stderr)
                        sub_ans, choice = "", 0
                    hop_llm_answer_sec += time.perf_counter() - t_ans0

                    chosen_para = top3_paras[choice]
                    if has_gold_hop and gold_pi is not None:
                        sel_correct = int(chosen_para.get("idx", -1)) == gold_pi
                        sel_accum[method]["all"].update(sel_correct)
                        sel_accum[method][K].update(sel_correct)
                        sel_accum_hop[method][(K, hop_j - 1)].update(sel_correct)
                    else:
                        sel_correct = None
                    ex_sel_correct[method].append(sel_correct)
                    method_hop_row["selection"] = {
                        "mode":           "concat_answer",
                        "choice":         choice,
                        "chosen_para_id": int(chosen_para.get("idx", -1)),
                        "correct":        sel_correct,
                    }
                    method_hop_row["sub_answer"] = sub_ans

                elif with_llm and llama_model is not None:
                    t_sel0 = time.perf_counter()
                    try:
                        choice = llm_select_passage(
                            top3_paras, expanded_qj,
                            llama_model, llama_tokenizer,
                            args.max_new_tokens_select,
                        )
                    except Exception as e:
                        print(f"  [warn] selection failed id={eid} hop={hop_j}: {e}", file=sys.stderr)
                        choice = 0
                    hop_llm_select_sec += time.perf_counter() - t_sel0

                    chosen_para = top3_paras[choice]
                    if has_gold_hop and gold_pi is not None:
                        sel_correct = int(chosen_para.get("idx", -1)) == gold_pi
                        sel_accum[method]["all"].update(sel_correct)
                        sel_accum[method][K].update(sel_correct)
                        sel_accum_hop[method][(K, hop_j - 1)].update(sel_correct)
                    else:
                        sel_correct = None
                    ex_sel_correct[method].append(sel_correct)
                    method_hop_row["selection"] = {
                        "mode":           "select",
                        "choice":         choice,
                        "chosen_para_id": int(chosen_para.get("idx", -1)),
                        "correct":        sel_correct,
                    }

                    t_ans0 = time.perf_counter()
                    try:
                        sub_ans = llm_short_answer(
                            expanded_qj, chosen_para,
                            llama_model, llama_tokenizer,
                            args.max_new_tokens_answer,
                        )
                    except Exception as e:
                        print(f"  [warn] short_ans failed id={eid} hop={hop_j}: {e}", file=sys.stderr)
                        sub_ans = ""
                    hop_llm_answer_sec += time.perf_counter() - t_ans0
                    method_hop_row["sub_answer"] = sub_ans

                else:
                    # No LLM: default to top-1 for metrics
                    chosen_para = top3_paras[0]
                    if has_gold_hop and gold_pi is not None:
                        sel_correct = int(chosen_para.get("idx", -1)) == gold_pi
                        sel_accum[method]["all"].update(sel_correct)
                        sel_accum[method][K].update(sel_correct)
                        sel_accum_hop[method][(K, hop_j - 1)].update(sel_correct)
                    else:
                        sel_correct = None
                    ex_sel_correct[method].append(sel_correct)
                    sub_ans = ""

                ms["prior"].append(sub_ans)
                ev_text = (chosen_para.get("paragraph_text") or "").strip()
                ms["hop_steps"].append((expanded_qj, ev_text))

                hop_row[method] = method_hop_row
                timing_accum[method].add_hop(
                    total=time.perf_counter() - hop_t0,
                    retrieval=hop_retrieval_sec,
                    gate=hop_gate_sec,
                    llm_select=hop_llm_select_sec,
                    llm_answer=hop_llm_answer_sec,
                )

            hop_results.append(hop_row)

        # ── Case / K-match chain accumulators (gold-hop aligned) ──────────────
        per_method_gold_ranks: dict[str, list[int | None]] = {}
        per_method_gold_sels: dict[str, list[bool | None]] = {}
        per_method_case_r1: dict[str, bool] = {}
        per_method_case_r3: dict[str, bool] = {}
        for m in ranker_methods:
            gold_ranks = gold_hop_ranks(hop_results, m, K_gold)
            gold_sels = gold_hop_selections(hop_results, m, K_gold)
            per_method_gold_ranks[m] = gold_ranks
            per_method_gold_sels[m] = gold_sels

            case_recall_accum[m]["all"].update(gold_ranks)
            if K_gold in case_recall_accum[m]:
                case_recall_accum[m][K_gold].update(gold_ranks)

            cr1, cr3 = case_recall_flags(gold_ranks)
            per_method_case_r1[m] = cr1
            per_method_case_r3[m] = cr3

            if k_match:
                chain_accum_k_match[m]["all"].update(gold_ranks)
                chain_accum_k_match[m][K].update(gold_ranks)
                sel_chain_accum_k_match[m]["all"].update(gold_sels)
                sel_chain_accum_k_match[m][K].update(gold_sels)

        # ── Update chain / answer / oracle chain accumulators ─────────────────
        for m in ranker_methods:
            if ex_ranks[m]:
                chain_accum[m]["all"].update(ex_ranks[m])
                chain_accum[m][K].update(ex_ranks[m])
            if ex_sel_correct[m]:
                sel_chain_accum[m]["all"].update(ex_sel_correct[m])
                sel_chain_accum[m][K].update(ex_sel_correct[m])
            if m in GATED and ex_gate_fired.get(m):
                gate_accum[m].update_example(any(ex_gate_fired[m]))
            if "oracle" in args.methods:
                for ok in oracle_ks:
                    if ex_oracle_pool[m][ok]:
                        all_in = all(ex_oracle_pool[m][ok])
                        oracle_chain[m][ok].update([1 if all_in else None])

            # Final answer EM/F1
            predicted = state[m]["prior"][-1] if state[m]["prior"] else ""
            if with_llm and predicted and judge_answer_official is not None:
                try:
                    em, f1 = judge_answer_official(predicted, mrow)
                except Exception:
                    em, f1 = None, None
                ans_accum[m]["all"].update(em, f1)
                ans_accum[m][K].update(em, f1)
            else:
                em, f1 = None, None

        # ── Per-case JSONL ────────────────────────────────────────────────────
        # Collect per-method answer EM/F1 for shard merging.
        per_method_em: dict[str, int | None] = {}
        per_method_f1: dict[str, float | None] = {}
        for m in ranker_methods:
            predicted = state[m]["prior"][-1] if state[m]["prior"] else ""
            if with_llm and predicted and judge_answer_official is not None:
                try:
                    em_m, f1_m = judge_answer_official(predicted, mrow)
                except Exception:
                    em_m, f1_m = None, None
            else:
                em_m, f1_m = None, None
            per_method_em[m] = em_m
            per_method_f1[m] = f1_m

        case_row: dict = {
            "id":       eid,
            "question": q_main,
            "K_pred":   K,
            "K_gold":   K_gold,
            "K_match":  k_match,
            "chain_recall1": {
                m: all(r == 1 for r in ex_ranks[m]) if ex_ranks[m] else None
                for m in ranker_methods
            },
            "chain_recall3": {
                m: all(r is not None and r <= 3 for r in ex_ranks[m]) if ex_ranks[m] else None
                for m in ranker_methods
            },
            "chain_recall1_k_match": {
                m: per_method_case_r1[m] if k_match else None for m in ranker_methods
            },
            "chain_recall3_k_match": {
                m: per_method_case_r3[m] if k_match else None for m in ranker_methods
            },
            "case_recall1": {m: per_method_case_r1[m] for m in ranker_methods},
            "case_recall3": {m: per_method_case_r3[m] for m in ranker_methods},
            "gold_hop_ranks": per_method_gold_ranks,
            "chain_selection": {
                m: all(c is True for c in ex_sel_correct[m]) if ex_sel_correct[m] else None
                for m in ranker_methods
            },
            "chain_selection_k_match": {
                m: all(c is True for c in per_method_gold_sels[m])
                if k_match and per_method_gold_sels[m]
                else None
                for m in ranker_methods
            },
            "predicted_answers": {
                m: state[m]["prior"][-1] if state[m]["prior"] else "" for m in ranker_methods
            },
            "answer_em":  per_method_em,
            "answer_f1":  per_method_f1,
            "hop_results": hop_results,
        }
        if any(m in GATED for m in ranker_methods):
            case_row["gate_fired_any"] = {
                m: any(ex_gate_fired[m]) if ex_gate_fired.get(m) else None
                for m in ranker_methods if m in GATED
            }
        cases_fp.write(json.dumps(case_row, ensure_ascii=False) + "\n")
        cases_fp.flush()
        for m in ranker_methods:
            timing_accum[m].add_example()

    cases_fp.close()
    run_wall_sec = time.perf_counter() - run_t0

    # ── Aggregate results ──────────────────────────────────────────────────────
    def transition_label(j: int, K: int) -> str:
        if j == 0:
            return "Q->E1"
        if j == K:
            return f"E{K}->Final"
        return f"E{j}->E{j+1}"

    config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    results: dict = {
        "config": config_dict,
        "output_dir": str(args.out_dir),
        "n_skip": n_skip,
        "n_examples": n_examples,
        "n_k_match": n_k_match,
        "k_match_rate": round(n_k_match / n_examples, 4) if n_examples else 0.0,
    }

    for m in ranker_methods:
        mres: dict = {
            "overall":       accum[m]["all"].result(),
            "by_K":          {K: accum[m][K].result() for K in K_BUCKETS},
            "by_hop": {
                f"K{K}_j{j}_{transition_label(j, K)}": accum_hop[m][(K, j)].result()
                for (K, j) in sorted(accum_hop[m].keys())
            },
            "chain_overall": chain_accum[m]["all"].result(),
            "chain_by_K":    {K: chain_accum[m][K].result() for K in K_BUCKETS},
            "chain_overall_k_match": chain_accum_k_match[m]["all"].result(),
            "chain_by_K_k_match": {
                K: chain_accum_k_match[m][K].result() for K in K_BUCKETS
            },
            "case_recall_overall": case_recall_accum[m]["all"].result(),
            "case_recall_by_K_gold": {
                K: case_recall_accum[m][K].result() for K in K_BUCKETS
            },
            "selection_overall":       sel_accum[m]["all"].result(),
            "selection_by_K":          {K: sel_accum[m][K].result() for K in K_BUCKETS},
            "selection_by_hop": {
                f"K{K}_j{j}_{transition_label(j, K)}": sel_accum_hop[m][(K, j)].result()
                for (K, j) in sorted(sel_accum_hop[m].keys())
            },
            "chain_selection_overall": sel_chain_accum[m]["all"].result(),
            "chain_selection_by_K":    {K: sel_chain_accum[m][K].result() for K in K_BUCKETS},
            "chain_selection_overall_k_match": sel_chain_accum_k_match[m]["all"].result(),
            "chain_selection_by_K_k_match": {
                K: sel_chain_accum_k_match[m][K].result() for K in K_BUCKETS
            },
            "answer_overall":          ans_accum[m]["all"].result(),
            "answer_by_K":             {K: ans_accum[m][K].result() for K in K_BUCKETS},
        }
        if m in GATED:
            mres["gate_stats"] = gate_accum[m].result()
        if "oracle" in args.methods:
            mres["oracle_coverage"] = {
                f"top{ok}": {
                    "hop_level":   oracle_accum[m][ok].result(),
                    "chain_level": oracle_chain[m][ok].result(),
                }
                for ok in oracle_ks
            }
        results[m] = mres

    results["timing"] = {
        "wall_clock_sec": round(run_wall_sec, 2),
        "wall_clock_hr": round(run_wall_sec / 3600, 3),
        "note": "Per-method times are cumulative over hops (combined multi-method run). "
                "For isolated wall time, run each --methods alone.",
        "by_method": {m: timing_accum[m].result() for m in ranker_methods},
    }

    out_path = args.out_dir / f"retrieval_exp_{args.split}{tag_suffix}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nMetrics saved to  {out_path}", flush=True)
    print(f"Cases saved to    {args.out_cases}", flush=True)

    # ── Print summary ──────────────────────────────────────────────────────────
    W = 100
    print("\n" + "=" * W)
    print("RETRIEVAL  (hop-level)")
    print(f"{'METHOD':<22}  {'R@1':>7}  {'R@3':>7}  {'MRR':>7}  {'n_hops':>7}")
    print("-" * W)
    for m in ranker_methods:
        ov = results[m]["overall"]
        print(f"  {m:<20}  {ov['recall@1']:>7.4f}  {ov['recall@3']:>7.4f}  "
              f"{ov['mrr']:>7.4f}  {ov['n_hops']:>7}")
    print("=" * W)

    print("\n" + "=" * W)
    print("CHAIN RETRIEVAL  (all pred hops must recall gold)")
    print(f"{'METHOD':<22}  {'Chain R@1':>10}  {'Chain R@3':>10}  {'n_examples':>11}")
    print("-" * W)
    for m in ranker_methods:
        ch = results[m]["chain_overall"]
        print(f"  {m:<20}  {ch['chain_recall@1']:>10.4f}  {ch['chain_recall@3']:>10.4f}  "
              f"{ch['n_examples']:>11}")
    print("=" * W)

    print("\n" + "=" * W)
    print(f"CASE RECALL  (all {results['n_examples']} examples; gold hops only)")
    print(f"{'METHOD':<22}  {'Case R@1':>10}  {'Case R@3':>10}  {'n_examples':>11}")
    print("-" * W)
    for m in ranker_methods:
        cr = results[m]["case_recall_overall"]
        print(f"  {m:<20}  {cr['case_recall@1']:>10.4f}  {cr['case_recall@3']:>10.4f}  "
              f"{cr['n_examples']:>11}")
    print("=" * W)

    print("\n" + "=" * W)
    print(f"CHAIN / CASE (K-match only: K_pred==K_gold, n={results['n_k_match']}, "
          f"rate={results['k_match_rate']:.3f})")
    print(f"{'METHOD':<22}  {'ChnR@1':>8}  {'ChnR@3':>8}  {'CaseR@3':>8}  {'ChnSel':>8}")
    print("-" * W)
    for m in ranker_methods:
        ch = results[m]["chain_overall_k_match"]
        cs = results[m]["chain_selection_overall_k_match"]
        print(f"  {m:<20}  {ch['chain_recall@1']:>8.4f}  {ch['chain_recall@3']:>8.4f}  "
              f"{ch['chain_recall@3']:>8.4f}  {cs['chain_selection_acc']:>8.4f}")
    print("=" * W)

    print("\n" + "=" * W)
    print("SELECTION  (hop-level: did LLM pick gold?)")
    print(f"{'METHOD':<22}  {'Sel Acc':>9}  {'Chain Sel':>10}  {'Answer EM':>10}  {'Answer F1':>10}")
    print("-" * W)
    for m in ranker_methods:
        sa  = results[m]["selection_overall"]
        cs  = results[m]["chain_selection_overall"]
        ans = results[m]["answer_overall"]
        print(f"  {m:<20}  {sa['selection_acc']:>9.4f}  {cs['chain_selection_acc']:>10.4f}  "
              f"{ans['answer_em']:>10.4f}  {ans['answer_f1']:>10.4f}")
    print("=" * W)

    for m in ranker_methods:
        if m in GATED:
            gs = results[m]["gate_stats"]
            print(f"\nGate [{m}]: "
                  f"hop_trigger={gs['hop_trigger_rate']:.3f} "
                  f"({gs['n_triggered_hops']}/{gs['n_hops']})  "
                  f"ex_trigger={gs['example_trigger_rate']:.3f} "
                  f"({gs['n_examples_any_triggered']}/{gs['n_examples']})")

    if "oracle" in args.methods:
        print(f"\nOracle coverage (per method, using that method's own retrieval query):")
        for m in ranker_methods:
            print(f"  [{m}]")
            for ok in oracle_ks:
                ov = results[m]["oracle_coverage"][f"top{ok}"]
                print(f"    top-{ok:<2}: hop={ov['hop_level']['recall@1']:.4f}  "
                      f"chain={ov['chain_level']['chain_recall@1']:.4f}")

    for m in ranker_methods:
        print(f"\nBy K — retrieval + selection + answer ({m}):")
        for K in K_BUCKETS:
            bk  = results[m]["by_K"][K]
            ch  = results[m]["chain_by_K"][K]
            sk  = results[m]["selection_by_K"][K]
            csk = results[m]["chain_selection_by_K"][K]
            ans = results[m]["answer_by_K"][K]
            if bk["n_hops"] == 0:
                continue
            print(f"  K={K}: R@1={bk['recall@1']:.3f}  Chain_R@1={ch['chain_recall@1']:.3f}  "
                  f"Sel={sk['selection_acc']:.3f}  ChainSel={csk['chain_selection_acc']:.3f}  "
                  f"EM={ans['answer_em']:.3f}  n={ch['n_examples']}")

    if ranker_methods:
        print(f"\nBy (K, hop) — retrieval + selection:")
        for m in ranker_methods:
            print(f"  [{m}]")
            for key in sorted(results[m]["by_hop"]):
                hm  = results[m]["by_hop"][key]
                sm  = results[m]["selection_by_hop"].get(key, {})
                if hm.get("n_hops", 0) == 0:
                    continue
                line = (f"    {key:<26}: R@1={hm['recall@1']:.3f}"
                        f"  R@3={hm['recall@3']:.3f}"
                        f"  n={hm['n_hops']}")
                if sm:
                    line += f"  Sel={sm.get('selection_acc', 0):.3f}"
                print(line)

    if results.get("timing"):
        print("\n" + "=" * W)
        print("TIMING  (per-method cumulative; combined multi-method run)")
        print(f"{'METHOD':<22}  {'total_hr':>9}  {'sec/hop':>9}  {'sec/ex':>9}")
        print("-" * W)
        for m in ranker_methods:
            t = results["timing"]["by_method"][m]
            print(f"  {m:<20}  {t['total_hr']:>9.3f}  {t['sec_per_hop'] or 0:>9.3f}  "
                  f"{t['sec_per_example'] or 0:>9.1f}")
        print(f"  Wall clock (all methods): {results['timing']['wall_clock_hr']:.3f} hr")

    print(f"\n(skipped {n_skip} examples)")


if __name__ == "__main__":
    main()
