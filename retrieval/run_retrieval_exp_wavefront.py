#!/usr/bin/env python3
"""Wavefront retrieval experiment: batched hidden-state gate + batched vLLM generation.

This is an experimental rewrite of run_retrieval_exp.py that keeps the same
retrieval/gate/evaluation semantics, but changes execution order:

  1. Process all examples hop-by-hop instead of one example at a time.
  2. Use transformers only for LR detector hidden states.
  3. Use vLLM batch generation for passage selection and hop answers.
  4. After all hops, run a batched final reader (re-read question + direct answer).

The original script is intentionally left untouched.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from run_retrieval_exp import (
    ARTIFACTS_DIR,
    HOTPOT_DEV_FILE,
    POOLED_ARTIFACTS_DIR,
    K_BUCKETS,
    MUSIQUE_DIR,
    TWOWIKI_DEV_FILE,
    AnswerAccum,
    CaseRecallAccum,
    ChainAccum,
    GateAccum,
    MetricAccum,
    SelectionAccum,
    SelectionChainAccum,
    build_trace_prefix,
    case_recall_flags,
    escape_double_quotes,
    find_gold_rank,
    get_gate_artifact,
    gold_hop_ranks,
    gold_hop_selections,
    load_artifacts,
    load_dataset_records,
    load_decompose_index,
    normalize_short_answer,
    prepare_sub_questions,
    resolve_decompose_file,
    resolve_output_dir,
    retrieve_candidates,
    retriever_for_method,
    rerank_by_final_score,
    sanitize_run_tag,
    select_example_ids,
)


GATED = {"gated_rule_a", "gated_rule_b"}

FINAL_ANSWER_MARKER = "Final answer:"
FINAL_ANSWER_RE = re.compile(
    r"^\s*(?:###\s*)?Final\s+[Aa]nswer\s*:\s*(.*?)\s*$",
    re.MULTILINE,
)

FINAL_READER_SYSTEM_PROMPT = (
    "You are a precise multi-hop question answering reader. "
    "Use only the provided reasoning trace. "
    "After re-reading the original question, give a direct short answer on exactly "
    f"one line: {FINAL_ANSWER_MARKER} <short answer>. "
    "Do not write anything after the final answer line."
)


def build_reasoning_context(
    q_main: str,
    hop_steps: list[tuple[str, str]],
    prior: list[str],
) -> str:
    """Build cumulative reasoning trace for LLM selection / answer prompts."""
    lines = [f"Original question:\n{q_main}"]
    if hop_steps:
        lines.append("\nPrevious reasoning steps:")
        for i, ((subq, ev), ans) in enumerate(zip(hop_steps, prior), start=1):
            lines.append(f"Step {i}:")
            lines.append(f"Subquestion: {subq}")
            lines.append(f"Selected evidence: {ev}")
            lines.append(f"Answer: {ans}")
            lines.append("")
    return "\n".join(lines).strip()


def prompt_select_passage_with_context(
    q_main: str,
    hop_steps: list[tuple[str, str]],
    prior: list[str],
    expanded_q: str,
    candidates: list[dict[str, Any]],
) -> str:
    ctx = build_reasoning_context(q_main, hop_steps, prior)
    lines = [
        ctx,
        "",
        "Current subquestion:",
        expanded_q,
        "",
        "Choose exactly ONE passage below that can DIRECTLY answer the current subquestion.",
        "Do not pick the passage that is merely related or background context.",
        'Reply with JSON only: {"choice": <0|1|2>} matching the passage index.',
        "",
        "Candidate passages:",
    ]
    for i, p in enumerate(candidates):
        t = (p.get("paragraph_text") or "").strip()
        title = (p.get("title") or "").strip()
        lines.append(f"[{i}] title: {title}\npassage: {t[:1200]}")
        lines.append("")
    return "\n".join(lines)


def prompt_short_answer_with_context(
    q_main: str,
    hop_steps: list[tuple[str, str]],
    prior: list[str],
    expanded_q: str,
    passage: dict[str, Any],
) -> str:
    ctx = build_reasoning_context(q_main, hop_steps, prior)
    t = (passage.get("paragraph_text") or "").strip()
    title = (passage.get("title") or "").strip()
    return (
        f"{ctx}\n\n"
        "Current subquestion:\n"
        f"{expanded_q}\n\n"
        f"Selected evidence title: {title}\n"
        f"Selected evidence:\n{t}\n\n"
        "Read the selected evidence and answer the current subquestion with a SHORT span or phrase only "
        "(no full sentences or explanation). If numeric, output the number. "
        "If the passage does not contain the answer, output exactly: NA\n\n"
        "Answer:"
    )


def build_final_reader_cot_prompt(
    q_main: str,
    hop_steps: list[tuple[str, str]],
    prior: list[str],
) -> str:
    lines: list[str] = []
    if hop_steps:
        lines.append("Reasoning trace:")
        for i, ((subq, ev), ans) in enumerate(zip(hop_steps, prior), start=1):
            lines.append(f"Step {i}:")
            lines.append(f"Subquestion: {subq}")
            lines.append(f"Selected evidence: {ev}")
            lines.append(f"Answer: {ans}")
            lines.append("")
    else:
        lines.append("Reasoning trace: [empty]")
        lines.append("")
    lines.extend(
        [
            "Re-read the original question before answering:",
            q_main,
            "",
            "Answer the original question directly using only the reasoning trace above.",
            "Give a short span or phrase only (entity, date, number, or yes/no).",
            "For yes/no questions, answer yes or no only.",
            "",
            "Output exactly one line:",
            f"{FINAL_ANSWER_MARKER} <short answer>",
        ]
    )
    return "\n".join(lines)


def parse_final_answer_from_cot(raw: str, *, fallback: str = "") -> str:
    text = (raw or "").strip()
    if not text:
        return fallback or "NA"
    matches = FINAL_ANSWER_RE.findall(text)
    if matches:
        ans = matches[-1].strip()
        ans = re.sub(r"^[`*_]+|[`*_]+$", "", ans).strip().strip('"').rstrip(".")
        return ans or fallback or "NA"
    if fallback:
        return fallback
    return "NA"


def _get_pipeline_helpers():
    from run_retrieval_exp import _get_pipeline_helpers as original_helpers

    return original_helpers()


@dataclass
class MethodState:
    prior: list[str]
    hop_steps: list[tuple[str, str]]
    ex_ranks: list[int | None]
    ex_sel_correct: list[bool | None]
    ex_gate_fired: list[bool]
    ex_oracle_pool: dict[int, list[bool]]
    final_answer: str = ""
    final_cot: str = ""


@dataclass
class ExampleState:
    eid: str
    row: dict[str, Any]
    q_main: str
    paragraphs: list[dict[str, Any]]
    sub_questions: list[str]
    gold_decomp: list[dict[str, Any]]
    K: int
    K_gold: int
    k_match: bool
    hop_results: list[dict[str, Any]]
    methods: dict[str, MethodState]


@dataclass
class HopContext:
    ex: ExampleState
    method: str
    hop_j: int
    raw_sq: str
    expanded_q: str
    prefix_before: str
    candidates_all: list[tuple[dict[str, Any], float]]
    top3: list[tuple[dict[str, Any], float]] | None = None
    gate_fired: bool = False
    top3_initial_ab_scores: list[float] | None = None
    gold_rank: int | None = None


@dataclass
class GateRequest:
    req_id: int
    ctx: HopContext
    candidates: list[tuple[dict[str, Any], float]]


class BatchVllmGenerator:
    def __init__(
        self,
        model: str,
        *,
        dtype: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int | None,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        kwargs: dict[str, Any] = {
            "model": model,
            "dtype": dtype,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        if max_model_len:
            kwargs["max_model_len"] = max_model_len
        self.llm = LLM(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        self.SamplingParams = SamplingParams

    def render(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int,
        *,
        desc: str = "vllm",
    ) -> list[str]:
        if not prompts:
            return []
        print(f"  {desc}: {len(prompts)} prompts", flush=True)
        sampling = self.SamplingParams(temperature=0.0, max_tokens=max_tokens)
        rendered = [self.render(prompt) for prompt in prompts]
        outputs = self.llm.generate(rendered, sampling, use_tqdm=True)
        return [out.outputs[0].text.strip() for out in outputs]

    def generate_chat_batch(
        self,
        system_prompt: str,
        user_prompts: list[str],
        max_tokens: int,
        *,
        desc: str = "vllm-chat",
    ) -> list[str]:
        if not user_prompts:
            return []
        print(f"  {desc}: {len(user_prompts)} prompts", flush=True)
        sampling = self.SamplingParams(temperature=0.0, max_tokens=max_tokens)
        rendered = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in user_prompts
        ]
        outputs = self.llm.generate(rendered, sampling, use_tqdm=True)
        return [out.outputs[0].text.strip() for out in outputs]


def load_gate_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    device = args.gate_device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev_map = None if device == "cpu" else {"": device}
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {"dtype": dtype, "device_map": dev_map}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    return model, tokenizer


def render_gate_chat_text(tokenizer, text: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=False,
            tokenize=False,
        )
    except Exception:
        return text


def batch_last_hidden(
    *,
    texts: list[str],
    model,
    tokenizer,
    layer: int,
    batch_size: int,
    cache: dict[tuple[int, str], np.ndarray],
    desc: str | None = None,
) -> list[np.ndarray]:
    import torch

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    missing = [text for text in texts if (layer, text) not in cache]
    dev = next(model.parameters()).device
    batch_starts = range(0, len(missing), batch_size)
    if desc and missing and tqdm is not None:
        n_batches = (len(missing) + batch_size - 1) // batch_size
        batch_starts = tqdm(
            batch_starts,
            total=n_batches,
            desc=desc,
            unit="batch",
            file=sys.stderr,
            dynamic_ncols=True,
        )
    for start in batch_starts:
        chunk = missing[start : start + batch_size]
        if not chunk:
            continue
        rendered = [render_gate_chat_text(tokenizer, text) for text in chunk]
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
            raise ValueError(f"bad hidden_states len={len(hs) if hs else 0} layer={layer}")
        layer_h = hs[layer + 1]
        last_indices = (
            attention_mask.size(1)
            - 1
            - torch.flip(attention_mask, dims=[1]).argmax(dim=1)
        )
        batch_indices = torch.arange(layer_h.size(0), device=dev)
        hidden = layer_h[batch_indices, last_indices].float().cpu().numpy().astype(np.float32)
        for text, vec in zip(chunk, hidden):
            cache[(layer, text)] = vec
    return [cache[(layer, text)] for text in texts]


def score_gate_requests(
    requests: list[GateRequest],
    *,
    artifacts: dict,
    gate_artifact_mode: str,
    model,
    tokenizer,
    layer: int,
    hidden_batch_size: int,
    hidden_cache: dict[tuple[int, str], np.ndarray],
    desc: str | None = None,
) -> dict[int, list[tuple[dict[str, Any], float, float]]]:
    outputs: dict[int, list[tuple[dict[str, Any], float, float]]] = {}
    texts: list[str] = []
    owners: list[tuple[int, int | None]] = []
    metadata: dict[int, tuple[dict[str, Any], Any, str]] = {}

    for req in requests:
        ctx = req.ctx
        art = get_gate_artifact(artifacts, gate_artifact_mode, ctx.ex.K, ctx.hop_j - 1)
        if art is None:
            outputs[req.req_id] = [(p, s, 0.0) for p, s in req.candidates]
            continue
        pca = art["pca"]
        lr = art["lr"]
        metadata[req.req_id] = (pca, lr, ctx.prefix_before)
        texts.append(ctx.prefix_before)
        owners.append((req.req_id, None))
        for cand_idx, (para, _) in enumerate(req.candidates):
            para_text = (para.get("paragraph_text") or "").strip()
            text = (
                f'{ctx.prefix_before} Step {ctx.hop_j}: {ctx.expanded_q}'
                f' Evidence: "{escape_double_quotes(para_text)}"'
            )
            texts.append(text)
            owners.append((req.req_id, cand_idx))

    if texts:
        hiddens = batch_last_hidden(
            texts=texts,
            model=model,
            tokenizer=tokenizer,
            layer=layer,
            batch_size=hidden_batch_size,
            cache=hidden_cache,
            desc=desc,
        )
    else:
        hiddens = []

    prev_by_req: dict[int, np.ndarray] = {}
    cand_by_req: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for (req_id, cand_idx), hidden in zip(owners, hiddens):
        if cand_idx is None:
            prev_by_req[req_id] = hidden
        else:
            cand_by_req[req_id][cand_idx] = hidden

    req_by_id = {req.req_id: req for req in requests}
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    req_items = metadata.items()
    if desc and metadata and tqdm is not None:
        req_items = tqdm(
            list(metadata.items()),
            desc=f"{desc} score",
            unit="req",
            file=sys.stderr,
            dynamic_ncols=True,
        )
    for req_id, (pca, lr, _) in req_items:
        req = req_by_id[req_id]
        h_prev = prev_by_req[req_id]
        scored: list[tuple[dict[str, Any], float, float]] = []
        for cand_idx, (para, emb_score) in enumerate(req.candidates):
            h_j = cand_by_req[req_id][cand_idx]
            delta = (h_j - h_prev).astype("float64").reshape(1, -1)
            z = pca.transform(delta)
            abnormal = float(lr.predict_proba(z)[0, 1])
            scored.append((para, emb_score, abnormal))
        outputs[req_id] = scored
    return outputs


def init_example_state(
    eid: str,
    row: dict[str, Any],
    sub_questions: list[str],
    methods: list[str],
    oracle_ks: list[int],
) -> ExampleState:
    paragraphs = row.get("paragraphs") or []
    gold_decomp = row.get("question_decomposition") or []
    q_main = (row.get("question") or "").strip()
    hop_results: list[dict[str, Any]] = []
    for hop_j, raw_sq in enumerate(sub_questions, start=1):
        gj = hop_j - 1
        has_gold_hop = gj < len(gold_decomp)
        gold_pi: int | None = None
        gold_title = ""
        if has_gold_hop:
            raw_gold_pi = gold_decomp[gj].get("paragraph_support_idx")
            try:
                gold_pi = int(raw_gold_pi)
            except (TypeError, ValueError):
                gold_pi = None
                has_gold_hop = False
            if gold_pi is not None:
                gold_para = next(
                    (p for p in paragraphs if int(p.get("idx", -1)) == gold_pi),
                    None,
                )
                gold_title = (gold_para.get("title") or "") if gold_para else ""
        hop_results.append(
            {
                "hop": hop_j,
                "raw_subq": raw_sq,
                "gold_para_idx": gold_pi,
                "gold_title": gold_title,
                "has_gold_hop": has_gold_hop,
            }
        )
    method_states = {
        method: MethodState(
            prior=[],
            hop_steps=[],
            ex_ranks=[],
            ex_sel_correct=[],
            ex_gate_fired=[],
            ex_oracle_pool={k: [] for k in oracle_ks},
        )
        for method in methods
    }
    K = len(sub_questions)
    K_gold = len(gold_decomp)
    return ExampleState(
        eid=eid,
        row=row,
        q_main=q_main,
        paragraphs=paragraphs,
        sub_questions=sub_questions,
        gold_decomp=gold_decomp,
        K=K,
        K_gold=K_gold,
        k_match=K == K_gold,
        hop_results=hop_results,
        methods=method_states,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--dataset", choices=["musique", "2wiki", "hotpot"], default="musique")
    ap.add_argument("--musique-dir", type=Path, default=MUSIQUE_DIR)
    ap.add_argument("--twowiki-file", type=Path, default=TWOWIKI_DEV_FILE)
    ap.add_argument("--hotpot-file", type=Path, default=HOTPOT_DEV_FILE)
    ap.add_argument("--decompose-mode", choices=["gt", "bart_decompose"], default="bart_decompose")
    ap.add_argument("--decompose-file", type=Path, default=None)
    ap.add_argument("--gate-artifact-mode", choices=["nonpooled", "pooled"], default="pooled")
    ap.add_argument("--artifacts-dir", type=Path, default=None)
    ap.add_argument("--results-root", type=Path, default=Path(__file__).resolve().parent / "results")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--out-cases", type=Path, default=None)
    ap.add_argument("--methods", nargs="+",
                    choices=["baseline", "oracle", "lr_rerank", "gated_rule_a", "gated_rule_b"],
                    default=["baseline", "oracle", "lr_rerank", "gated_rule_a", "gated_rule_b"])
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--expand-topk", type=int, default=10)
    ap.add_argument("--lambda-lr", type=float, default=0.25)
    ap.add_argument("--abnormal-threshold", type=float, default=0.5)
    ap.add_argument("--cos-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--colbert-model", default="colbert-ir/colbertv2.0")
    ap.add_argument("--answerable-only", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--limit-per-k", type=int, default=0)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--gate-device", default="cuda:0")
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.35,
        help="vLLM GPU memory fraction. Lower this when gate model shares the same GPU.",
    )
    ap.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="vLLM max sequence length (0 = model default, often 131k). "
        "Capping this saves KV cache when gate shares the same GPU.",
    )
    ap.add_argument("--vllm-batch-size", type=int, default=128)
    ap.add_argument(
        "--hidden-batch-size",
        type=int,
        default=8,
        help="Batch size for gate hidden-state forward passes.",
    )
    ap.add_argument("--max-new-tokens-select", type=int, default=64)
    ap.add_argument("--max-new-tokens-answer", type=int, default=64)
    ap.add_argument(
        "--final-reader",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After all hops, run a batched final reader (re-read question + direct answer).",
    )
    ap.add_argument(
        "--max-new-tokens-final",
        type=int,
        default=128,
        help="Max tokens for final reader generation (direct answer line).",
    )
    ap.add_argument("--selection-mode", choices=["select"], default="select")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.artifacts_dir is None:
        args.artifacts_dir = (
            POOLED_ARTIFACTS_DIR
            if args.gate_artifact_mode == "pooled"
            else ARTIFACTS_DIR
        )
    args.decompose_file = resolve_decompose_file(
        args.decompose_mode, args.split, args.decompose_file, args.dataset
    )
    ranker_methods = [m for m in args.methods if m != "oracle"]
    if not ranker_methods:
        raise SystemExit("No ranker methods selected.")

    records = load_dataset_records(args)
    decompose_idx = load_decompose_index(args.decompose_file)
    ids = sorted(set(records) & set(decompose_idx))
    if args.answerable_only:
        ids = [eid for eid in ids if records[eid].get("answerable", True)]
    ids = select_example_ids(
        ids,
        decompose_idx,
        limit=args.limit,
        limit_per_k=args.limit_per_k,
        seed=args.sample_seed,
    )
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.num_shards > 1:
        ids = ids[args.shard_index :: args.num_shards]

    tag = args.run_tag or f"wavefront_{args.dataset}_{sanitize_run_tag('_'.join(args.methods))}"
    args.out_dir = resolve_output_dir(
        out_dir=args.out_dir,
        results_root=args.results_root,
        run_tag=tag,
        decompose_mode=args.decompose_mode,
        methods=args.methods,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
    if args.out_cases is None:
        args.out_cases = args.out_dir / f"retrieval_cases_{args.split}{tag_suffix}.jsonl"

    print(f"Output directory: {args.out_dir}")
    print(f"Dataset={args.dataset} examples={len(ids)} decompose={args.decompose_file}")
    vllm_max_len = args.max_model_len or None
    print(
        f"Methods={ranker_methods} vLLM gpu_memory_utilization={args.gpu_memory_utilization} "
        f"max_model_len={vllm_max_len or 'model_default'} gate_device={args.gate_device} "
        f"hidden_batch_size={args.hidden_batch_size} final_reader={args.final_reader}",
        flush=True,
    )

    artifacts: dict = {}
    needs_gate = any(m in ("lr_rerank", "gated_rule_a", "gated_rule_b") for m in ranker_methods)
    if needs_gate:
        artifacts = load_artifacts(args.artifacts_dir, args.gate_artifact_mode)
        print("Loading gate transformers model ...", flush=True)
        gate_model, gate_tokenizer = load_gate_model(args)
    else:
        gate_model = gate_tokenizer = None

    print("Loading vLLM generator ...", flush=True)
    generator = BatchVllmGenerator(
        args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=vllm_max_len,
    )

    (
        expand_hop_template,
        _prompt_select_passage,
        parse_json_choice,
        _prompt_short_answer,
        _prompt_concat_answer,
        _parse_json_answer_choice,
        judge_answer_official,
    ) = _get_pipeline_helpers()

    oracle_ks = [3, 5, 10, 20]
    examples: list[ExampleState] = []
    for eid in ids:
        sub_questions = prepare_sub_questions(
            decompose_idx.get(eid) or [],
            decompose_mode=args.decompose_mode,
        )
        if not sub_questions:
            continue
        examples.append(
            init_example_state(eid, records[eid], sub_questions, ranker_methods, oracle_ks)
        )

    def _make_accum(cls):
        return {"all": cls(), **{K: cls() for K in K_BUCKETS}}

    accum = {m: _make_accum(MetricAccum) for m in ranker_methods}
    chain_accum = {m: _make_accum(ChainAccum) for m in ranker_methods}
    chain_accum_k_match = {m: _make_accum(ChainAccum) for m in ranker_methods}
    case_recall_accum = {m: _make_accum(CaseRecallAccum) for m in ranker_methods}
    accum_hop = {m: defaultdict(MetricAccum) for m in ranker_methods}
    sel_accum = {m: _make_accum(SelectionAccum) for m in ranker_methods}
    sel_chain_accum = {m: _make_accum(SelectionChainAccum) for m in ranker_methods}
    sel_chain_accum_k_match = {m: _make_accum(SelectionChainAccum) for m in ranker_methods}
    sel_accum_hop = {m: defaultdict(SelectionAccum) for m in ranker_methods}
    ans_accum = {m: _make_accum(AnswerAccum) for m in ranker_methods}
    gate_accum = {m: GateAccum() for m in ranker_methods if m in GATED}
    oracle_accum = {m: {k: MetricAccum() for k in oracle_ks} for m in ranker_methods}
    oracle_chain = {m: {k: ChainAccum() for k in oracle_ks} for m in ranker_methods}

    retrieve_k = max(args.topk, args.expand_topk, 3)
    max_hops = max((ex.K for ex in examples), default=0)
    hidden_cache: dict[tuple[int, str], np.ndarray] = {}
    run_t0 = time.perf_counter()

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    for hop_j in range(1, max_hops + 1):
        hop_t0 = time.perf_counter()
        print(f"\n=== Wavefront hop {hop_j}/{max_hops} ===", flush=True)
        contexts: list[HopContext] = []

        retrieval_tasks: list[tuple[ExampleState, str, str]] = []
        for ex in examples:
            if hop_j > ex.K:
                continue
            raw_sq = ex.sub_questions[hop_j - 1]
            for method in ranker_methods:
                retrieval_tasks.append((ex, method, raw_sq))

        retrieve_iter = retrieval_tasks
        if retrieval_tasks and tqdm is not None:
            retrieve_iter = tqdm(
                retrieval_tasks,
                desc=f"hop{hop_j} retrieve",
                unit="ctx",
                file=sys.stderr,
                dynamic_ncols=True,
            )
        print(f"[hop {hop_j}] retrieval: {len(retrieval_tasks)} contexts", flush=True)
        for ex, method, raw_sq in retrieve_iter:
            ms = ex.methods[method]
            expanded_q = expand_hop_template(raw_sq, ms.prior)
            prefix_before = build_trace_prefix(ex.q_main, ms.hop_steps)
            retriever = retriever_for_method(method)
            try:
                candidates_all = retrieve_candidates(
                    expanded_q,
                    ex.paragraphs,
                    retrieve_k,
                    retriever=retriever,
                    cos_model=args.cos_model,
                    colbert_model=args.colbert_model,
                )
            except Exception as exc:
                print(f"  [warn] retrieval failed id={ex.eid} method={method}: {exc}", file=sys.stderr)
                candidates_all = []
            if not candidates_all:
                continue
            contexts.append(
                HopContext(
                    ex=ex,
                    method=method,
                    hop_j=hop_j,
                    raw_sq=raw_sq,
                    expanded_q=expanded_q,
                    prefix_before=prefix_before,
                    candidates_all=candidates_all,
                )
            )
        print(f"[hop {hop_j}] retrieved: {len(contexts)}/{len(retrieval_tasks)} contexts", flush=True)

        gate_req_id = 0
        gate_reqs: list[GateRequest] = []
        req_kind: dict[int, str] = {}
        req_to_ctx: dict[int, HopContext] = {}
        for ctx in contexts:
            if ctx.method in ("baseline", "colbert"):
                ctx.top3 = ctx.candidates_all[:3]
            elif ctx.method == "lr_rerank":
                gate_reqs.append(GateRequest(gate_req_id, ctx, ctx.candidates_all[: args.topk]))
                req_kind[gate_req_id] = "lr"
                req_to_ctx[gate_req_id] = ctx
                gate_req_id += 1
            else:
                gate_reqs.append(GateRequest(gate_req_id, ctx, ctx.candidates_all[:3]))
                req_kind[gate_req_id] = "gated_initial"
                req_to_ctx[gate_req_id] = ctx
                gate_req_id += 1

        if gate_reqs:
            print(f"[hop {hop_j}] gate scoring: {len(gate_reqs)} requests", flush=True)
            scored = score_gate_requests(
                gate_reqs,
                artifacts=artifacts,
                gate_artifact_mode=args.gate_artifact_mode,
                model=gate_model,
                tokenizer=gate_tokenizer,
                layer=args.layer,
                hidden_batch_size=args.hidden_batch_size,
                hidden_cache=hidden_cache,
                desc=f"hop{hop_j} gate",
            )
            expand_reqs: list[GateRequest] = []
            for req in gate_reqs:
                ctx = req.ctx
                sc = scored[req.req_id]
                if req_kind[req.req_id] == "lr":
                    ctx.top3 = rerank_by_final_score(sc, args.lambda_lr)[:3]
                    continue
                ab_top3 = [ab for _, _, ab in sc]
                art = get_gate_artifact(artifacts, args.gate_artifact_mode, ctx.ex.K, ctx.hop_j - 1) or {}
                threshold = art.get("threshold", args.abnormal_threshold)
                if ctx.method == "gated_rule_a":
                    ctx.gate_fired = bool(ab_top3 and ab_top3[0] > threshold)
                else:
                    ctx.gate_fired = sum(ab > threshold for ab in ab_top3) >= 2
                ctx.top3_initial_ab_scores = [round(ab, 4) for ab in ab_top3]
                ctx.ex.methods[ctx.method].ex_gate_fired.append(ctx.gate_fired)
                gate_accum[ctx.method].update_hop(ctx.gate_fired)
                if ctx.gate_fired:
                    expand_reqs.append(GateRequest(gate_req_id, ctx, ctx.candidates_all[: args.expand_topk]))
                    req_to_ctx[gate_req_id] = ctx
                    gate_req_id += 1
                else:
                    ctx.top3 = [(p, s) for p, s, _ in sc]

            if expand_reqs:
                print(f"[hop {hop_j}] gate expand scoring: {len(expand_reqs)} requests", flush=True)
                expanded_scored = score_gate_requests(
                    expand_reqs,
                    artifacts=artifacts,
                    gate_artifact_mode=args.gate_artifact_mode,
                    model=gate_model,
                    tokenizer=gate_tokenizer,
                    layer=args.layer,
                    hidden_batch_size=args.hidden_batch_size,
                    hidden_cache=hidden_cache,
                    desc=f"hop{hop_j} gate-expand",
                )
                for req in expand_reqs:
                    req.ctx.top3 = rerank_by_final_score(expanded_scored[req.req_id], args.lambda_lr)[:3]

        # Retrieval and oracle metrics after top3 is fixed.
        ready_contexts = [ctx for ctx in contexts if ctx.top3]
        for ctx in ready_contexts:
            ex = ctx.ex
            hop_row = ex.hop_results[ctx.hop_j - 1]
            gold_pi = hop_row.get("gold_para_idx")
            has_gold_hop = bool(hop_row.get("has_gold_hop"))
            if has_gold_hop and gold_pi is not None:
                ctx.gold_rank = find_gold_rank(ctx.top3 or [], int(gold_pi))
                accum[ctx.method]["all"].update(ctx.gold_rank)
                if ex.K in accum[ctx.method]:
                    accum[ctx.method][ex.K].update(ctx.gold_rank)
                accum_hop[ctx.method][(ex.K, ctx.hop_j - 1)].update(ctx.gold_rank)
                if "oracle" in args.methods:
                    for ok in oracle_ks:
                        pool = ctx.candidates_all[:ok]
                        in_pool = find_gold_rank(pool, int(gold_pi)) is not None
                        oracle_accum[ctx.method][ok].update(1 if in_pool else None)
                        ex.methods[ctx.method].ex_oracle_pool[ok].append(in_pool)
            else:
                ctx.gold_rank = None
            ex.methods[ctx.method].ex_ranks.append(ctx.gold_rank)

        # Batched vLLM passage selection (with prior-hop reasoning context).
        selection_prompts = [
            prompt_select_passage_with_context(
                ctx.ex.q_main,
                ctx.ex.methods[ctx.method].hop_steps,
                ctx.ex.methods[ctx.method].prior,
                ctx.expanded_q,
                [p for p, _ in (ctx.top3 or [])],
            )
            for ctx in ready_contexts
        ]
        selection_raw = generator.generate_batch(
            selection_prompts,
            args.max_new_tokens_select,
            desc=f"select-hop{hop_j}",
        )
        answer_prompts: list[str] = []
        answer_contexts: list[tuple[HopContext, int, dict[str, Any], bool | None]] = []
        for ctx, raw in zip(ready_contexts, selection_raw):
            top3_paras = [p for p, _ in (ctx.top3 or [])]
            choice = parse_json_choice(raw)
            if choice is None:
                choice = 0
            choice = max(0, min(len(top3_paras) - 1, choice))
            chosen_para = top3_paras[choice]
            hop_row = ctx.ex.hop_results[ctx.hop_j - 1]
            gold_pi = hop_row.get("gold_para_idx")
            if hop_row.get("has_gold_hop") and gold_pi is not None:
                sel_correct = int(chosen_para.get("idx", -1)) == int(gold_pi)
                sel_accum[ctx.method]["all"].update(sel_correct)
                if ctx.ex.K in sel_accum[ctx.method]:
                    sel_accum[ctx.method][ctx.ex.K].update(sel_correct)
                sel_accum_hop[ctx.method][(ctx.ex.K, ctx.hop_j - 1)].update(sel_correct)
            else:
                sel_correct = None
            ctx.ex.methods[ctx.method].ex_sel_correct.append(sel_correct)
            answer_contexts.append((ctx, choice, chosen_para, sel_correct))
            ms = ctx.ex.methods[ctx.method]
            answer_prompts.append(
                prompt_short_answer_with_context(
                    ctx.ex.q_main,
                    ms.hop_steps,
                    ms.prior,
                    ctx.expanded_q,
                    chosen_para,
                )
            )

        answer_raw = generator.generate_batch(
            answer_prompts,
            args.max_new_tokens_answer,
            desc=f"answer-hop{hop_j}",
        )
        for (ctx, choice, chosen_para, sel_correct), raw in zip(answer_contexts, answer_raw):
            sub_ans = normalize_short_answer(raw)
            ms = ctx.ex.methods[ctx.method]
            ms.prior.append(sub_ans)
            ev_text = (chosen_para.get("paragraph_text") or "").strip()
            ms.hop_steps.append((ctx.expanded_q, ev_text))
            method_hop_row: dict[str, Any] = {
                "expanded_subq": ctx.expanded_q,
                "top3_para_ids": [int(p.get("idx", -1)) for p, _ in (ctx.top3 or [])],
                "top3_scores": [round(s, 4) for _, s in (ctx.top3 or [])],
                "gold_rank": ctx.gold_rank,
                "gold_in_top3": ctx.gold_rank is not None and ctx.gold_rank <= 3,
                "selection": {
                    "mode": "select",
                    "choice": choice,
                    "chosen_para_id": int(chosen_para.get("idx", -1)),
                    "correct": sel_correct,
                },
                "sub_answer": sub_ans,
            }
            if ctx.method in GATED:
                method_hop_row["gate_fired"] = ctx.gate_fired
                if ctx.top3_initial_ab_scores is not None:
                    method_hop_row["top3_initial_ab_scores"] = ctx.top3_initial_ab_scores
            ctx.ex.hop_results[ctx.hop_j - 1][ctx.method] = method_hop_row

        hop_elapsed = time.perf_counter() - hop_t0
        print(
            f"[hop {hop_j}] done: {len(ready_contexts)} contexts, "
            f"hop elapsed {hop_elapsed / 60:.1f} min",
            flush=True,
        )

    if args.final_reader:
        print("\n=== Final reader ===", flush=True)
        final_t0 = time.perf_counter()
        final_tasks: list[tuple[ExampleState, str]] = []
        for ex in examples:
            for method in ranker_methods:
                ms = ex.methods[method]
                if ms.hop_steps:
                    final_tasks.append((ex, method))
        final_prompts = [
            build_final_reader_cot_prompt(
                ex.q_main,
                ex.methods[method].hop_steps,
                ex.methods[method].prior,
            )
            for ex, method in final_tasks
        ]
        final_raw = generator.generate_chat_batch(
            FINAL_READER_SYSTEM_PROMPT,
            final_prompts,
            args.max_new_tokens_final,
            desc="final-reader",
        )
        for (ex, method), raw in zip(final_tasks, final_raw):
            ms = ex.methods[method]
            ms.final_cot = raw
            hop_fallback = ms.prior[-1] if ms.prior else ""
            ms.final_answer = parse_final_answer_from_cot(raw, fallback=hop_fallback)
        final_elapsed = time.perf_counter() - final_t0
        print(
            f"[final reader] done: {len(final_tasks)} prompts, "
            f"elapsed {final_elapsed / 60:.1f} min",
            flush=True,
        )

    # Final per-example metrics and case writing.
    args.out_cases.parent.mkdir(parents=True, exist_ok=True)
    cases_fp = args.out_cases.open("w", encoding="utf-8")
    n_examples = 0
    n_k_match = 0
    n_skip = 0
    for ex in examples:
        n_examples += 1
        if ex.k_match:
            n_k_match += 1

        per_method_gold_ranks: dict[str, list[int | None]] = {}
        per_method_gold_sels: dict[str, list[bool | None]] = {}
        per_method_case_r1: dict[str, bool] = {}
        per_method_case_r3: dict[str, bool] = {}
        per_method_em: dict[str, int | None] = {}
        per_method_f1: dict[str, float | None] = {}

        for method in ranker_methods:
            ms = ex.methods[method]
            gold_ranks = gold_hop_ranks(ex.hop_results, method, ex.K_gold)
            gold_sels = gold_hop_selections(ex.hop_results, method, ex.K_gold)
            per_method_gold_ranks[method] = gold_ranks
            per_method_gold_sels[method] = gold_sels
            case_recall_accum[method]["all"].update(gold_ranks)
            if ex.K_gold in case_recall_accum[method]:
                case_recall_accum[method][ex.K_gold].update(gold_ranks)
            cr1, cr3 = case_recall_flags(gold_ranks)
            per_method_case_r1[method] = cr1
            per_method_case_r3[method] = cr3

            if ex.k_match:
                chain_accum_k_match[method]["all"].update(gold_ranks)
                if ex.K in chain_accum_k_match[method]:
                    chain_accum_k_match[method][ex.K].update(gold_ranks)
                sel_chain_accum_k_match[method]["all"].update(gold_sels)
                if ex.K in sel_chain_accum_k_match[method]:
                    sel_chain_accum_k_match[method][ex.K].update(gold_sels)

            if ms.ex_ranks:
                chain_accum[method]["all"].update(ms.ex_ranks)
                if ex.K in chain_accum[method]:
                    chain_accum[method][ex.K].update(ms.ex_ranks)
            if ms.ex_sel_correct:
                sel_chain_accum[method]["all"].update(ms.ex_sel_correct)
                if ex.K in sel_chain_accum[method]:
                    sel_chain_accum[method][ex.K].update(ms.ex_sel_correct)
            if method in GATED and ms.ex_gate_fired:
                gate_accum[method].update_example(any(ms.ex_gate_fired))
            if "oracle" in args.methods:
                for ok in oracle_ks:
                    if ms.ex_oracle_pool[ok]:
                        all_in = all(ms.ex_oracle_pool[ok])
                        oracle_chain[method][ok].update([1 if all_in else None])

            predicted = (
                ms.final_answer
                if args.final_reader
                else (ms.prior[-1] if ms.prior else "")
            )
            if predicted:
                try:
                    em, f1 = judge_answer_official(predicted, ex.row)
                except Exception:
                    em, f1 = None, None
            else:
                em, f1 = None, None
            per_method_em[method] = em
            per_method_f1[method] = f1
            ans_accum[method]["all"].update(em, f1)
            if ex.K in ans_accum[method]:
                ans_accum[method][ex.K].update(em, f1)

        case_row = {
            "id": ex.eid,
            "question": ex.q_main,
            "K_pred": ex.K,
            "K_gold": ex.K_gold,
            "K_match": ex.k_match,
            "chain_recall1": {
                m: all(r == 1 for r in ex.methods[m].ex_ranks) if ex.methods[m].ex_ranks else None
                for m in ranker_methods
            },
            "chain_recall3": {
                m: all(r is not None and r <= 3 for r in ex.methods[m].ex_ranks)
                if ex.methods[m].ex_ranks
                else None
                for m in ranker_methods
            },
            "chain_recall1_k_match": {
                m: per_method_case_r1[m] if ex.k_match else None for m in ranker_methods
            },
            "chain_recall3_k_match": {
                m: per_method_case_r3[m] if ex.k_match else None for m in ranker_methods
            },
            "case_recall1": {m: per_method_case_r1[m] for m in ranker_methods},
            "case_recall3": {m: per_method_case_r3[m] for m in ranker_methods},
            "gold_hop_ranks": per_method_gold_ranks,
            "chain_selection": {
                m: all(c is True for c in ex.methods[m].ex_sel_correct)
                if ex.methods[m].ex_sel_correct
                else None
                for m in ranker_methods
            },
            "chain_selection_k_match": {
                m: all(c is True for c in per_method_gold_sels[m])
                if ex.k_match and per_method_gold_sels[m]
                else None
                for m in ranker_methods
            },
            "predicted_answers": {
                m: (
                    ex.methods[m].final_answer
                    if args.final_reader
                    else (ex.methods[m].prior[-1] if ex.methods[m].prior else "")
                )
                for m in ranker_methods
            },
            "answer_em": per_method_em,
            "answer_f1": per_method_f1,
            "hop_results": ex.hop_results,
        }
        if args.final_reader:
            case_row["final_reader_cot"] = {
                m: ex.methods[m].final_cot for m in ranker_methods
            }
        if any(m in GATED for m in ranker_methods):
            case_row["gate_fired_any"] = {
                m: any(ex.methods[m].ex_gate_fired) if ex.methods[m].ex_gate_fired else None
                for m in ranker_methods if m in GATED
            }
        cases_fp.write(json.dumps(case_row, ensure_ascii=False) + "\n")
    cases_fp.close()

    run_wall_sec = time.perf_counter() - run_t0
    results: dict[str, Any] = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "output_dir": str(args.out_dir),
        "n_skip": n_skip,
        "n_examples": n_examples,
        "n_k_match": n_k_match,
        "k_match_rate": round(n_k_match / n_examples, 4) if n_examples else 0.0,
        "timing": {
            "wall_clock_sec": round(run_wall_sec, 2),
            "wall_clock_hr": round(run_wall_sec / 3600, 3),
            "note": "Wavefront batched vLLM generation; per-method timing not collected.",
        },
    }

    def transition_label(j: int, K: int) -> str:
        if j == 0:
            return "Q->E1"
        if j == K:
            return f"E{K}->Final"
        return f"E{j}->E{j+1}"

    for method in ranker_methods:
        mres: dict[str, Any] = {
            "overall": accum[method]["all"].result(),
            "by_K": {K: accum[method][K].result() for K in K_BUCKETS},
            "by_hop": {
                f"K{K}_j{j}_{transition_label(j, K)}": accum_hop[method][(K, j)].result()
                for (K, j) in sorted(accum_hop[method].keys())
            },
            "chain_overall": chain_accum[method]["all"].result(),
            "chain_by_K": {K: chain_accum[method][K].result() for K in K_BUCKETS},
            "chain_overall_k_match": chain_accum_k_match[method]["all"].result(),
            "chain_by_K_k_match": {
                K: chain_accum_k_match[method][K].result() for K in K_BUCKETS
            },
            "case_recall_overall": case_recall_accum[method]["all"].result(),
            "case_recall_by_K_gold": {
                K: case_recall_accum[method][K].result() for K in K_BUCKETS
            },
            "selection_overall": sel_accum[method]["all"].result(),
            "selection_by_K": {K: sel_accum[method][K].result() for K in K_BUCKETS},
            "selection_by_hop": {
                f"K{K}_j{j}_{transition_label(j, K)}": sel_accum_hop[method][(K, j)].result()
                for (K, j) in sorted(sel_accum_hop[method].keys())
            },
            "chain_selection_overall": sel_chain_accum[method]["all"].result(),
            "chain_selection_by_K": {
                K: sel_chain_accum[method][K].result() for K in K_BUCKETS
            },
            "chain_selection_overall_k_match": sel_chain_accum_k_match[method]["all"].result(),
            "chain_selection_by_K_k_match": {
                K: sel_chain_accum_k_match[method][K].result() for K in K_BUCKETS
            },
            "answer_overall": ans_accum[method]["all"].result(),
            "answer_by_K": {K: ans_accum[method][K].result() for K in K_BUCKETS},
        }
        if method in GATED:
            mres["gate_stats"] = gate_accum[method].result()
        if "oracle" in args.methods:
            mres["oracle_coverage"] = {
                f"top{ok}": {
                    "hop_level": oracle_accum[method][ok].result(),
                    "chain_level": oracle_chain[method][ok].result(),
                }
                for ok in oracle_ks
            }
        results[method] = mres

    out_path = args.out_dir / f"retrieval_exp_{args.split}{tag_suffix}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMetrics saved to  {out_path}")
    print(f"Cases saved to    {args.out_cases}")
    print(f"Wall clock hr     {results['timing']['wall_clock_hr']:.3f}")
    for method in ranker_methods:
        ov = results[method]["overall"]
        ans = results[method]["answer_overall"]
        print(
            f"{method:<14} R@1={ov['recall@1']:.4f} R@3={ov['recall@3']:.4f} "
            f"EM={ans['answer_em']:.4f} F1={ans['answer_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
