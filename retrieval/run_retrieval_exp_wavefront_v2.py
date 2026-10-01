#!/usr/bin/env python3
"""
Extensible successor to run_retrieval_exp_wavefront.py: same wavefront
(hop-by-hop, batched vLLM + batched gate scoring) execution model and the
same output format, but "which candidates / how scored / how reranked" is
composed from pluggable classes (candidate_pool.py, retrievers.py,
rerankers.py, method_registry.py) instead of hardcoded if/elif branches on a
method-name string. Adds two new methods — baseline_global / colbert_global —
that search a precomputed, whole-dataset embedding cache (see
embedding_cache.py / build_embedding_cache.py) instead of each example's own
local paragraph pool.

run_retrieval_exp.py and run_retrieval_exp_wavefront.py are NOT modified —
this script only imports from them. The existing 5 methods (baseline,
lr_rerank, gated_rule_a, gated_rule_b, colbert) must produce byte-identical
retrieval_exp_*.json / retrieval_cases_*.jsonl output to the original script
given the same inputs — that's the whole point of the refactor, see
0705update.md-style verification notes in the design plan
(.claude/plans/adaptive-imagining-blum.md).

Usage:
  python run_retrieval_exp_wavefront_v2.py --methods baseline lr_rerank gated_rule_a --limit 20
  python run_retrieval_exp_wavefront_v2.py --methods baseline_global --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

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
    gold_hop_ranks,
    gold_hop_selections,
    load_artifacts,
    load_dataset_records,
    load_decompose_index,
    normalize_short_answer,
    prepare_sub_questions,
    resolve_decompose_file,
    resolve_output_dir,
    sanitize_run_tag,
    select_example_ids,
)
from run_retrieval_exp_wavefront import (
    BatchVllmGenerator,
    ExampleState,
    HopContext,
    build_final_reader_cot_prompt,
    FINAL_READER_SYSTEM_PROMPT,
    _get_pipeline_helpers,
    init_example_state,
    load_gate_model,
    parse_final_answer_from_cot,
    prompt_select_passage_with_context,
    prompt_short_answer_with_context,
)
from candidate_pool import find_gold_rank_scoped
from embedding_cache import DEFAULT_CACHE_ROOT, GlobalEmbeddingCache
from method_registry import COLBERT_GLOBAL_METHODS, COSINE_GLOBAL_METHODS, build_method_registry
from rerank_driver import run_rerank_batch
from rerankers import GatedRuleRerank


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
    ap.add_argument(
        "--methods", nargs="+",
        choices=["baseline", "colbert", "oracle", "lr_rerank", "gated_rule_a", "gated_rule_b",
                 "baseline_global", "colbert_global",
                 "baseline_global_lr_rerank", "colbert_global_lr_rerank"],
        default=["baseline", "oracle", "lr_rerank", "gated_rule_a", "gated_rule_b"],
    )
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--expand-topk", type=int, default=10)
    ap.add_argument("--lambda-lr", type=float, default=0.25)
    ap.add_argument("--abnormal-threshold", type=float, default=0.5)
    ap.add_argument("--cos-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--colbert-model", default="colbert-ir/colbertv2.0")
    ap.add_argument("--global-cache-root", type=Path, default=DEFAULT_CACHE_ROOT,
                    help="Where build_embedding_cache.py wrote its cache(s). Required when "
                         "--methods includes baseline_global / colbert_global.")
    ap.add_argument("--skip-global-cache-freshness-check", action="store_true",
                    help="Skip verifying the global cache was built from the current raw data "
                         "file (size+mtime match) — use only if you know the cache is fine and "
                         "want to avoid the stat() call.")
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
    ap.add_argument("--layer", type=int, default=31,
                    help="Fallback layer for gate artifacts saved before the per-artifact 'layer' "
                         "field existed (see fit_lr_gate_pooled.py --per-j-layers); the layer "
                         "actually used per request comes from that request's own gate artifact.")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--vllm-batch-size", type=int, default=128)
    ap.add_argument("--hidden-batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens-select", type=int, default=64)
    ap.add_argument("--max-new-tokens-answer", type=int, default=64)
    ap.add_argument(
        "--final-reader", action=argparse.BooleanOptionalAction, default=True,
        help="After all hops, run a batched final reader (re-read question + direct answer).",
    )
    ap.add_argument("--max-new-tokens-final", type=int, default=128)
    ap.add_argument("--selection-mode", choices=["select"], default="select")
    return ap.parse_args()


def _load_global_caches(args: argparse.Namespace) -> tuple[GlobalEmbeddingCache | None, GlobalEmbeddingCache | None]:
    """Load whichever global caches this run's --methods actually need. Fails
    loudly (not a lazy build) if a needed cache is missing — building one can
    take a long time and shouldn't be a surprise buried inside an experiment
    run; see build_embedding_cache.py."""
    cosine_cache = colbert_cache = None
    needs_cosine_global = any(m in COSINE_GLOBAL_METHODS for m in args.methods)
    needs_colbert_global = any(m in COLBERT_GLOBAL_METHODS for m in args.methods)
    if not (needs_cosine_global or needs_colbert_global):
        return None, None

    common = dict(
        dataset=args.dataset, split=args.split, cache_root=args.global_cache_root,
        musique_dir=args.musique_dir, twowiki_file=args.twowiki_file, hotpot_file=args.hotpot_file,
        check_freshness=not args.skip_global_cache_freshness_check,
    )
    if needs_cosine_global:
        cosine_cache = GlobalEmbeddingCache.load(
            retriever_kind="cosine", model_name=args.cos_model, **common,
        )
        print(f"[global pool] loaded cosine cache: {len(cosine_cache.paragraphs)} passages", flush=True)
    if needs_colbert_global:
        colbert_cache = GlobalEmbeddingCache.load(
            retriever_kind="colbert", model_name=args.colbert_model, **common,
        )
        print(f"[global pool] loaded colbert cache: {len(colbert_cache.paragraphs)} passages", flush=True)
    return cosine_cache, colbert_cache


def main() -> None:
    args = parse_args()
    if args.artifacts_dir is None:
        args.artifacts_dir = (
            POOLED_ARTIFACTS_DIR if args.gate_artifact_mode == "pooled" else ARTIFACTS_DIR
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
        ids, decompose_idx, limit=args.limit, limit_per_k=args.limit_per_k, seed=args.sample_seed,
    )
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.num_shards > 1:
        ids = ids[args.shard_index :: args.num_shards]

    tag = args.run_tag or f"wavefrontv2_{args.dataset}_{sanitize_run_tag('_'.join(args.methods))}"
    args.out_dir = resolve_output_dir(
        out_dir=args.out_dir, results_root=args.results_root, run_tag=tag,
        decompose_mode=args.decompose_mode, methods=args.methods,
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

    global_cache_cosine, global_cache_colbert = _load_global_caches(args)

    method_specs = build_method_registry(
        methods=ranker_methods,
        artifacts=artifacts,
        gate_artifact_mode=args.gate_artifact_mode,
        topk=args.topk,
        expand_topk=args.expand_topk,
        lambda_lr=args.lambda_lr,
        abnormal_threshold=args.abnormal_threshold,
        cos_model=args.cos_model,
        colbert_model=args.colbert_model,
        global_cache_cosine=global_cache_cosine,
        global_cache_colbert=global_cache_colbert,
    )
    gated_methods = {m for m in ranker_methods if isinstance(method_specs[m].reranker, GatedRuleRerank)}

    print("Loading vLLM generator ...", flush=True)
    generator = BatchVllmGenerator(
        args.model, dtype=args.dtype, tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=vllm_max_len,
    )

    (
        expand_hop_template, _prompt_select_passage, parse_json_choice,
        _prompt_short_answer, _prompt_concat_answer, _parse_json_answer_choice,
        judge_answer_official,
    ) = _get_pipeline_helpers()

    oracle_ks = [3, 5, 10, 20]
    examples: list[ExampleState] = []
    for eid in ids:
        sub_questions = prepare_sub_questions(
            decompose_idx.get(eid) or [], decompose_mode=args.decompose_mode,
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
    gate_accum = {m: GateAccum() for m in gated_methods}
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
                retrieval_tasks, desc=f"hop{hop_j} retrieve", unit="ctx",
                file=sys.stderr, dynamic_ncols=True,
            )
        print(f"[hop {hop_j}] retrieval: {len(retrieval_tasks)} contexts", flush=True)
        for ex, method, raw_sq in retrieve_iter:
            ms = ex.methods[method]
            expanded_q = expand_hop_template(raw_sq, ms.prior)
            prefix_before = build_trace_prefix(ex.q_main, ms.hop_steps)
            spec = method_specs[method]
            try:
                pool = spec.pool_source.get_pool(ex)
                candidates_all = spec.retriever.score(expanded_q, pool, retrieve_k)
            except Exception as exc:
                print(f"  [warn] retrieval failed id={ex.eid} method={method}: {exc}", file=sys.stderr)
                candidates_all = []
            if not candidates_all:
                continue
            contexts.append(
                HopContext(
                    ex=ex, method=method, hop_j=hop_j, raw_sq=raw_sq, expanded_q=expanded_q,
                    prefix_before=prefix_before, candidates_all=candidates_all,
                )
            )
        print(f"[hop {hop_j}] retrieved: {len(contexts)}/{len(retrieval_tasks)} contexts", flush=True)

        print(f"[hop {hop_j}] rerank/gate scoring for {len(contexts)} contexts", flush=True)
        run_rerank_batch(
            contexts, method_specs,
            artifacts=artifacts, gate_artifact_mode=args.gate_artifact_mode,
            model=gate_model, tokenizer=gate_tokenizer, default_layer=args.layer,
            hidden_batch_size=args.hidden_batch_size, hidden_cache=hidden_cache,
            desc=f"hop{hop_j} gate",
        )
        # Gated-method bookkeeping that the original hop loop did inline —
        # GatedRuleRerank.apply_initial() already set ctx.gate_fired /
        # ctx.top3_initial_ab_scores on each ctx; the reranker itself doesn't
        # know about these run-level accumulators, so it's collected here.
        for ctx in contexts:
            if ctx.method in gated_methods:
                ctx.ex.methods[ctx.method].ex_gate_fired.append(ctx.gate_fired)
                gate_accum[ctx.method].update_hop(ctx.gate_fired)

        # Retrieval and oracle metrics after top3 is fixed.
        ready_contexts = [ctx for ctx in contexts if ctx.top3]
        for ctx in ready_contexts:
            ex = ctx.ex
            hop_row = ex.hop_results[ctx.hop_j - 1]
            gold_pi = hop_row.get("gold_para_idx")
            has_gold_hop = bool(hop_row.get("has_gold_hop"))
            if has_gold_hop and gold_pi is not None:
                ctx.gold_rank = find_gold_rank_scoped(ctx.top3 or [], int(gold_pi), example_id=ex.eid)
                accum[ctx.method]["all"].update(ctx.gold_rank)
                if ex.K in accum[ctx.method]:
                    accum[ctx.method][ex.K].update(ctx.gold_rank)
                accum_hop[ctx.method][(ex.K, ctx.hop_j - 1)].update(ctx.gold_rank)
                if "oracle" in args.methods:
                    for ok in oracle_ks:
                        pool = ctx.candidates_all[:ok]
                        in_pool = find_gold_rank_scoped(pool, int(gold_pi), example_id=ex.eid) is not None
                        oracle_accum[ctx.method][ok].update(1 if in_pool else None)
                        ex.methods[ctx.method].ex_oracle_pool[ok].append(in_pool)
            else:
                ctx.gold_rank = None
            ex.methods[ctx.method].ex_ranks.append(ctx.gold_rank)

        # Batched vLLM passage selection (with prior-hop reasoning context).
        selection_prompts = [
            prompt_select_passage_with_context(
                ctx.ex.q_main, ctx.ex.methods[ctx.method].hop_steps, ctx.ex.methods[ctx.method].prior,
                ctx.expanded_q, [p for p, _ in (ctx.top3 or [])],
            )
            for ctx in ready_contexts
        ]
        selection_raw = generator.generate_batch(
            selection_prompts, args.max_new_tokens_select, desc=f"select-hop{hop_j}",
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
                sel_correct = (
                    int(chosen_para.get("idx", -1)) == int(gold_pi)
                    and chosen_para.get("_source_id", ctx.ex.eid) == ctx.ex.eid
                )
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
                    ctx.ex.q_main, ms.hop_steps, ms.prior, ctx.expanded_q, chosen_para,
                )
            )

        answer_raw = generator.generate_batch(
            answer_prompts, args.max_new_tokens_answer, desc=f"answer-hop{hop_j}",
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
            if ctx.method in gated_methods:
                method_hop_row["gate_fired"] = ctx.gate_fired
                if ctx.top3_initial_ab_scores is not None:
                    method_hop_row["top3_initial_ab_scores"] = ctx.top3_initial_ab_scores
            ctx.ex.hop_results[ctx.hop_j - 1][ctx.method] = method_hop_row

        hop_elapsed = time.perf_counter() - hop_t0
        print(
            f"[hop {hop_j}] done: {len(ready_contexts)} contexts, hop elapsed {hop_elapsed / 60:.1f} min",
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
            build_final_reader_cot_prompt(ex.q_main, ex.methods[method].hop_steps, ex.methods[method].prior)
            for ex, method in final_tasks
        ]
        final_raw = generator.generate_chat_batch(
            FINAL_READER_SYSTEM_PROMPT, final_prompts, args.max_new_tokens_final, desc="final-reader",
        )
        for (ex, method), raw in zip(final_tasks, final_raw):
            ms = ex.methods[method]
            ms.final_cot = raw
            hop_fallback = ms.prior[-1] if ms.prior else ""
            ms.final_answer = parse_final_answer_from_cot(raw, fallback=hop_fallback)
        final_elapsed = time.perf_counter() - final_t0
        print(f"[final reader] done: {len(final_tasks)} prompts, elapsed {final_elapsed / 60:.1f} min", flush=True)

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
            if method in gated_methods and ms.ex_gate_fired:
                gate_accum[method].update_example(any(ms.ex_gate_fired))
            if "oracle" in args.methods:
                for ok in oracle_ks:
                    if ms.ex_oracle_pool[ok]:
                        all_in = all(ms.ex_oracle_pool[ok])
                        oracle_chain[method][ok].update([1 if all_in else None])

            predicted = ms.final_answer if args.final_reader else (ms.prior[-1] if ms.prior else "")
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
                if ex.methods[m].ex_ranks else None
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
                if ex.methods[m].ex_sel_correct else None
                for m in ranker_methods
            },
            "chain_selection_k_match": {
                m: all(c is True for c in per_method_gold_sels[m])
                if ex.k_match and per_method_gold_sels[m] else None
                for m in ranker_methods
            },
            "predicted_answers": {
                m: (
                    ex.methods[m].final_answer if args.final_reader
                    else (ex.methods[m].prior[-1] if ex.methods[m].prior else "")
                )
                for m in ranker_methods
            },
            "answer_em": per_method_em,
            "answer_f1": per_method_f1,
            "hop_results": ex.hop_results,
        }
        if args.final_reader:
            case_row["final_reader_cot"] = {m: ex.methods[m].final_cot for m in ranker_methods}
        if gated_methods:
            case_row["gate_fired_any"] = {
                m: any(ex.methods[m].ex_gate_fired) if ex.methods[m].ex_gate_fired else None
                for m in ranker_methods if m in gated_methods
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
            "note": "Wavefront v2 (extensible pool/retriever/reranker); per-method timing not collected.",
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
            "chain_by_K_k_match": {K: chain_accum_k_match[method][K].result() for K in K_BUCKETS},
            "case_recall_overall": case_recall_accum[method]["all"].result(),
            "case_recall_by_K_gold": {K: case_recall_accum[method][K].result() for K in K_BUCKETS},
            "selection_overall": sel_accum[method]["all"].result(),
            "selection_by_K": {K: sel_accum[method][K].result() for K in K_BUCKETS},
            "selection_by_hop": {
                f"K{K}_j{j}_{transition_label(j, K)}": sel_accum_hop[method][(K, j)].result()
                for (K, j) in sorted(sel_accum_hop[method].keys())
            },
            "chain_selection_overall": sel_chain_accum[method]["all"].result(),
            "chain_selection_by_K": {K: sel_chain_accum[method][K].result() for K in K_BUCKETS},
            "chain_selection_overall_k_match": sel_chain_accum_k_match[method]["all"].result(),
            "chain_selection_by_K_k_match": {
                K: sel_chain_accum_k_match[method][K].result() for K in K_BUCKETS
            },
            "answer_overall": ans_accum[method]["all"].result(),
            "answer_by_K": {K: ans_accum[method][K].result() for K in K_BUCKETS},
        }
        if method in gated_methods:
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
            f"{method:<16} R@1={ov['recall@1']:.4f} R@3={ov['recall@3']:.4f} "
            f"EM={ans['answer_em']:.4f} F1={ans['answer_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
