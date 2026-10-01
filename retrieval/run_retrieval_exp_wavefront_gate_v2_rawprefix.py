#!/usr/bin/env python3
"""
Gate v2 (Delta only) wavefront, with the SAME "rawprefix" fix already applied to gate v3
(see run_retrieval_exp_wavefront_gate_v3_rawprefix.py): a SEPARATE raw (unresolved)
accumulated prefix used only for gate scoring, so the text gate v2 sees at inference matches
its training distribution (hidden_states/pilot_multilayer, which never resolves a
sub-question's "[Answer N]" back-reference) -- not just at the current hop but across the
whole accumulated prefix.

This exists so the controlled 3-way comparison (Delta only / h only / h+Delta fused, see
gate_v3_main_method_report.md appendix B.2) isolates ONLY "which gate signal" as the
variable: without this, comparing gate v2 (scored via run_retrieval_exp_wavefront.py's
score_gate_requests, which uses the EXPANDED sub-question) against gate v3-rawprefix (scored
via the raw/unexpanded sub-question) would be confounded by a second, unrelated difference
(train/inference text-distribution alignment) on top of the actual feature difference.

Retrieval, the per-hop short-answer generation, and the final reader are IDENTICAL to
run_retrieval_exp_wavefront_gate_v3_rawprefix.py -- only which gate artifact is loaded and
how it's scored differs (Delta-only PCA+LR, reusing gate/lr_artifacts.py's
load_pooled_artifacts/score_lr_delta instead of gate v3's concat-feature scorer).

Needs (train first if missing): gate/artifacts_pooled_v2/j{0..3}.joblib

Reuses (imports, does not copy) retrieval/run_retrieval_exp.py,
retrieval/run_retrieval_exp_wavefront.py, and gate/lr_artifacts.py -- none of those files
are modified.

Usage:
  python run_retrieval_exp_wavefront_gate_v2_rawprefix.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
RETRIEVAL_DIR = HERE
PROJECT_ROOT = HERE.parent
GATE_DIR = PROJECT_ROOT / "gate"
GATE_V2_ARTIFACTS_DIR = GATE_DIR / "artifacts_pooled_v2"
TRACES_DIR = PROJECT_ROOT / "traces"
sys.path.insert(0, str(TRACES_DIR))
sys.path.insert(0, str(RETRIEVAL_DIR))
sys.path.insert(0, str(GATE_DIR))

from run_retrieval_exp import (  # noqa: E402
    HOTPOT_DEV_FILE,
    K_BUCKETS,
    MUSIQUE_DIR,
    TWOWIKI_DEV_FILE,
    AnswerAccum,
    MetricAccum,
    _get_pipeline_helpers,
    build_trace_prefix,
    embed_retrieval,
    find_gold_rank,
    load_dataset_records,
    load_decompose_index,
    normalize_short_answer,
    prepare_sub_questions,
    resolve_decompose_file,
    resolve_output_dir,
    sanitize_run_tag,
    select_example_ids,
)
from run_retrieval_exp_wavefront import (  # noqa: E402
    FINAL_READER_SYSTEM_PROMPT,
    BatchVllmGenerator,
    batch_last_hidden,
    build_final_reader_cot_prompt,
    load_gate_model,
    parse_final_answer_from_cot,
    prompt_short_answer_with_context,
    warmup_gate_model_memory,
)
from lr_artifacts import load_pooled_artifacts, score_lr_delta  # noqa: E402
from trace_evidence import gold_evidence_musique  # noqa: E402
from trace_format import escape_double_quotes  # noqa: E402


@dataclass
class BeamPath:
    hop_steps: list[tuple[str, str]] = field(default_factory=list)       # (expanded_q, evidence) -- reader-facing
    gate_hop_steps: list[tuple[str, str]] = field(default_factory=list)  # (raw_sub_q, evidence) -- gate-scoring only
    prior: list[str] = field(default_factory=list)
    para_ids: list[int] = field(default_factory=list)
    rank_key: float = 0.0
    gate_v2_score: float = 0.0


@dataclass
class ExampleState:
    eid: str
    row: dict[str, Any]
    q_main: str
    paragraphs: list[dict[str, Any]]
    sub_questions: list[str]
    gold_idxs: list[int]
    K: int
    beams: list[BeamPath]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["musique", "2wiki", "hotpot"], default="musique")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--musique-dir", type=Path, default=MUSIQUE_DIR)
    ap.add_argument("--twowiki-file", type=Path, default=TWOWIKI_DEV_FILE)
    ap.add_argument("--hotpot-file", type=Path, default=HOTPOT_DEV_FILE)
    ap.add_argument("--decompose-mode", choices=["gt", "bart_decompose"], default="bart_decompose")
    ap.add_argument("--decompose-file", type=Path, default=None)
    ap.add_argument("--artifacts-dir", type=Path, default=GATE_V2_ARTIFACTS_DIR)
    ap.add_argument("--lambda-gate", type=float, default=0.50,
                     help="Matches the unified lambda used across the controlled 3-way "
                          "comparison (Delta only / h only / h+Delta fused) -- not "
                          "separately swept for gate v2.")
    ap.add_argument("--beam-width", type=int, default=1)
    ap.add_argument("--retrieve-k", type=int, default=10)
    ap.add_argument("--cos-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument(
        "--query-instruction",
        default="Represent this sentence for searching relevant passages: ",
        help="BGE's own documented instruction prefix for the QUERY side of asymmetric "
             "retrieval -- see run_retrieval_exp_wavefront_gate_v3_rawprefix.py for the "
             "same flag. Pass '' to reproduce the old unprefixed behavior.",
    )
    ap.add_argument("--answerable-only", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--results-root", type=Path, default=HERE / "results")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--gate-device", default="cuda:0")
    ap.add_argument("--hidden-batch-size", type=int, default=8)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-new-tokens-answer", type=int, default=64)
    ap.add_argument("--max-new-tokens-final", type=int, default=128)
    return ap.parse_args()


def main() -> None:
    run_t0 = time.perf_counter()
    args = parse_args()
    args.decompose_file = resolve_decompose_file(
        args.decompose_mode, args.split, args.decompose_file, args.dataset
    )

    ds_args = argparse.Namespace(
        dataset=args.dataset, musique_dir=args.musique_dir, split=args.split,
        twowiki_file=args.twowiki_file, hotpot_file=args.hotpot_file,
    )
    records = load_dataset_records(ds_args)
    decompose_idx = load_decompose_index(args.decompose_file)
    ids = sorted(set(records) & set(decompose_idx))
    if args.answerable_only:
        ids = [eid for eid in ids if records[eid].get("answerable", True)]
    ids = select_example_ids(ids, decompose_idx, limit=args.limit, limit_per_k=0, seed=args.sample_seed)

    tag = args.run_tag or f"gate_v2_rawprefix_{args.dataset}_bw{args.beam_width}"
    args.out_dir = resolve_output_dir(
        out_dir=args.out_dir, results_root=args.results_root, run_tag=tag,
        decompose_mode=args.decompose_mode, methods=["gate_v2_rawprefix"],
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{sanitize_run_tag(args.run_tag)}" if args.run_tag else ""
    out_cases = args.out_dir / f"retrieval_cases_{args.split}{tag_suffix}.jsonl"
    out_metrics = args.out_dir / f"retrieval_exp_{args.split}{tag_suffix}.json"
    print(f"Output directory: {args.out_dir}")

    (expand_hop_template, *_rest, judge_answer_official) = _get_pipeline_helpers()

    gate_v2_artifacts, _meta = load_pooled_artifacts(args.artifacts_dir)
    print(f"Gate v2: {args.artifacts_dir} (j -> layer: "
          f"{ {j: art['layer'] for j, art in gate_v2_artifacts.items()} })")

    print(f"Examples: {len(ids)}  beam_width={args.beam_width}  decompose={args.decompose_file}")
    examples: list[ExampleState] = []
    for eid in ids:
        row = records[eid]
        sub_qs = prepare_sub_questions(decompose_idx.get(eid) or [], decompose_mode=args.decompose_mode)
        if not sub_qs:
            continue
        _gold_texts, gold_idxs = gold_evidence_musique(row)
        examples.append(ExampleState(
            eid=eid, row=row, q_main=(row.get("question") or "").strip(),
            paragraphs=row.get("paragraphs") or [], sub_questions=sub_qs,
            gold_idxs=gold_idxs, K=len(sub_qs), beams=[BeamPath()],
        ))

    print("Loading Llama model for hidden-state scoring ...", flush=True)
    gate_model, gate_tokenizer = load_gate_model(args)
    warmup_gate_model_memory(gate_model, gate_tokenizer, args.hidden_batch_size)
    print("Loading vLLM generator ...", flush=True)
    generator = BatchVllmGenerator(
        args.model, dtype=args.dtype, tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
    )

    hidden_cache: dict[tuple[int, str], Any] = {}
    max_hops = max((ex.K for ex in examples), default=0)
    oracle_hop_survive: dict[Any, MetricAccum] = {"all": MetricAccum(), **{K: MetricAccum() for K in K_BUCKETS}}
    # Gold's rank in the FULL reranked candidate pool (up to --retrieve-k), computed BEFORE
    # the beam_width truncation below -- real recall@1/@3, same fix as gate v3's rawprefix.
    full_pool_rank: dict[Any, MetricAccum] = {"all": MetricAccum(), **{K: MetricAccum() for K in K_BUCKETS}}

    for hop_j in range(1, max_hops + 1):
        t0 = time.perf_counter()
        print(f"\n=== hop {hop_j}/{max_hops} ===", flush=True)

        art = gate_v2_artifacts.get(hop_j - 1)
        layer = int(art["layer"]) if art is not None else 31

        # Retrieval: UNCHANGED -- expanded_q resolves "[Answer N]" via the per-hop generated
        # short answers so far, same as gate v3's rawprefix script.
        expand_retrieve: list[tuple[ExampleState, int, str, str, list[tuple[dict, float]]]] = []
        for ex in examples:
            if hop_j > ex.K:
                continue
            raw_sq = ex.sub_questions[hop_j - 1]
            for path_idx, path in enumerate(ex.beams):
                expanded_q = expand_hop_template(raw_sq, path.prior)
                query_text = f"{args.query_instruction}{expanded_q}"
                candidates_all = embed_retrieval(query_text, ex.paragraphs, args.retrieve_k, args.cos_model)
                if not candidates_all:
                    continue
                expand_retrieve.append((ex, path_idx, raw_sq, expanded_q, candidates_all))
        print(f"[hop {hop_j}] {len(expand_retrieve)} (example, path) contexts", flush=True)

        # Gate-scoring text: built from gate_hop_steps (raw sub-questions, all prior hops
        # included) + the CURRENT hop's raw_sq -- matches gate v2's training distribution
        # end to end, same rawprefix fix as gate v3.
        texts: list[str] = []
        owners: list[tuple[int, int]] = []  # (context index, slot); slot=-1 -> prefix_before
        for ctx_idx, (ex, path_idx, raw_sq, expanded_q, candidates_all) in enumerate(expand_retrieve):
            parent = ex.beams[path_idx]
            prefix_before = build_trace_prefix(ex.q_main, parent.gate_hop_steps)
            texts.append(prefix_before)
            owners.append((ctx_idx, -1))
            for slot, (para, _emb_score) in enumerate(candidates_all):
                ev_text = (para.get("paragraph_text") or "").strip()
                text = f'{prefix_before} Step {hop_j}: {raw_sq} Evidence: "{escape_double_quotes(ev_text)}"'
                texts.append(text)
                owners.append((ctx_idx, slot))

        hiddens_by_layer = batch_last_hidden(
            texts=texts, model=gate_model, tokenizer=gate_tokenizer, layers=[layer],
            batch_size=args.hidden_batch_size, cache=hidden_cache, desc=f"hop{hop_j} hidden",
        )
        hiddens = hiddens_by_layer[layer]

        prefix_hidden: dict[int, np.ndarray] = {}
        cand_hidden: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
        for (ctx_idx, slot), h in zip(owners, hiddens):
            if slot == -1:
                prefix_hidden[ctx_idx] = h
            else:
                cand_hidden[ctx_idx][slot] = h

        continuations_by_ex: dict[str, list[tuple[float, float, int, str, str, dict]]] = defaultdict(list)
        for ctx_idx, (ex, path_idx, raw_sq, expanded_q, candidates_all) in enumerate(expand_retrieve):
            h_prev = prefix_hidden[ctx_idx]
            for slot, (para, emb_score) in enumerate(candidates_all):
                if art is not None:
                    delta = cand_hidden[ctx_idx][slot] - h_prev
                    gate_score = score_lr_delta(delta, art)["score"]
                else:
                    gate_score = 0.0
                rank_key = emb_score - args.lambda_gate * gate_score
                continuations_by_ex[ex.eid].append((rank_key, gate_score, path_idx, raw_sq, expanded_q, para))

        kept_by_ex: dict[str, list[tuple[float, float, int, str, str, dict]]] = {}
        for ex in examples:
            conts = continuations_by_ex.get(ex.eid)
            if not conts:
                continue
            conts.sort(key=lambda t: t[0], reverse=True)
            kept_by_ex[ex.eid] = conts[: args.beam_width]

            gold_idx = ex.gold_idxs[hop_j - 1] if hop_j - 1 < len(ex.gold_idxs) else None
            if gold_idx is not None:
                full_ranked = [(para, rk) for rk, _g, _p, _rq, _eq, para in conts]
                full_rank = find_gold_rank(full_ranked, gold_idx)
                full_pool_rank["all"].update(full_rank)
                if ex.K in full_pool_rank:
                    full_pool_rank[ex.K].update(full_rank)

                survivor_ranked = [(para, rk) for rk, _g, _p, _rq, _eq, para in kept_by_ex[ex.eid]]
                rank = find_gold_rank(survivor_ranked, gold_idx)
                oracle_hop_survive["all"].update(rank)
                if ex.K in oracle_hop_survive:
                    oracle_hop_survive[ex.K].update(rank)

        # Per-hop short-answer generation: UNCHANGED -- still uses hop_steps (expanded) + prior.
        answer_prompts: list[str] = []
        answer_meta: list[tuple[ExampleState, float, float, int, str, str, dict]] = []
        for ex in examples:
            for rank_key, gate_score, path_idx, raw_sq, expanded_q, para in kept_by_ex.get(ex.eid, []):
                parent = ex.beams[path_idx]
                answer_prompts.append(
                    prompt_short_answer_with_context(ex.q_main, parent.hop_steps, parent.prior, expanded_q, para)
                )
                answer_meta.append((ex, rank_key, gate_score, path_idx, raw_sq, expanded_q, para))
        answer_raw = (
            generator.generate_batch(answer_prompts, args.max_new_tokens_answer, desc=f"answer-hop{hop_j}")
            if answer_prompts else []
        )

        rebuilt: dict[str, list[BeamPath]] = defaultdict(list)
        for (ex, rank_key, gate_score, path_idx, raw_sq, expanded_q, para), raw in zip(answer_meta, answer_raw):
            parent = ex.beams[path_idx]
            sub_ans = normalize_short_answer(raw)
            ev_text = (para.get("paragraph_text") or "").strip()
            rebuilt[ex.eid].append(BeamPath(
                hop_steps=parent.hop_steps + [(expanded_q, ev_text)],
                gate_hop_steps=parent.gate_hop_steps + [(raw_sq, ev_text)],
                prior=parent.prior + [sub_ans],
                para_ids=parent.para_ids + [int(para.get("idx", -1))],
                rank_key=rank_key, gate_v2_score=gate_score,
            ))

        for ex in examples:
            if hop_j <= ex.K and ex.eid in rebuilt:
                ex.beams = rebuilt[ex.eid]

        elapsed = time.perf_counter() - t0
        print(f"[hop {hop_j}] {len(texts)} texts scored, elapsed {elapsed / 60:.1f} min", flush=True)

    # Final reader: UNCHANGED -- reads hop_steps (expanded) + prior, not gate_hop_steps.
    final_tasks: list[tuple[ExampleState, BeamPath]] = []
    for ex in examples:
        if not ex.beams:
            continue
        best = max(ex.beams, key=lambda p: p.rank_key)
        final_tasks.append((ex, best))
    final_prompts = [
        build_final_reader_cot_prompt(ex.q_main, best.hop_steps, best.prior) for ex, best in final_tasks
    ]
    final_raw = generator.generate_chat_batch(
        FINAL_READER_SYSTEM_PROMPT, final_prompts, args.max_new_tokens_final, desc="final-reader",
    )

    answer_accum: dict[str, AnswerAccum] = {"all": AnswerAccum(), **{K: AnswerAccum() for K in K_BUCKETS}}
    hop_match_counts: dict[Any, list[int]] = {"all": [0, 0], **{K: [0, 0] for K in K_BUCKETS}}
    chain_match_counts: dict[Any, list[int]] = {"all": [0, 0], **{K: [0, 0] for K in K_BUCKETS}}

    with out_cases.open("w", encoding="utf-8") as f:
        for (ex, best), raw in zip(final_tasks, final_raw):
            fallback = best.prior[-1] if best.prior else ""
            final_answer = parse_final_answer_from_cot(raw, fallback=fallback)
            em, f1 = judge_answer_official(final_answer, ex.row)
            answer_accum["all"].update(em, f1)
            if ex.K in answer_accum:
                answer_accum[ex.K].update(em, f1)

            hop_matches = [pid == gid for pid, gid in zip(best.para_ids, ex.gold_idxs)]
            for key in ("all", ex.K):
                if key in hop_match_counts:
                    hop_match_counts[key][0] += sum(hop_matches)
                    hop_match_counts[key][1] += len(hop_matches)
                    chain_match_counts[key][1] += 1
                    if hop_matches and all(hop_matches):
                        chain_match_counts[key][0] += 1

            f.write(json.dumps({
                "id": ex.eid, "K": ex.K, "question": ex.q_main,
                "final_answer": final_answer, "gold_answer": ex.row.get("answer"),
                "em": em, "f1": f1, "beam_final_rank_key": best.rank_key,
                "beam_final_gate_v2_score": best.gate_v2_score,
                "para_ids": best.para_ids, "gold_idxs": ex.gold_idxs, "prior": best.prior,
            }, ensure_ascii=False) + "\n")

    def rate(counts: list[int]) -> float | None:
        return round(counts[0] / counts[1], 4) if counts[1] else None

    run_wall_sec = time.perf_counter() - run_t0
    results: dict[str, Any] = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "output_dir": str(args.out_dir),
        "n_examples": len(final_tasks),
        "timing": {"wall_clock_sec": round(run_wall_sec, 2), "wall_clock_hr": round(run_wall_sec / 3600, 3)},
        "answer_overall": answer_accum["all"].result(),
        "answer_by_K": {K: answer_accum[K].result() for K in K_BUCKETS},
        "final_beam_hop_match_rate_overall": rate(hop_match_counts["all"]),
        "final_beam_hop_match_rate_by_K": {K: rate(hop_match_counts[K]) for K in K_BUCKETS},
        "final_beam_chain_match_rate_overall": rate(chain_match_counts["all"]),
        "final_beam_chain_match_rate_by_K": {K: rate(chain_match_counts[K]) for K in K_BUCKETS},
        "oracle_gold_rank_among_survivors_overall": oracle_hop_survive["all"].result(),
        "oracle_gold_rank_among_survivors_by_K": {K: oracle_hop_survive[K].result() for K in K_BUCKETS},
        "full_pool_gold_rank_overall": full_pool_rank["all"].result(),
        "full_pool_gold_rank_by_K": {K: full_pool_rank[K].result() for K in K_BUCKETS},
    }
    out_metrics.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nMetrics saved to  {out_metrics}")
    print(f"Cases saved to    {out_cases}")
    print(f"Wall clock hr     {results['timing']['wall_clock_hr']:.3f}")
    print(f"answer_overall    {results['answer_overall']}")
    print(f"hop_match_rate    {results['final_beam_hop_match_rate_overall']}")
    print(f"chain_match_rate  {results['final_beam_chain_match_rate_overall']}")
    print(f"oracle_survival   {results['oracle_gold_rank_among_survivors_overall']}")
    print(f"full_pool_rank    {results['full_pool_gold_rank_overall']} (real recall@1/@3 "
          f"over the full retrieve-k pool, before beam-width truncation)")


if __name__ == "__main__":
    main()
