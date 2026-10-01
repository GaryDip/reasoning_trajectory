#!/usr/bin/env python3
"""
Beam-search retrieval prototype: instead of committing to one evidence
choice per hop (the greedy behavior of retrieval/run_retrieval_exp_wavefront.py),
keep --beam-width parallel candidate paths per example, score every
(path, candidate) continuation with the EXISTING trained gate
(gate/artifacts_pooled_v2 by default), and at each hop keep only the paths
with the lowest CUMULATIVE sum of per-step abnormal scores. No embedding
similarity is used in the cross-path pruning criterion (only to shortlist
each path's own top-k candidates before scoring) — see the design
discussion this script implements: path score = sum of abnormal scores,
no LLM "pick 1 of 3" step (the beam prune IS the selection mechanism), and
the final answer is generated once, from whichever beam has the lowest
cumulative score (no multi-chain judging prompt, to avoid blowing up
context length).

Reuses (imports, does not copy) retrieval/run_retrieval_exp.py and
retrieval/run_retrieval_exp_wavefront.py's data loading, retrieval, batched
gate-scoring (score_gate_requests) and vLLM generation machinery — this
script only replaces the "one committed path" hop loop with a beam one.
Lives here, not in retrieval/, so the production wavefront scripts stay
untouched.

Usage (needs a real GPU + Llama checkpoint + vLLM):
  python beam_search_retrieval.py --limit 20 --beam-width 3
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

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"
GATE_DIR = PROJECT_ROOT / "gate"
sys.path.insert(0, str(RETRIEVAL_DIR))
sys.path.insert(0, str(GATE_DIR))

from run_retrieval_exp import (  # noqa: E402
    MUSIQUE_DIR,
    _get_pipeline_helpers,
    build_trace_prefix,
    embed_retrieval,
    load_dataset_records,
    load_decompose_index,
    normalize_short_answer,
    prepare_sub_questions,
    resolve_decompose_file,
    select_example_ids,
)
from run_retrieval_exp_wavefront import (  # noqa: E402
    FINAL_READER_SYSTEM_PROMPT,
    BatchVllmGenerator,
    GateRequest,
    build_final_reader_cot_prompt,
    load_gate_model,
    parse_final_answer_from_cot,
    prompt_short_answer_with_context,
    score_gate_requests,
)
from lr_artifacts import load_artifacts  # noqa: E402


@dataclass
class BeamPath:
    hop_steps: list[tuple[str, str]] = field(default_factory=list)  # (expanded_q, evidence_text)
    prior: list[str] = field(default_factory=list)  # per-hop short answers, feed [Answer N]
    para_ids: list[int] = field(default_factory=list)  # chosen paragraph idx per hop
    cum_score: float = 0.0  # sum of abnormal (LR gate) scores so far


@dataclass
class BeamExampleState:
    eid: str
    row: dict[str, Any]
    q_main: str
    paragraphs: list[dict[str, Any]]
    sub_questions: list[str]
    K: int
    beams: list[BeamPath]


@dataclass
class _ExK:
    K: int


@dataclass
class _PathCtx:
    """Minimal duck-typed stand-in for HopContext — score_gate_requests only
    ever touches ctx.ex.K, ctx.hop_j, ctx.prefix_before, ctx.expanded_q."""
    ex: _ExK
    hop_j: int
    prefix_before: str
    expanded_q: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["musique"], default="musique")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--musique-dir", type=Path, default=MUSIQUE_DIR)
    ap.add_argument("--decompose-mode", choices=["gt", "bart_decompose"], default="bart_decompose")
    ap.add_argument("--decompose-file", type=Path, default=None)
    ap.add_argument("--gate-artifact-mode", choices=["nonpooled", "pooled"], default="pooled")
    ap.add_argument("--artifacts-dir", type=Path, default=GATE_DIR / "artifacts_pooled_v2")
    ap.add_argument("--beam-width", type=int, default=3)
    ap.add_argument("--retrieve-k", type=int, default=10)
    ap.add_argument("--cos-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--answerable-only", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--gate-device", default="cuda:0")
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--hidden-batch-size", type=int, default=8)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-new-tokens-answer", type=int, default=64)
    ap.add_argument("--max-new-tokens-final", type=int, default=128)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.decompose_file = resolve_decompose_file(
        args.decompose_mode, args.split, args.decompose_file, args.dataset
    )

    ds_args = argparse.Namespace(dataset=args.dataset, musique_dir=args.musique_dir, split=args.split)
    records = load_dataset_records(ds_args)
    decompose_idx = load_decompose_index(args.decompose_file)
    ids = sorted(set(records) & set(decompose_idx))
    if args.answerable_only:
        ids = [eid for eid in ids if records[eid].get("answerable", True)]
    ids = select_example_ids(ids, decompose_idx, limit=args.limit, limit_per_k=0, seed=args.sample_seed)

    (expand_hop_template, _a, _b, _c, _d, _e, judge_answer_official) = _get_pipeline_helpers()

    print(f"Examples: {len(ids)}  beam_width={args.beam_width}  decompose={args.decompose_file}")

    examples: list[BeamExampleState] = []
    for eid in ids:
        row = records[eid]
        sub_qs = prepare_sub_questions(decompose_idx.get(eid) or [], decompose_mode=args.decompose_mode)
        if not sub_qs:
            continue
        examples.append(BeamExampleState(
            eid=eid, row=row, q_main=(row.get("question") or "").strip(),
            paragraphs=row.get("paragraphs") or [], sub_questions=sub_qs,
            K=len(sub_qs), beams=[BeamPath()],
        ))

    artifacts = load_artifacts(args.artifacts_dir, args.gate_artifact_mode)
    print("Loading gate model ...", flush=True)
    gate_model, gate_tokenizer = load_gate_model(args)
    print("Loading vLLM generator ...", flush=True)
    generator = BatchVllmGenerator(
        args.model, dtype=args.dtype, tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
    )

    hidden_cache: dict[tuple[int, str], Any] = {}
    max_hops = max((ex.K for ex in examples), default=0)

    for hop_j in range(1, max_hops + 1):
        t0 = time.perf_counter()
        print(f"\n=== beam hop {hop_j}/{max_hops} ===", flush=True)

        # Step 1: per (example, beam path) — expand subquestion, retrieve top-k candidates.
        expand_retrieve: list[tuple[BeamExampleState, int, str, str, list[tuple[dict, float]]]] = []
        for ex in examples:
            if hop_j > ex.K:
                continue
            raw_sq = ex.sub_questions[hop_j - 1]
            for path_idx, path in enumerate(ex.beams):
                expanded_q = expand_hop_template(raw_sq, path.prior)
                prefix_before = build_trace_prefix(ex.q_main, path.hop_steps)
                candidates_all = embed_retrieval(expanded_q, ex.paragraphs, args.retrieve_k, args.cos_model)
                if not candidates_all:
                    continue
                expand_retrieve.append((ex, path_idx, expanded_q, prefix_before, candidates_all))
        print(f"[hop {hop_j}] {len(expand_retrieve)} (example, path) contexts to score", flush=True)

        # Step 2: batch-score EVERY (path, candidate) continuation with the gate.
        gate_reqs: list[GateRequest] = []
        req_meta: dict[int, tuple[BeamExampleState, int, str]] = {}
        for req_id, (ex, path_idx, expanded_q, prefix_before, candidates_all) in enumerate(expand_retrieve):
            ctx = _PathCtx(ex=_ExK(ex.K), hop_j=hop_j, prefix_before=prefix_before, expanded_q=expanded_q)
            gate_reqs.append(GateRequest(req_id, ctx, candidates_all))
            req_meta[req_id] = (ex, path_idx, expanded_q)

        scored = (
            score_gate_requests(
                gate_reqs, artifacts=artifacts, gate_artifact_mode=args.gate_artifact_mode,
                model=gate_model, tokenizer=gate_tokenizer, default_layer=args.layer,
                hidden_batch_size=args.hidden_batch_size, hidden_cache=hidden_cache,
                desc=f"hop{hop_j} beam-gate",
            )
            if gate_reqs else {}
        )

        # Step 3: group continuations by example, keep only the top beam_width by cumulative score.
        continuations_by_ex: dict[str, list[tuple[float, int, str, dict]]] = defaultdict(list)
        for req_id, (ex, path_idx, expanded_q) in req_meta.items():
            parent_score = ex.beams[path_idx].cum_score
            for para, _emb_score, abnormal in scored[req_id]:
                continuations_by_ex[ex.eid].append((parent_score + abnormal, path_idx, expanded_q, para))

        kept_by_ex: dict[str, list[tuple[float, int, str, dict]]] = {}
        for ex in examples:
            conts = continuations_by_ex.get(ex.eid)
            if not conts:
                continue
            conts.sort(key=lambda t: t[0])  # lower cumulative abnormal score = better
            kept_by_ex[ex.eid] = conts[: args.beam_width]

        # Step 4: generate a hop-answer via vLLM ONLY for surviving continuations.
        answer_prompts: list[str] = []
        answer_meta: list[tuple[BeamExampleState, float, int, str, dict]] = []
        for ex in examples:
            for score, path_idx, expanded_q, para in kept_by_ex.get(ex.eid, []):
                parent = ex.beams[path_idx]
                answer_prompts.append(
                    prompt_short_answer_with_context(ex.q_main, parent.hop_steps, parent.prior, expanded_q, para)
                )
                answer_meta.append((ex, score, path_idx, expanded_q, para))
        answer_raw = (
            generator.generate_batch(answer_prompts, args.max_new_tokens_answer, desc=f"beam-answer-hop{hop_j}")
            if answer_prompts else []
        )

        # Step 5: materialize the new beams.
        rebuilt: dict[str, list[BeamPath]] = defaultdict(list)
        for (ex, score, path_idx, expanded_q, para), raw in zip(answer_meta, answer_raw):
            parent = ex.beams[path_idx]
            sub_ans = normalize_short_answer(raw)
            ev_text = (para.get("paragraph_text") or "").strip()
            rebuilt[ex.eid].append(BeamPath(
                hop_steps=parent.hop_steps + [(expanded_q, ev_text)],
                prior=parent.prior + [sub_ans],
                para_ids=parent.para_ids + [int(para.get("idx", -1))],
                cum_score=score,
            ))

        for ex in examples:
            if hop_j <= ex.K and ex.eid in rebuilt:
                ex.beams = rebuilt[ex.eid]
            # examples already finished (hop_j > ex.K) keep their final beams untouched.

        elapsed = time.perf_counter() - t0
        print(f"[hop {hop_j}] {len(gate_reqs)} candidates scored, elapsed {elapsed / 60:.1f} min", flush=True)

    # Final answer: best (lowest cumulative score) beam per example -> single final-reader generation.
    final_tasks: list[tuple[BeamExampleState, BeamPath]] = []
    for ex in examples:
        if not ex.beams:
            continue
        best = min(ex.beams, key=lambda p: p.cum_score)
        final_tasks.append((ex, best))
    final_prompts = [
        build_final_reader_cot_prompt(ex.q_main, best.hop_steps, best.prior) for ex, best in final_tasks
    ]
    final_raw = generator.generate_chat_batch(
        FINAL_READER_SYSTEM_PROMPT, final_prompts, args.max_new_tokens_final, desc="beam-final-reader",
    )

    out_path = args.out or (HERE / "results" / f"beam_search_{args.dataset}_{args.split}_bw{args.beam_width}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_scored = n_em = 0
    n_f1 = 0.0
    with out_path.open("w", encoding="utf-8") as f:
        for (ex, best), raw in zip(final_tasks, final_raw):
            fallback = best.prior[-1] if best.prior else ""
            final_answer = parse_final_answer_from_cot(raw, fallback=fallback)
            em, f1 = judge_answer_official(final_answer, ex.row)
            n_scored += 1
            n_em += int(em or 0)
            n_f1 += float(f1 or 0.0)
            f.write(json.dumps({
                "id": ex.eid, "K": ex.K, "question": ex.q_main,
                "final_answer": final_answer, "gold_answer": ex.row.get("answer"),
                "em": em, "f1": f1, "beam_cum_score": best.cum_score,
                "para_ids": best.para_ids, "prior": best.prior,
            }, ensure_ascii=False) + "\n")

    print(f"\nBeam search done: {n_scored} examples -> {out_path}")
    if n_scored:
        print(f"EM={n_em / n_scored:.4f} F1={n_f1 / n_scored:.4f}")


if __name__ == "__main__":
    main()
