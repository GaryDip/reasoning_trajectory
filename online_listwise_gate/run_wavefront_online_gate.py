#!/usr/bin/env python3
"""
Clean (non-training) end-to-end wavefront run using the frozen model produced by
train_online_wavefront.py -- retrieval, per-hop reranking, per-hop short answers, and the
final reader, all in one pass, with NO further model updates (this is the "second, clean
pass" the training script's docstring says is needed, since decisions made early in the
online-training run used a barely-trained model).

Reranking: this model's score already incorporates the raw BGE query/passage vectors (via
PCA), so unlike gate v2/v3's `final_score = emb_score - lambda*gate_score`, there is no
separate cosine term to mix in -- the model's own output IS the rank key.

Supports both decompose modes (unlike the training script, which is GT-only/MuSiQue-only):
`--decompose-mode gt` only works for musique (matches train_online_wavefront.py's own
training distribution most closely); `--decompose-mode bart_decompose` works on all three
datasets and is what a real cross-dataset evaluation (matching how gate v2/v3 are evaluated)
needs, since BART is the only decomposer available at real inference time.

Text stream: single accumulated hop_steps built from EXPANDED subquestions throughout (no
"rawprefix" dual-track needed here, unlike gate v2/v3) -- the training script builds its own
training data the same way (build_trace_prefix over expanded hop_steps), so there is no
train/inference text-distribution mismatch to correct for in the first place.

Usage:
  python run_wavefront_online_gate.py --artifact results/<run>/artifact.joblib --limit 20
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
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"
TRACES_DIR = PROJECT_ROOT / "traces"
sys.path.insert(0, str(TRACES_DIR))
sys.path.insert(0, str(RETRIEVAL_DIR))
sys.path.insert(0, str(HERE))

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
    build_final_reader_cot_prompt_comparison_hint,
    load_gate_model,
    parse_final_answer_from_cot,
    prompt_short_answer_with_context,
    warmup_gate_model_memory,
)
from trace_evidence import gold_evidence_musique  # noqa: E402
from trace_format import escape_double_quotes  # noqa: E402
from model import load_artifact  # noqa: E402
from train_online_wavefront import encode_query_and_docs  # noqa: E402

FINAL_READER_PROMPT_BUILDERS = {
    "default": build_final_reader_cot_prompt,
    "comparison_hint": build_final_reader_cot_prompt_comparison_hint,
}


@dataclass
class BeamPath:
    hop_steps: list[tuple[str, str]] = field(default_factory=list)
    prior: list[str] = field(default_factory=list)
    para_ids: list[int] = field(default_factory=list)
    rank_key: float = 0.0
    gate_score: float = 0.0


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
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--dataset", choices=["musique", "2wiki", "hotpot"], default="musique")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--musique-dir", type=Path, default=MUSIQUE_DIR)
    ap.add_argument("--twowiki-file", type=Path, default=TWOWIKI_DEV_FILE)
    ap.add_argument("--hotpot-file", type=Path, default=HOTPOT_DEV_FILE)
    ap.add_argument("--decompose-mode", choices=["gt", "bart_decompose"], default="bart_decompose")
    ap.add_argument("--decompose-file", type=Path, default=None)
    ap.add_argument("--beam-width", type=int, default=1)
    ap.add_argument("--retrieve-k", type=int, default=10)
    ap.add_argument("--cos-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument(
        "--query-instruction",
        default="Represent this sentence for searching relevant passages: ",
        help="BGE's own asymmetric-retrieval instruction prefix, applied to the query text "
             "passed into embed_retrieval (both for the candidate pool AND for this "
             "script's own raw-vector encoding, so the model sees the same query "
             "representation used for the pool it's reranking). Pass '' to disable.",
    )
    ap.add_argument("--final-reader-prompt", choices=list(FINAL_READER_PROMPT_BUILDERS), default="comparison_hint")
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

    tag = args.run_tag or f"online_gate_{args.dataset}_bw{args.beam_width}"
    out_dir = args.out_dir or resolve_output_dir(
        out_dir=None, results_root=args.results_root, run_tag=tag,
        decompose_mode=args.decompose_mode, methods=["online_gate"],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{sanitize_run_tag(args.run_tag)}" if args.run_tag else ""
    out_cases = out_dir / f"retrieval_cases_{args.split}{tag_suffix}.jsonl"
    out_metrics = out_dir / f"retrieval_exp_{args.split}{tag_suffix}.json"
    print(f"Output directory: {out_dir}")

    (expand_hop_template, *_rest, judge_answer_official) = _get_pipeline_helpers()

    feature_pca, listwise_model, artifact_meta = load_artifact(args.artifact)
    listwise_model.eval()
    layer_by_j = {int(k): v for k, v in artifact_meta["layer_by_j"].items()}
    print(f"Loaded online listwise gate: {args.artifact} (trained on {artifact_meta['trained_on']}, "
          f"layer_by_j={layer_by_j})")

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
    full_pool_rank: dict[Any, MetricAccum] = {"all": MetricAccum(), **{K: MetricAccum() for K in K_BUCKETS}}

    for hop_j in range(1, max_hops + 1):
        t0 = time.perf_counter()
        print(f"\n=== hop {hop_j}/{max_hops} ===", flush=True)
        layer = layer_by_j.get(hop_j - 1, 23)

        expand_retrieve: list[tuple[ExampleState, int, str, list[dict], np.ndarray, np.ndarray]] = []
        for ex in examples:
            if hop_j > ex.K:
                continue
            raw_sq = ex.sub_questions[hop_j - 1]
            for path_idx, path in enumerate(ex.beams):
                expanded_q = expand_hop_template(raw_sq, path.prior)
                query_text = f"{args.query_instruction}{expanded_q}"
                ranked = embed_retrieval(query_text, ex.paragraphs, args.retrieve_k, args.cos_model)
                paras = [p for p, _s in ranked]
                if not paras:
                    continue
                docs = [f"{p.get('title') or ''}\n{(p.get('paragraph_text') or '').strip()}" for p in paras]
                qv, dv = encode_query_and_docs(query_text, docs, args.cos_model)
                expand_retrieve.append((ex, path_idx, expanded_q, paras, qv, dv))
        print(f"[hop {hop_j}] {len(expand_retrieve)} (example, path) contexts", flush=True)

        texts: list[str] = []
        owners: list[tuple[int, int]] = []
        for ctx_idx, (ex, path_idx, expanded_q, paras, _qv, _dv) in enumerate(expand_retrieve):
            parent = ex.beams[path_idx]
            prefix_before = build_trace_prefix(ex.q_main, parent.hop_steps)
            texts.append(prefix_before)
            owners.append((ctx_idx, -1))
            for slot, para in enumerate(paras):
                ev_text = (para.get("paragraph_text") or "").strip()
                text = f'{prefix_before} Step {hop_j}: {expanded_q} Evidence: "{escape_double_quotes(ev_text)}"'
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

        continuations_by_ex: dict[str, list[tuple[float, float, int, str, dict]]] = defaultdict(list)
        for ctx_idx, (ex, path_idx, expanded_q, paras, qv, dv) in enumerate(expand_retrieve):
            h_prev = prefix_hidden[ctx_idx]
            n = len(paras)
            h_after = np.stack([cand_hidden[ctx_idx][s] for s in range(n)], axis=0)
            delta = h_after - h_prev
            q_rep = np.repeat(qv[None, :], n, axis=0)
            feats = feature_pca.transform(q_rep, dv, h_after, delta)
            with torch.no_grad():
                scores = listwise_model(torch.tensor(feats, dtype=torch.float32)).numpy()
            for slot, para in enumerate(paras):
                rank_key = float(scores[slot])
                continuations_by_ex[ex.eid].append((rank_key, rank_key, path_idx, expanded_q, para))

        kept_by_ex: dict[str, list[tuple[float, float, int, str, dict]]] = {}
        for ex in examples:
            conts = continuations_by_ex.get(ex.eid)
            if not conts:
                continue
            conts.sort(key=lambda t: t[0], reverse=True)
            kept_by_ex[ex.eid] = conts[: args.beam_width]

            gold_idx = ex.gold_idxs[hop_j - 1] if hop_j - 1 < len(ex.gold_idxs) else None
            if gold_idx is not None:
                full_ranked = [(para, rk) for rk, _g, _p, _eq, para in conts]
                full_rank = find_gold_rank(full_ranked, gold_idx)
                full_pool_rank["all"].update(full_rank)
                if ex.K in full_pool_rank:
                    full_pool_rank[ex.K].update(full_rank)

                survivor_ranked = [(para, rk) for rk, _g, _p, _eq, para in kept_by_ex[ex.eid]]
                rank = find_gold_rank(survivor_ranked, gold_idx)
                oracle_hop_survive["all"].update(rank)
                if ex.K in oracle_hop_survive:
                    oracle_hop_survive[ex.K].update(rank)

        answer_prompts: list[str] = []
        answer_meta: list[tuple[ExampleState, float, float, int, str, dict]] = []
        for ex in examples:
            for rank_key, gate_score, path_idx, expanded_q, para in kept_by_ex.get(ex.eid, []):
                parent = ex.beams[path_idx]
                answer_prompts.append(
                    prompt_short_answer_with_context(ex.q_main, parent.hop_steps, parent.prior, expanded_q, para)
                )
                answer_meta.append((ex, rank_key, gate_score, path_idx, expanded_q, para))
        answer_raw = (
            generator.generate_batch(answer_prompts, args.max_new_tokens_answer, desc=f"answer-hop{hop_j}")
            if answer_prompts else []
        )

        rebuilt: dict[str, list[BeamPath]] = defaultdict(list)
        for (ex, rank_key, gate_score, path_idx, expanded_q, para), raw in zip(answer_meta, answer_raw):
            parent = ex.beams[path_idx]
            sub_ans = normalize_short_answer(raw)
            ev_text = (para.get("paragraph_text") or "").strip()
            rebuilt[ex.eid].append(BeamPath(
                hop_steps=parent.hop_steps + [(expanded_q, ev_text)],
                prior=parent.prior + [sub_ans],
                para_ids=parent.para_ids + [int(para.get("idx", -1))],
                rank_key=rank_key, gate_score=gate_score,
            ))

        for ex in examples:
            if hop_j <= ex.K and ex.eid in rebuilt:
                ex.beams = rebuilt[ex.eid]

        elapsed = time.perf_counter() - t0
        print(f"[hop {hop_j}] {len(texts)} texts scored, elapsed {elapsed / 60:.1f} min", flush=True)

    final_reader_prompt_fn = FINAL_READER_PROMPT_BUILDERS[args.final_reader_prompt]
    final_tasks: list[tuple[ExampleState, BeamPath]] = []
    for ex in examples:
        if not ex.beams:
            continue
        best = max(ex.beams, key=lambda p: p.rank_key)
        final_tasks.append((ex, best))
    final_prompts = [
        final_reader_prompt_fn(ex.q_main, best.hop_steps, best.prior) for ex, best in final_tasks
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
                "para_ids": best.para_ids, "gold_idxs": ex.gold_idxs, "prior": best.prior,
            }, ensure_ascii=False) + "\n")

    def rate(counts: list[int]) -> float | None:
        return round(counts[0] / counts[1], 4) if counts[1] else None

    run_wall_sec = time.perf_counter() - run_t0
    results: dict[str, Any] = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "output_dir": str(out_dir),
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
    print(f"full_pool_rank    {results['full_pool_gold_rank_overall']}")


if __name__ == "__main__":
    main()
