#!/usr/bin/env python3
"""
Wavefront-style (batched across ALL examples per hop) end-to-end retrieval + answer
generation using the trained gate model (see model.py / train_lambda_model.py) instead
of a fixed --lambda-lr. Both emb_score (BGE) and abnormal_score (the
error_propagation_probe) stay frozen; per hop, within that hop's own candidate pool, both
are z-scored (comparable scale) and combined as a bounded convex combination:

    emb_z_i  = zscore(emb_score)_i        (pool-relative, "higher = better", unchanged direction)
    abn_z_i  = zscore(-abnormal_score)_i  (pool-relative, sign flipped so "higher = better" too)
    final_score_i = gate * emb_z_i + (1 - gate) * abn_z_i,   gate in (0, 1)

This replaced an earlier version that plugged an open-ended `lambda` straight into
`emb_score - lambda * abnormal_score` — that ran away chasing lower training loss (no
natural stopping point) and, once capped, just collapsed to a constant at the cap. A
bounded gate over pre-calibrated signals can't do either: neither signal can be weighted
"infinitely more" than the other since the weights must sum to 1. gate is predicted per
hop by the model from (q_main_emb, h_prev, expanded_q_emb), then applied uniformly to
every candidate in that hop's pool (never per-candidate — this only decides HOW MUCH to
trust cosine vs the probe for this kind of hop, not which candidate is best on its own).

Single committed path per example by default (--beam-width 1), matching production's own
greedy structure for a direct apples-to-apples comparison against --lambda-lr fixed
baselines; --beam-width > 1 combines this with beam search if you want to try both ideas
together, reusing the exact same keep-top-N mechanism either way.

Needs (train first if missing):
  - error_propagation_probe/probe_artifacts/{pca,lr}.joblib  (the frozen abnormal-score probe)
  - adaptive_lambda_rerank/lambda_artifacts/lambda_model.pt  (train_lambda_model.py)

Reuses (imports, does not copy) retrieval/run_retrieval_exp.py and
retrieval/run_retrieval_exp_wavefront.py's data loading, retrieval, batched hidden-state
extraction, accumulators and vLLM generation machinery, plus
error_propagation_probe/wavefront_hidden_probe_beam.py::score_prefixes for the per-candidate
probe scoring (same code, not copied) — production files are untouched.

Usage:
  python wavefront_adaptive_lambda.py --limit 20
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
PROBE_DIR = PROJECT_ROOT / "error_propagation_probe"
sys.path.insert(0, str(RETRIEVAL_DIR))
sys.path.insert(0, str(TRACES_DIR))
sys.path.insert(0, str(PROBE_DIR))
sys.path.insert(0, str(HERE))

from run_retrieval_exp import (  # noqa: E402
    K_BUCKETS,
    MUSIQUE_DIR,
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
)
from trace_evidence import gold_evidence_musique  # noqa: E402
from wavefront_hidden_probe_beam import score_prefixes  # noqa: E402

from model import LambdaModel  # noqa: E402
from train_lambda_model import zscore  # noqa: E402


def load_probe(probe_dir: Path):
    import joblib
    pca = joblib.load(probe_dir / "pca.joblib")
    lr = joblib.load(probe_dir / "lr.joblib")
    return pca, lr


def load_lambda_model(lambda_dir: Path, device: str) -> LambdaModel:
    meta = json.loads((lambda_dir / "meta.json").read_text(encoding="utf-8"))
    model = LambdaModel(hidden=meta["hidden"], use_diff=meta["use_diff"]).to(device)
    model.load_state_dict(torch.load(lambda_dir / "lambda_model.pt", map_location=device))
    model.eval()
    return model, meta


@dataclass
class BeamPath:
    hop_steps: list[tuple[str, str]] = field(default_factory=list)
    prior: list[str] = field(default_factory=list)
    para_ids: list[int] = field(default_factory=list)
    rank_key: float = 0.0
    probe_score: float = 0.0
    gate_used: float = 0.0


@dataclass
class ExampleState:
    eid: str
    row: dict[str, Any]
    q_main: str
    q_main_emb: np.ndarray  # BGE, computed once per example (not per hop)
    paragraphs: list[dict[str, Any]]
    sub_questions: list[str]
    gold_idxs: list[int]
    K: int
    beams: list[BeamPath]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["musique"], default="musique")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--musique-dir", type=Path, default=MUSIQUE_DIR)
    ap.add_argument("--decompose-mode", choices=["gt", "bart_decompose"], default="bart_decompose")
    ap.add_argument("--decompose-file", type=Path, default=None)
    ap.add_argument("--probe-dir", type=Path, default=PROBE_DIR / "probe_artifacts")
    ap.add_argument("--lambda-dir", type=Path, default=HERE / "lambda_artifacts")
    ap.add_argument("--beam-width", type=int, default=1,
                     help="1 = single committed path per example (fair comparison against "
                          "production's own greedy lr_rerank). >1 combines this with beam search.")
    ap.add_argument("--retrieve-k", type=int, default=10)
    ap.add_argument("--cos-model", default="BAAI/bge-base-en-v1.5")
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
    ap.add_argument("--layer", type=int, default=31)
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

    ds_args = argparse.Namespace(dataset=args.dataset, musique_dir=args.musique_dir, split=args.split)
    records = load_dataset_records(ds_args)
    decompose_idx = load_decompose_index(args.decompose_file)
    ids = sorted(set(records) & set(decompose_idx))
    if args.answerable_only:
        ids = [eid for eid in ids if records[eid].get("answerable", True)]
    ids = select_example_ids(ids, decompose_idx, limit=args.limit, limit_per_k=0, seed=args.sample_seed)

    tag = args.run_tag or f"adaptive_lambda_{args.dataset}_bw{args.beam_width}"
    args.out_dir = resolve_output_dir(
        out_dir=args.out_dir, results_root=args.results_root, run_tag=tag,
        decompose_mode=args.decompose_mode, methods=["adaptive_lambda"],
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{sanitize_run_tag(args.run_tag)}" if args.run_tag else ""
    out_cases = args.out_dir / f"retrieval_cases_{args.split}{tag_suffix}.jsonl"
    out_metrics = args.out_dir / f"retrieval_exp_{args.split}{tag_suffix}.json"
    print(f"Output directory: {args.out_dir}")

    (expand_hop_template, *_rest, judge_answer_official) = _get_pipeline_helpers()

    device = args.gate_device if torch.cuda.is_available() else "cpu"
    pca, lr = load_probe(args.probe_dir)
    lambda_model, lambda_meta = load_lambda_model(args.lambda_dir, device)
    print(f"Probe: {args.probe_dir}   Lambda model: {args.lambda_dir} ({lambda_meta})")

    from sentence_transformers import SentenceTransformer
    print(f"Loading BGE ({args.cos_model}) for context embeddings ...", flush=True)
    st = SentenceTransformer(args.cos_model)

    print(f"Examples: {len(ids)}  beam_width={args.beam_width}  decompose={args.decompose_file}")
    examples: list[ExampleState] = []
    q_mains = []
    for eid in ids:
        row = records[eid]
        sub_qs = prepare_sub_questions(decompose_idx.get(eid) or [], decompose_mode=args.decompose_mode)
        if not sub_qs:
            continue
        _gold_texts, gold_idxs = gold_evidence_musique(row)
        q_main = (row.get("question") or "").strip()
        examples.append(ExampleState(
            eid=eid, row=row, q_main=q_main, q_main_emb=None,  # filled in below
            paragraphs=row.get("paragraphs") or [], sub_questions=sub_qs,
            gold_idxs=gold_idxs, K=len(sub_qs), beams=[BeamPath()],
        ))
        q_mains.append(q_main)

    print(f"Encoding {len(examples)} main questions ...", flush=True)
    q_main_vecs = st.encode(q_mains, normalize_embeddings=True, show_progress_bar=True, batch_size=256)
    for ex, vec in zip(examples, q_main_vecs):
        ex.q_main_emb = vec

    print("Loading Llama model for hidden-state scoring ...", flush=True)
    gate_model, gate_tokenizer = load_gate_model(args)
    print("Loading vLLM generator ...", flush=True)
    generator = BatchVllmGenerator(
        args.model, dtype=args.dtype, tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
    )

    hidden_cache: dict[tuple[int, str], Any] = {}
    max_hops = max((ex.K for ex in examples), default=0)
    oracle_hop_survive: dict[Any, MetricAccum] = {"all": MetricAccum(), **{K: MetricAccum() for K in K_BUCKETS}}
    gate_values_seen: list[float] = []

    for hop_j in range(1, max_hops + 1):
        t0 = time.perf_counter()
        print(f"\n=== hop {hop_j}/{max_hops} ===", flush=True)

        expand_retrieve: list[tuple[ExampleState, int, str, list[tuple[dict, float]]]] = []
        for ex in examples:
            if hop_j > ex.K:
                continue
            raw_sq = ex.sub_questions[hop_j - 1]
            for path_idx, path in enumerate(ex.beams):
                expanded_q = expand_hop_template(raw_sq, path.prior)
                candidates_all = embed_retrieval(expanded_q, ex.paragraphs, args.retrieve_k, args.cos_model)
                if not candidates_all:
                    continue
                expand_retrieve.append((ex, path_idx, expanded_q, candidates_all))
        print(f"[hop {hop_j}] {len(expand_retrieve)} (example, path) contexts", flush=True)

        # 1) h_prev per (example, path) context — ONE Llama text each (prefix_before only).
        prefix_texts = [
            build_trace_prefix(ex.q_main, ex.beams[path_idx].hop_steps)
            for ex, path_idx, _eq, _cands in expand_retrieve
        ]
        h_prev_by_layer = batch_last_hidden(
            texts=prefix_texts, model=gate_model, tokenizer=gate_tokenizer, layers=[args.layer],
            batch_size=args.hidden_batch_size, cache=hidden_cache, desc=f"hop{hop_j} h_prev",
        )
        h_prev_arr = np.stack(h_prev_by_layer[args.layer]).astype(np.float32)

        # 2) BGE-encode expanded_q for every context, then predict a gate per context.
        expanded_q_vecs = st.encode(
            [eq for _ex, _pi, eq, _c in expand_retrieve], normalize_embeddings=True, batch_size=256,
        )
        q_main_arr = np.stack([ex.q_main_emb for ex, _pi, _eq, _c in expand_retrieve])
        with torch.no_grad():
            gates = lambda_model(
                torch.tensor(q_main_arr, dtype=torch.float32, device=device),
                torch.tensor(h_prev_arr, dtype=torch.float32, device=device),
                torch.tensor(np.asarray(expanded_q_vecs, dtype=np.float32), device=device),
            ).cpu().numpy()
        gate_values_seen.extend(gates.tolist())

        # 3) probe-score every (context, candidate) pair.
        texts: list[str] = []
        owners: list[tuple[int, dict, float]] = []  # (context index, candidate para, emb_score)
        for ctx_idx, (ex, path_idx, expanded_q, candidates_all) in enumerate(expand_retrieve):
            parent = ex.beams[path_idx]
            for para, emb_score in candidates_all:
                ev_text = (para.get("paragraph_text") or "").strip()
                prefix = build_trace_prefix(ex.q_main, parent.hop_steps + [(expanded_q, ev_text)])
                texts.append(prefix)
                owners.append((ctx_idx, para, emb_score))

        probe_scores = score_prefixes(
            texts, model=gate_model, tokenizer=gate_tokenizer, layer=args.layer,
            batch_size=args.hidden_batch_size, pca=pca, lr=lr, hidden_cache=hidden_cache,
            desc=f"hop{hop_j} probe-score",
        )

        # Group by context FIRST so emb_score/probe_score get z-scored within their own
        # candidate pool (same formula training used — see train_lambda_model.py) before
        # combining via the gate, instead of combining raw, differently-scaled numbers.
        by_ctx: dict[int, list[tuple[dict, float, float]]] = defaultdict(list)
        for (ctx_idx, para, emb_score), probe_score in zip(owners, probe_scores):
            by_ctx[ctx_idx].append((para, emb_score, probe_score))

        continuations_by_ex: dict[str, list[tuple[float, float, float, int, str, dict]]] = defaultdict(list)
        for ctx_idx, cands in by_ctx.items():
            ex, path_idx, expanded_q, _cands = expand_retrieve[ctx_idx]
            gate = float(gates[ctx_idx])
            emb_arr = np.asarray([c[1] for c in cands], dtype=np.float64)
            probe_arr = np.asarray([c[2] for c in cands], dtype=np.float64)
            emb_z = zscore(emb_arr)
            abn_z = zscore(-probe_arr)  # flip sign: higher = less corrupted = better
            rank_keys = gate * emb_z + (1 - gate) * abn_z
            for (para, _emb_score, probe_score), rank_key in zip(cands, rank_keys):
                continuations_by_ex[ex.eid].append(
                    (float(rank_key), probe_score, gate, path_idx, expanded_q, para)
                )

        kept_by_ex: dict[str, list[tuple[float, float, float, int, str, dict]]] = {}
        for ex in examples:
            conts = continuations_by_ex.get(ex.eid)
            if not conts:
                continue
            conts.sort(key=lambda t: t[0], reverse=True)
            kept_by_ex[ex.eid] = conts[: args.beam_width]

            gold_idx = ex.gold_idxs[hop_j - 1] if hop_j - 1 < len(ex.gold_idxs) else None
            if gold_idx is not None:
                survivor_ranked = [(para, rk) for rk, _ps, _l, _p, _q, para in kept_by_ex[ex.eid]]
                rank = find_gold_rank(survivor_ranked, gold_idx)
                oracle_hop_survive["all"].update(rank)
                if ex.K in oracle_hop_survive:
                    oracle_hop_survive[ex.K].update(rank)

        answer_prompts: list[str] = []
        answer_meta: list[tuple[ExampleState, float, float, float, int, str, dict]] = []
        for ex in examples:
            for rank_key, probe_score, gate_val, path_idx, expanded_q, para in kept_by_ex.get(ex.eid, []):
                parent = ex.beams[path_idx]
                answer_prompts.append(
                    prompt_short_answer_with_context(ex.q_main, parent.hop_steps, parent.prior, expanded_q, para)
                )
                answer_meta.append((ex, rank_key, probe_score, gate_val, path_idx, expanded_q, para))
        answer_raw = (
            generator.generate_batch(answer_prompts, args.max_new_tokens_answer, desc=f"answer-hop{hop_j}")
            if answer_prompts else []
        )

        rebuilt: dict[str, list[BeamPath]] = defaultdict(list)
        for (ex, rank_key, probe_score, gate_val, path_idx, expanded_q, para), raw in zip(answer_meta, answer_raw):
            parent = ex.beams[path_idx]
            sub_ans = normalize_short_answer(raw)
            ev_text = (para.get("paragraph_text") or "").strip()
            rebuilt[ex.eid].append(BeamPath(
                hop_steps=parent.hop_steps + [(expanded_q, ev_text)],
                prior=parent.prior + [sub_ans],
                para_ids=parent.para_ids + [int(para.get("idx", -1))],
                rank_key=rank_key, probe_score=probe_score, gate_used=gate_val,
            ))

        for ex in examples:
            if hop_j <= ex.K and ex.eid in rebuilt:
                ex.beams = rebuilt[ex.eid]

        elapsed = time.perf_counter() - t0
        print(f"[hop {hop_j}] {len(texts)} candidates scored, elapsed {elapsed / 60:.1f} min", flush=True)

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
                "beam_final_probe_score": best.probe_score, "beam_final_gate": best.gate_used,
                "para_ids": best.para_ids, "gold_idxs": ex.gold_idxs, "prior": best.prior,
            }, ensure_ascii=False) + "\n")

    def rate(counts: list[int]) -> float | None:
        return round(counts[0] / counts[1], 4) if counts[1] else None

    run_wall_sec = time.perf_counter() - run_t0
    gate_arr = np.asarray(gate_values_seen, dtype=np.float64)
    results: dict[str, Any] = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "output_dir": str(args.out_dir),
        "n_examples": len(final_tasks),
        "timing": {"wall_clock_sec": round(run_wall_sec, 2), "wall_clock_hr": round(run_wall_sec / 3600, 3)},
        "gate_stats": {
            "n": int(gate_arr.size), "mean": float(gate_arr.mean()) if gate_arr.size else None,
            "std": float(gate_arr.std()) if gate_arr.size else None,
            "min": float(gate_arr.min()) if gate_arr.size else None,
            "max": float(gate_arr.max()) if gate_arr.size else None,
        },
        "answer_overall": answer_accum["all"].result(),
        "answer_by_K": {K: answer_accum[K].result() for K in K_BUCKETS},
        "final_beam_hop_match_rate_overall": rate(hop_match_counts["all"]),
        "final_beam_hop_match_rate_by_K": {K: rate(hop_match_counts[K]) for K in K_BUCKETS},
        "final_beam_chain_match_rate_overall": rate(chain_match_counts["all"]),
        "final_beam_chain_match_rate_by_K": {K: rate(chain_match_counts[K]) for K in K_BUCKETS},
        "oracle_gold_rank_among_survivors_overall": oracle_hop_survive["all"].result(),
        "oracle_gold_rank_among_survivors_by_K": {K: oracle_hop_survive[K].result() for K in K_BUCKETS},
    }
    out_metrics.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nMetrics saved to  {out_metrics}")
    print(f"Cases saved to    {out_cases}")
    print(f"Wall clock hr     {results['timing']['wall_clock_hr']:.3f}")
    print(f"gate stats        {results['gate_stats']}")
    print(f"answer_overall    {results['answer_overall']}")
    print(f"hop_match_rate    {results['final_beam_hop_match_rate_overall']}")
    print(f"chain_match_rate  {results['final_beam_chain_match_rate_overall']}")
    print(f"oracle_survival   {results['oracle_gold_rank_among_survivors_overall']}")


if __name__ == "__main__":
    main()
