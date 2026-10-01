#!/usr/bin/env python3
"""
Wavefront-style (batched across ALL examples per hop, comparable in structure to
retrieval/run_retrieval_exp_wavefront.py) beam search using the error_propagation_probe
h_j probe (PCA+LR fit on pooled prefix positions — see score_and_analyze.py
--pool-positions) instead of the production Delta-based gate.

Beam algorithm (this is the exact scheme discussed, NOT beam_search_retrieval.py's
cumulative-Delta-sum design):
  hop 1: for every top-`--retrieve-k` cosine candidate, build the FULL prefix
  "Question: ... Step 1: <q> Evidence: "<candidate>"" (build_trace_prefix — same format
  assemble_trace/cumulative_prefix_strings used to build the probe's own training data),
  score it directly with the probe (absolute h_1, not a delta, not a running sum: h_j
  already encodes "how much went wrong up to here" on its own, per error_propagation_probe's
  validated finding), keep the --beam-width best-ranked prefixes.
  hop j>1: for every surviving beam x every candidate at hop j, build ITS OWN full prefix
  (parent's hop_steps + this candidate), score the WHOLE new prefix directly (not
  incrementally), keep the --beam-width best-ranked prefixes GLOBALLY (not per-parent).
  end: the single best-ranked final beam -> one final-reader generation (same choice
  beam_search_retrieval.py made, to avoid a multi-chain judging prompt blowing up context).

--rerank-mode controls what "best-ranked" means (added after a first full-dev run came in
clearly WORSE than production baseline across recall/EM/F1 — see conversation): production
NEVER ranks candidates by its own gate score alone (lr_rerank mixes emb_score - lam*abnormal,
gated_rule_a only uses the gate score as a binary expand/no-expand trigger) — probe_only mode
here was an untested new use case (pure probe-only ranking), not a re-run of something already
validated elsewhere.
  probe_only (original/default): rank_key = -probe_score (lower P(corrupted) = better).
  weighted: rank_key = emb_score - lambda_probe * probe_score, mirroring production's
  rerank_by_final_score(emb_score, abnormal_score, lam) in retrieval/run_retrieval_exp.py —
  cosine similarity re-enters the ranking instead of only shortlisting the top-k pool.

Needs a probe already fit and saved via:
  python score_and_analyze.py --train-hidden-dir hidden_states_full/train \
      --dev-hidden-dir hidden_states_full/dev --pool-positions --out-dir probe_artifacts
(--pool-positions matters here specifically: this script scores PARTIAL prefixes at every
hop, not just a completed h_K. A probe fit only on final h_K would be zero-shot-generalized
to interior positions instead of trained for them.)

Reuses (imports, does not copy) retrieval/run_retrieval_exp.py and
retrieval/run_retrieval_exp_wavefront.py's data loading, retrieval, batched hidden-state
extraction and vLLM generation machinery, plus traces/trace_evidence.py's gold-evidence
helper for metrics — production files are untouched.

Metrics note: production's MetricAccum/ChainAccum assume ONE committed evidence choice per
hop with a well-defined rank in a scored pool. Beam search keeps --beam-width survivors per
hop with no single committed choice until the very end, so "gold rank among survivors" is
reported per hop as an oracle pruning-quality diagnostic (reusing find_gold_rank + MetricAccum
as-is, since that abstraction still applies to "the ranked list of beam survivors"), while the
FINAL beam's hop/chain match rate is reported separately with its own plain counters — these
are comparable in spirit to production's recall@1/chain_recall@1 but not the identical metric,
since there's no single per-hop "committed choice with a rank" outside the survivor set.

Usage (needs a real GPU + Llama checkpoint + vLLM):
  python wavefront_hidden_probe_beam.py --probe-dir probe_artifacts --limit 20 --beam-width 3
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
PROJECT_ROOT = HERE.parent
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"
TRACES_DIR = PROJECT_ROOT / "traces"
sys.path.insert(0, str(RETRIEVAL_DIR))
sys.path.insert(0, str(TRACES_DIR))

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


def load_probe(probe_dir: Path):
    import joblib
    pca = joblib.load(probe_dir / "pca.joblib")
    lr = joblib.load(probe_dir / "lr.joblib")
    meta = json.loads((probe_dir / "meta.json").read_text(encoding="utf-8"))
    if not meta.get("pool_positions"):
        print(
            "WARNING: this probe was NOT fit with --pool-positions (trained on final h_K "
            "only) — its scores at intermediate hops here are a zero-shot generalization, "
            "not what it was trained for. Re-fit with --pool-positions for this use case.",
            file=sys.stderr,
        )
    return pca, lr, meta


def score_prefixes(
    texts: list[str], *, model, tokenizer, layer: int, batch_size: int, pca, lr,
    hidden_cache: dict[tuple[int, str], Any], desc: str,
) -> list[float]:
    """Absolute h_j score (probability this prefix has been corrupted so far so, NOT a
    delta and NOT a running sum — batch_last_hidden already gives the full-prefix hidden
    state, and error_propagation_probe's own validation showed h_j on its own already
    encodes cumulative history up to that point)."""
    if not texts:
        return []
    hiddens_by_layer = batch_last_hidden(
        texts=texts, model=model, tokenizer=tokenizer, layers=[layer],
        batch_size=batch_size, cache=hidden_cache, desc=desc,
    )
    X = np.stack(hiddens_by_layer[layer]).astype(np.float64)
    Z = pca.transform(X)
    return lr.predict_proba(Z)[:, 1].tolist()


@dataclass
class BeamPath:
    hop_steps: list[tuple[str, str]] = field(default_factory=list)  # (expanded_q, evidence_text)
    prior: list[str] = field(default_factory=list)  # per-hop short answers, feed [Answer N]
    para_ids: list[int] = field(default_factory=list)
    rank_key: float = 0.0  # HIGHER = better (matches production's rerank_by_final_score convention)
    probe_score: float = 0.0  # raw probe score (unmixed), kept only for diagnostics/output


@dataclass
class BeamExampleState:
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
    ap.add_argument("--dataset", choices=["musique"], default="musique")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--musique-dir", type=Path, default=MUSIQUE_DIR)
    ap.add_argument("--decompose-mode", choices=["gt", "bart_decompose"], default="bart_decompose")
    ap.add_argument("--decompose-file", type=Path, default=None)
    ap.add_argument("--probe-dir", type=Path, required=True,
                     help="Directory with pca.joblib/lr.joblib/meta.json from "
                          "score_and_analyze.py --pool-positions --out-dir <this dir>.")
    ap.add_argument("--beam-width", type=int, default=3)
    ap.add_argument("--retrieve-k", type=int, default=10)
    ap.add_argument("--cos-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--rerank-mode", choices=["probe_only", "weighted"], default="probe_only",
                     help="probe_only (current/original behavior): rank candidates purely by "
                          "probe score, ignoring cosine similarity after the initial top-k "
                          "retrieval. weighted: rank by emb_score - lambda_probe * probe_score, "
                          "mirroring production's rerank_by_final_score(emb_score, abnormal_score, "
                          "lam) in retrieval/run_retrieval_exp.py — production never uses its own "
                          "gate score as a standalone ranker, always mixed with emb_score like this.")
    ap.add_argument("--lambda-probe", type=float, default=0.25,
                     help="Only used when --rerank-mode weighted (same default as production's "
                          "--lambda-lr in run_retrieval_exp_wavefront.py).")
    ap.add_argument("--answerable-only", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--results-root", type=Path, default=HERE / "results",
                     help="Parent of the timestamped run folder (matches "
                          "retrieval/run_retrieval_exp_wavefront.py's convention).")
    ap.add_argument("--out-dir", type=Path, default=None,
                     help="Exact run folder to use instead of an auto timestamped one.")
    ap.add_argument("--run-tag", default=None,
                     help="Folder name suffix and cases/metrics filename suffix; default "
                          "'hidden_probe_beam_{dataset}_bw{beam_width}'.")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--gate-device", default="cuda:0")
    ap.add_argument("--layer", type=int, default=31,
                     help="Must match the layer error_propagation_probe/extract_full_hidden_states.py "
                          "used to build the probe's training data (default 31 matches its default).")
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

    tag = args.run_tag or f"hidden_probe_beam_{args.dataset}_bw{args.beam_width}"
    args.out_dir = resolve_output_dir(
        out_dir=args.out_dir, results_root=args.results_root, run_tag=tag,
        decompose_mode=args.decompose_mode, methods=["hidden_probe_beam"],
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{sanitize_run_tag(args.run_tag)}" if args.run_tag else ""
    out_cases = args.out_dir / f"retrieval_cases_{args.split}{tag_suffix}.jsonl"
    out_metrics = args.out_dir / f"retrieval_exp_{args.split}{tag_suffix}.json"
    print(f"Output directory: {args.out_dir}")

    (expand_hop_template, _a, _b, _c, _d, _e, judge_answer_official) = _get_pipeline_helpers()

    pca, lr, probe_meta = load_probe(args.probe_dir)
    print(f"Probe: {args.probe_dir} (pool_positions={probe_meta.get('pool_positions')}, "
          f"pca_dim={probe_meta.get('pca_dim')})")
    print(f"Examples: {len(ids)}  beam_width={args.beam_width}  decompose={args.decompose_file}")

    examples: list[BeamExampleState] = []
    for eid in ids:
        row = records[eid]
        sub_qs = prepare_sub_questions(decompose_idx.get(eid) or [], decompose_mode=args.decompose_mode)
        if not sub_qs:
            continue
        _gold_texts, gold_idxs = gold_evidence_musique(row)
        examples.append(BeamExampleState(
            eid=eid, row=row, q_main=(row.get("question") or "").strip(),
            paragraphs=row.get("paragraphs") or [], sub_questions=sub_qs,
            gold_idxs=gold_idxs, K=len(sub_qs), beams=[BeamPath()],
        ))

    print("Loading Llama model for hidden-state scoring ...", flush=True)
    gate_model, gate_tokenizer = load_gate_model(args)
    print("Loading vLLM generator ...", flush=True)
    generator = BatchVllmGenerator(
        args.model, dtype=args.dtype, tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
    )

    hidden_cache: dict[tuple[int, str], Any] = {}
    max_hops = max((ex.K for ex in examples), default=0)

    # Oracle diagnostic: gold's rank AMONG SURVIVORS at each hop (pruning-quality check,
    # independent of whether the final single best beam ends up picking it).
    oracle_hop_survive: dict[Any, MetricAccum] = {"all": MetricAccum(), **{K: MetricAccum() for K in K_BUCKETS}}

    for hop_j in range(1, max_hops + 1):
        t0 = time.perf_counter()
        print(f"\n=== beam hop {hop_j}/{max_hops} ===", flush=True)

        expand_retrieve: list[tuple[BeamExampleState, int, str, list[tuple[dict, float]]]] = []
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
        print(f"[hop {hop_j}] {len(expand_retrieve)} (example, path) contexts to score", flush=True)

        # Build the FULL candidate prefix text for every (path, candidate) continuation and
        # score it directly (absolute h_j, not incremental) with the probe.
        texts: list[str] = []
        owners: list[tuple[int, dict, float]] = []  # (index into expand_retrieve, candidate para, emb_score)
        for er_idx, (ex, path_idx, expanded_q, candidates_all) in enumerate(expand_retrieve):
            parent = ex.beams[path_idx]
            for para, emb_score in candidates_all:
                ev_text = (para.get("paragraph_text") or "").strip()
                prefix = build_trace_prefix(ex.q_main, parent.hop_steps + [(expanded_q, ev_text)])
                texts.append(prefix)
                owners.append((er_idx, para, emb_score))

        probe_scores = score_prefixes(
            texts, model=gate_model, tokenizer=gate_tokenizer, layer=args.layer,
            batch_size=args.hidden_batch_size, pca=pca, lr=lr, hidden_cache=hidden_cache,
            desc=f"hop{hop_j} probe-score",
        )

        # rank_key: HIGHER = better, matching production's rerank_by_final_score convention.
        # probe_only: raw probe score is "P(corrupted)", so lower is better -> negate.
        # weighted: emb_score - lambda * probe_score, same formula run_retrieval_exp.py uses
        # for its own gate score — production never ranks by gate score alone (see docstring).
        continuations_by_ex: dict[str, list[tuple[float, float, int, str, dict]]] = defaultdict(list)
        for (er_idx, para, emb_score), probe_score in zip(owners, probe_scores):
            ex, path_idx, expanded_q, _candidates_all = expand_retrieve[er_idx]
            if args.rerank_mode == "weighted":
                rank_key = emb_score - args.lambda_probe * probe_score
            else:
                rank_key = -probe_score
            continuations_by_ex[ex.eid].append((rank_key, probe_score, path_idx, expanded_q, para))

        kept_by_ex: dict[str, list[tuple[float, float, int, str, dict]]] = {}
        for ex in examples:
            conts = continuations_by_ex.get(ex.eid)
            if not conts:
                continue
            conts.sort(key=lambda t: t[0], reverse=True)  # higher rank_key = better
            kept_by_ex[ex.eid] = conts[: args.beam_width]

            gold_idx = ex.gold_idxs[hop_j - 1] if hop_j - 1 < len(ex.gold_idxs) else None
            if gold_idx is not None:
                survivor_ranked = [(para, rk) for rk, _ps, _p, _q, para in kept_by_ex[ex.eid]]
                rank = find_gold_rank(survivor_ranked, gold_idx)
                oracle_hop_survive["all"].update(rank)
                if ex.K in oracle_hop_survive:
                    oracle_hop_survive[ex.K].update(rank)

        answer_prompts: list[str] = []
        answer_meta: list[tuple[BeamExampleState, float, float, int, str, dict]] = []
        for ex in examples:
            for rank_key, probe_score, path_idx, expanded_q, para in kept_by_ex.get(ex.eid, []):
                parent = ex.beams[path_idx]
                answer_prompts.append(
                    prompt_short_answer_with_context(ex.q_main, parent.hop_steps, parent.prior, expanded_q, para)
                )
                answer_meta.append((ex, rank_key, probe_score, path_idx, expanded_q, para))
        answer_raw = (
            generator.generate_batch(answer_prompts, args.max_new_tokens_answer, desc=f"beam-answer-hop{hop_j}")
            if answer_prompts else []
        )

        rebuilt: dict[str, list[BeamPath]] = defaultdict(list)
        for (ex, rank_key, probe_score, path_idx, expanded_q, para), raw in zip(answer_meta, answer_raw):
            parent = ex.beams[path_idx]
            sub_ans = normalize_short_answer(raw)
            ev_text = (para.get("paragraph_text") or "").strip()
            rebuilt[ex.eid].append(BeamPath(
                hop_steps=parent.hop_steps + [(expanded_q, ev_text)],
                prior=parent.prior + [sub_ans],
                para_ids=parent.para_ids + [int(para.get("idx", -1))],
                rank_key=rank_key,
                probe_score=probe_score,
            ))

        for ex in examples:
            if hop_j <= ex.K and ex.eid in rebuilt:
                ex.beams = rebuilt[ex.eid]
            # examples already finished (hop_j > ex.K) keep their final beams untouched.

        elapsed = time.perf_counter() - t0
        print(f"[hop {hop_j}] {len(texts)} candidates scored, elapsed {elapsed / 60:.1f} min", flush=True)

    # Final answer: single highest-rank_key beam per example -> one final reader call
    # (not a multi-chain judging prompt, to avoid blowing up context length).
    final_tasks: list[tuple[BeamExampleState, BeamPath]] = []
    for ex in examples:
        if not ex.beams:
            continue
        best = max(ex.beams, key=lambda p: p.rank_key)
        final_tasks.append((ex, best))
    final_prompts = [
        build_final_reader_cot_prompt(ex.q_main, best.hop_steps, best.prior) for ex, best in final_tasks
    ]
    final_raw = generator.generate_chat_batch(
        FINAL_READER_SYSTEM_PROMPT, final_prompts, args.max_new_tokens_final, desc="beam-final-reader",
    )

    answer_accum: dict[str, AnswerAccum] = {"all": AnswerAccum(), **{K: AnswerAccum() for K in K_BUCKETS}}
    hop_match_counts: dict[Any, list[int]] = {"all": [0, 0], **{K: [0, 0] for K in K_BUCKETS}}  # [match, total]
    chain_match_counts: dict[Any, list[int]] = {"all": [0, 0], **{K: [0, 0] for K in K_BUCKETS}}  # [match, n]

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
                "beam_final_probe_score": best.probe_score,
                "para_ids": best.para_ids, "gold_idxs": ex.gold_idxs, "prior": best.prior,
            }, ensure_ascii=False) + "\n")

    def rate(counts: list[int]) -> float | None:
        return round(counts[0] / counts[1], 4) if counts[1] else None

    run_wall_sec = time.perf_counter() - run_t0
    results: dict[str, Any] = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "output_dir": str(args.out_dir),
        "n_examples": len(final_tasks),
        "timing": {
            "wall_clock_sec": round(run_wall_sec, 2),
            "wall_clock_hr": round(run_wall_sec / 3600, 3),
        },
        "answer_overall": answer_accum["all"].result(),
        "answer_by_K": {K: answer_accum[K].result() for K in K_BUCKETS},
        "final_beam_hop_match_rate_overall": rate(hop_match_counts["all"]),
        "final_beam_hop_match_rate_by_K": {K: rate(hop_match_counts[K]) for K in K_BUCKETS},
        "final_beam_chain_match_rate_overall": rate(chain_match_counts["all"]),
        "final_beam_chain_match_rate_by_K": {K: rate(chain_match_counts[K]) for K in K_BUCKETS},
        "oracle_gold_rank_among_survivors_overall": oracle_hop_survive["all"].result(),
        "oracle_gold_rank_among_survivors_by_K": {K: oracle_hop_survive[K].result() for K in K_BUCKETS},
        "_metrics_note": (
            "final_beam_hop/chain_match_rate are NOT the same metric as production's "
            "recall@1/chain_recall@1 — beam search keeps beam_width survivors per hop with no "
            "single committed choice until the final beam is picked, see script docstring. "
            "oracle_gold_rank_among_survivors reuses find_gold_rank+MetricAccum on the ranked "
            "list of beam survivors at each hop (pruning-quality diagnostic, independent of "
            "whether the final best beam ends up picking gold)."
        ),
    }
    out_metrics.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nMetrics saved to  {out_metrics}")
    print(f"Cases saved to    {out_cases}")
    print(f"Wall clock hr     {results['timing']['wall_clock_hr']:.3f}")
    print(f"answer_overall    {results['answer_overall']}")
    print(f"hop_match_rate    {results['final_beam_hop_match_rate_overall']}")
    print(f"chain_match_rate  {results['final_beam_chain_match_rate_overall']}")
    print(f"oracle_survival   {results['oracle_gold_rank_among_survivors_overall']}")


if __name__ == "__main__":
    main()
