#!/usr/bin/env python3
"""
Wavefront-style (batched across ALL examples per hop) end-to-end retrieval + answer
generation combining TWO already-trained, already-separately-validated signals for the
same candidate:
  - gate v2 (gate/artifacts_pooled_v2): Delta_j = h_j - h_{j-1}, 4 separate pooled-j
    models, LOCAL "did this step just get corrupted relative to the step before" signal.
    Already validated standalone via lr_rerank: recall@1=0.6627, chain_recall@1=0.4357,
    EM=0.4100, F1=0.5045 (all four beat baseline simultaneously — see conversation).
  - the error_propagation_probe (error_propagation_probe/probe_artifacts): absolute h_j,
    CUMULATIVE "how much has gone wrong so far" signal. Already validated standalone via
    a fixed-lambda weighted formula: oracle recall@1=0.6735, chain_match=0.4659, but
    EM=0.393/F1=0.4816 (retrieval-side beats baseline, answer-side doesn't).
  - A direct case-level overlap check (see conversation) found ~13-14% of dev examples
    where only ONE of the two methods gets the example right (chain match or answer EM),
    not the other — real, non-redundant complementary coverage, which is why combining
    them is worth trying instead of picking whichever one is currently ahead.

--vote-mode controls how the two per-candidate scores get combined into one rank_key
(gate v2 and h_j-probe scores are each frozen — never retrained here):
  sum (default): final_score = emb_score - lambda_gate_v2 * gate_v2_score
                                          - lambda_probe   * probe_score
                 lambda_gate_v2/lambda_probe default to 0.25 each — reusing the SAME
                 value each signal already uses successfully on its own (see above),
                 not a freshly tuned number.
  veto:          exclude any candidate where EITHER score exceeds --veto-threshold
                 (default 0.5 — each score is a raw LogisticRegression probability of
                 "this step is corrupted", so 0.5 is each model's own natural decision
                 boundary, not an invented cutoff); rank survivors by emb_score alone.
                 If every candidate gets vetoed (both models flag the whole pool), falls
                 back to ranking the ORIGINAL (unfiltered) pool by emb_score, so a hop
                 never ends up with zero candidates to choose from.
  rrf:           don't blend scores at all — let gate v2 and the probe EACH produce a
                 complete, independent rerank first (their own emb_score - lambda*score
                 formula), then fuse the resulting RANK POSITIONS (not raw scores) via
                 Reciprocal Rank Fusion: 1/(k+rank_gate_v2) + 1/(k+rank_probe). No score
                 calibration needed (only rank order is compared, not magnitude), and
                 --rrf-k (default 60, the standard IR-literature value) is known to be a
                 very insensitive knob, not something that needs tuning.

Both signals are scored from the SAME single batched Llama forward pass per hop (only
the union of layers each needs — {15, 23} from gate v2's own artifacts + 31 from the
probe — costs no extra time per run_retrieval_exp_wavefront.py::batch_last_hidden's own
design: one forward pass already computes every layer, reading out more of them is free),
not two separate forward passes.

Single committed path per example by default (--beam-width 1), matching production's own
greedy structure; --beam-width > 1 turns this into a beam search using the same combined
scoring.

Needs (train first if missing):
  - gate/artifacts_pooled_v2/{j0..j3}.joblib          (production gate v2, already trained)
  - error_propagation_probe/probe_artifacts/{pca,lr}.joblib  (already trained)

Reuses (imports, does not copy) retrieval/run_retrieval_exp.py and
retrieval/run_retrieval_exp_wavefront.py's data loading, retrieval, batched hidden-state
extraction, accumulators, get_gate_artifact and vLLM generation machinery, plus
gate/lr_artifacts.py::load_artifacts — production files are untouched.

Usage:
  python wavefront_combined_gate.py --limit 20
  python wavefront_combined_gate.py --limit 20 --vote-mode veto
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
GATE_DIR = PROJECT_ROOT / "gate"
PROBE_DIR = PROJECT_ROOT / "error_propagation_probe"
sys.path.insert(0, str(RETRIEVAL_DIR))
sys.path.insert(0, str(TRACES_DIR))
sys.path.insert(0, str(GATE_DIR))

from run_retrieval_exp import (  # noqa: E402
    K_BUCKETS,
    MUSIQUE_DIR,
    AnswerAccum,
    MetricAccum,
    _get_pipeline_helpers,
    build_trace_prefix,
    embed_retrieval,
    find_gold_rank,
    get_gate_artifact,
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
from lr_artifacts import load_artifacts  # noqa: E402
from trace_evidence import gold_evidence_musique  # noqa: E402
from trace_format import escape_double_quotes  # noqa: E402


def load_probe(probe_dir: Path):
    import joblib
    pca = joblib.load(probe_dir / "pca.joblib")
    lr = joblib.load(probe_dir / "lr.joblib")
    return pca, lr


@dataclass
class BeamPath:
    hop_steps: list[tuple[str, str]] = field(default_factory=list)
    prior: list[str] = field(default_factory=list)
    para_ids: list[int] = field(default_factory=list)
    rank_key: float = 0.0
    gate_v2_score: float = 0.0
    probe_score: float = 0.0


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
    ap.add_argument("--dataset", choices=["musique"], default="musique")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--musique-dir", type=Path, default=MUSIQUE_DIR)
    ap.add_argument("--decompose-mode", choices=["gt", "bart_decompose"], default="bart_decompose")
    ap.add_argument("--decompose-file", type=Path, default=None)
    ap.add_argument("--gate-v2-artifacts-dir", type=Path, default=GATE_DIR / "artifacts_pooled_v2")
    ap.add_argument("--gate-v2-artifact-mode", choices=["pooled"], default="pooled",
                     help="pooled only: this script looks up ONE gate-v2 artifact per hop "
                          "position (shared across every example/path in that hop, since all "
                          "candidates in a hop are scored in a single batched forward pass). "
                          "nonpooled artifacts are keyed by (K, j) — a per-example K, not just "
                          "hop position — which doesn't fit that shared-per-hop lookup, so it's "
                          "not offered here.")
    ap.add_argument("--probe-dir", type=Path, default=PROBE_DIR / "probe_artifacts")
    ap.add_argument("--probe-layer", type=int, default=31,
                     help="error_propagation_probe/probe_artifacts has no layer field in its "
                          "own meta.json (unlike gate v2's per-j artifacts) — 31 matches what "
                          "extract_full_hidden_states.py used by default when it was built.")
    ap.add_argument("--vote-mode", choices=["sum", "veto", "rrf"], default="sum")
    ap.add_argument("--lambda-gate-v2", type=float, default=0.25,
                     help="Used in --vote-mode sum and rrf. Reuses gate v2's own already-"
                          "validated --lambda-lr default, not a freshly tuned value.")
    ap.add_argument("--lambda-probe", type=float, default=0.25,
                     help="Used in --vote-mode sum and rrf. Reuses the h_j-probe's own already-"
                          "validated weighted-mode default, not a freshly tuned value.")
    ap.add_argument("--veto-threshold", type=float, default=0.5,
                     help="Only used in --vote-mode veto. Each score is a raw LR probability "
                          "of corruption, so 0.5 is each model's own natural decision boundary.")
    ap.add_argument("--rrf-k", type=float, default=60.0,
                     help="Only used in --vote-mode rrf. Reciprocal Rank Fusion smoothing "
                          "constant (1/(k+rank)) — 60 is the standard IR-literature default, "
                          "RRF is known to be insensitive to this value.")
    ap.add_argument("--beam-width", type=int, default=1)
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
    ap.add_argument("--hidden-batch-size", type=int, default=8)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-new-tokens-answer", type=int, default=64)
    ap.add_argument("--max-new-tokens-final", type=int, default=128)
    return ap.parse_args()


def to_ranks(final_scores: np.ndarray) -> np.ndarray:
    """1 = best. final_scores: higher = better."""
    order = np.argsort(-final_scores, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(final_scores) + 1)
    return ranks


def combine_scores(
    emb_scores: np.ndarray, gate_v2_scores: np.ndarray, probe_scores: np.ndarray, *, args: argparse.Namespace,
) -> np.ndarray:
    """rank_key, higher = better, for every candidate in ONE hop's pool."""
    if args.vote_mode == "sum":
        return emb_scores - args.lambda_gate_v2 * gate_v2_scores - args.lambda_probe * probe_scores
    if args.vote_mode == "veto":
        # drop anything either model flags as corrupted (p > threshold); rank the rest by
        # emb_score alone; if that empties the pool, fall back to ranking everyone by emb_score.
        survives = (gate_v2_scores <= args.veto_threshold) & (probe_scores <= args.veto_threshold)
        if not survives.any():
            return emb_scores
        return np.where(survives, emb_scores, -np.inf)
    # rrf: let gate v2 and the probe EACH do a complete, independent rerank first (their own
    # already-validated emb_score - lambda*score formula), rank-order each of those two full
    # rerankings, then fuse the RANKS (not the raw scores) via Reciprocal Rank Fusion — no
    # score calibration needed since only rank position is compared, not magnitude.
    gate_v2_final = emb_scores - args.lambda_gate_v2 * gate_v2_scores
    probe_final = emb_scores - args.lambda_probe * probe_scores
    gate_v2_ranks = to_ranks(gate_v2_final)
    probe_ranks = to_ranks(probe_final)
    return 1.0 / (args.rrf_k + gate_v2_ranks) + 1.0 / (args.rrf_k + probe_ranks)


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

    tag = args.run_tag or f"combined_gate_{args.vote_mode}_{args.dataset}_bw{args.beam_width}"
    args.out_dir = resolve_output_dir(
        out_dir=args.out_dir, results_root=args.results_root, run_tag=tag,
        decompose_mode=args.decompose_mode, methods=["combined_gate"],
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{sanitize_run_tag(args.run_tag)}" if args.run_tag else ""
    out_cases = args.out_dir / f"retrieval_cases_{args.split}{tag_suffix}.jsonl"
    out_metrics = args.out_dir / f"retrieval_exp_{args.split}{tag_suffix}.json"
    print(f"Output directory: {args.out_dir}")
    print(f"vote_mode={args.vote_mode}", flush=True)

    (expand_hop_template, *_rest, judge_answer_official) = _get_pipeline_helpers()

    gate_v2_artifacts = load_artifacts(args.gate_v2_artifacts_dir, args.gate_v2_artifact_mode)
    probe_pca, probe_lr = load_probe(args.probe_dir)
    print(f"Gate v2: {args.gate_v2_artifacts_dir} ({args.gate_v2_artifact_mode})")
    print(f"Probe:   {args.probe_dir} (layer={args.probe_layer})")

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
    print("Loading vLLM generator ...", flush=True)
    generator = BatchVllmGenerator(
        args.model, dtype=args.dtype, tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
    )

    hidden_cache: dict[tuple[int, str], Any] = {}
    max_hops = max((ex.K for ex in examples), default=0)
    oracle_hop_survive: dict[Any, MetricAccum] = {"all": MetricAccum(), **{K: MetricAccum() for K in K_BUCKETS}}

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

        # gate v2's own artifact tells us which layer THIS pooled transition (j = hop_j-1)
        # needs; the probe always reads --probe-layer. Union of both -> one forward pass.
        art = get_gate_artifact(gate_v2_artifacts, args.gate_v2_artifact_mode, 0, hop_j - 1)
        gate_v2_layer = int(art["layer"]) if art is not None else args.probe_layer
        needed_layers = sorted({gate_v2_layer, args.probe_layer})

        texts: list[str] = []
        owners: list[tuple[int, int]] = []  # (context index, slot); slot=-1 -> prefix_before
        for ctx_idx, (ex, path_idx, expanded_q, candidates_all) in enumerate(expand_retrieve):
            parent = ex.beams[path_idx]
            prefix_before = build_trace_prefix(ex.q_main, parent.hop_steps)
            texts.append(prefix_before)
            owners.append((ctx_idx, -1))
            for slot, (para, _emb_score) in enumerate(candidates_all):
                ev_text = (para.get("paragraph_text") or "").strip()
                text = f'{prefix_before} Step {hop_j}: {expanded_q} Evidence: "{escape_double_quotes(ev_text)}"'
                texts.append(text)
                owners.append((ctx_idx, slot))

        hiddens_by_layer = batch_last_hidden(
            texts=texts, model=gate_model, tokenizer=gate_tokenizer, layers=needed_layers,
            batch_size=args.hidden_batch_size, cache=hidden_cache, desc=f"hop{hop_j} hidden",
        )

        prefix_hidden: dict[int, dict[int, np.ndarray]] = defaultdict(dict)  # ctx_idx -> layer -> h
        cand_hidden: dict[int, dict[int, dict[int, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
        for idx, (ctx_idx, slot) in enumerate(owners):
            for layer in needed_layers:
                h = hiddens_by_layer[layer][idx]
                if slot == -1:
                    prefix_hidden[ctx_idx][layer] = h
                else:
                    cand_hidden[ctx_idx][slot][layer] = h

        continuations_by_ex: dict[str, list[tuple[float, float, float, int, str, dict]]] = defaultdict(list)
        for ctx_idx, (ex, path_idx, expanded_q, candidates_all) in enumerate(expand_retrieve):
            n_cand = len(candidates_all)
            emb_arr = np.asarray([s for _p, s in candidates_all], dtype=np.float64)

            if art is not None:
                pca_g, lr_g = art["pca"], art["lr"]
                h_prev_g = prefix_hidden[ctx_idx][gate_v2_layer]
                deltas = np.stack([
                    cand_hidden[ctx_idx][slot][gate_v2_layer] - h_prev_g for slot in range(n_cand)
                ]).astype(np.float64)
                gate_v2_arr = lr_g.predict_proba(pca_g.transform(deltas))[:, 1]
            else:
                gate_v2_arr = np.zeros(n_cand, dtype=np.float64)

            cand_h_probe = np.stack([
                cand_hidden[ctx_idx][slot][args.probe_layer] for slot in range(n_cand)
            ]).astype(np.float64)
            probe_arr = probe_lr.predict_proba(probe_pca.transform(cand_h_probe))[:, 1]

            rank_keys = combine_scores(emb_arr, gate_v2_arr, probe_arr, args=args)

            for slot, (para, _emb_score) in enumerate(candidates_all):
                continuations_by_ex[ex.eid].append((
                    float(rank_keys[slot]), float(gate_v2_arr[slot]), float(probe_arr[slot]),
                    path_idx, expanded_q, para,
                ))

        kept_by_ex: dict[str, list[tuple[float, float, float, int, str, dict]]] = {}
        for ex in examples:
            conts = continuations_by_ex.get(ex.eid)
            if not conts:
                continue
            conts.sort(key=lambda t: t[0], reverse=True)
            kept_by_ex[ex.eid] = conts[: args.beam_width]

            gold_idx = ex.gold_idxs[hop_j - 1] if hop_j - 1 < len(ex.gold_idxs) else None
            if gold_idx is not None:
                survivor_ranked = [(para, rk) for rk, _g, _p, _pi, _q, para in kept_by_ex[ex.eid]]
                rank = find_gold_rank(survivor_ranked, gold_idx)
                oracle_hop_survive["all"].update(rank)
                if ex.K in oracle_hop_survive:
                    oracle_hop_survive[ex.K].update(rank)

        answer_prompts: list[str] = []
        answer_meta: list[tuple[ExampleState, float, float, float, int, str, dict]] = []
        for ex in examples:
            for rank_key, gate_v2_s, probe_s, path_idx, expanded_q, para in kept_by_ex.get(ex.eid, []):
                parent = ex.beams[path_idx]
                answer_prompts.append(
                    prompt_short_answer_with_context(ex.q_main, parent.hop_steps, parent.prior, expanded_q, para)
                )
                answer_meta.append((ex, rank_key, gate_v2_s, probe_s, path_idx, expanded_q, para))
        answer_raw = (
            generator.generate_batch(answer_prompts, args.max_new_tokens_answer, desc=f"answer-hop{hop_j}")
            if answer_prompts else []
        )

        rebuilt: dict[str, list[BeamPath]] = defaultdict(list)
        for (ex, rank_key, gate_v2_s, probe_s, path_idx, expanded_q, para), raw in zip(answer_meta, answer_raw):
            parent = ex.beams[path_idx]
            sub_ans = normalize_short_answer(raw)
            ev_text = (para.get("paragraph_text") or "").strip()
            rebuilt[ex.eid].append(BeamPath(
                hop_steps=parent.hop_steps + [(expanded_q, ev_text)],
                prior=parent.prior + [sub_ans],
                para_ids=parent.para_ids + [int(para.get("idx", -1))],
                rank_key=rank_key, gate_v2_score=gate_v2_s, probe_score=probe_s,
            ))

        for ex in examples:
            if hop_j <= ex.K and ex.eid in rebuilt:
                ex.beams = rebuilt[ex.eid]

        elapsed = time.perf_counter() - t0
        print(f"[hop {hop_j}] {len(texts)} texts scored, elapsed {elapsed / 60:.1f} min", flush=True)

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
                "beam_final_gate_v2_score": best.gate_v2_score, "beam_final_probe_score": best.probe_score,
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
