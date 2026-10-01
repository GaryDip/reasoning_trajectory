#!/usr/bin/env python3
"""
Scheme B (listwise contrastive ranker) — step 1: build REALISTIC top-k
candidate pools per hop, with the gold candidate marked if present.

This is the data-engineering gap Scheme A's reuse of existing hidden_states/
data doesn't have: current train/dev .npz files only ever contain "one gold
+ a handful of manually-sampled distractors" per hop, not the actual top-k a
real BGE retrieval would return. Scheme B needs the real thing, because the
whole point is training the ranker to win against the SAME competitors it
will actually face at inference time — not an arbitrarily easier or harder
sample.

BGE-only (no Llama) — cheap, runs on CPU. Uses GOLD prior-hop answers to
fill [Answer N] placeholders (same oracle-prior isolation principle as
reader_ceiling_probe / rerank_pairwise_mlp's eval), so pool quality reflects
this hop's retrieval only, not compounding errors from earlier hops.

Output: one jsonl, one row per (example, hop) with its top-k candidates
(idx, title, paragraph_text, emb_score, is_gold) plus the prefix/expanded
query text needed later to extract Delta for each candidate.

Usage:
  python build_topk_pools.py --split train --retrieve-k 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"
TRACES_DIR = PROJECT_ROOT / "traces"
sys.path.insert(0, str(RETRIEVAL_DIR))
sys.path.insert(0, str(TRACES_DIR))

from run_retrieval_exp import (  # noqa: E402
    MUSIQUE_DIR,
    _get_pipeline_helpers,
    build_trace_prefix,
    embed_retrieval,
    load_dataset_records,
    load_decompose_index,
    prepare_sub_questions,
    resolve_decompose_file,
    select_example_ids,
)
from run_retrieval_exp_wavefront import init_example_state  # noqa: E402
from trace_evidence import gold_hop_answers_musique  # noqa: E402

METHOD = "pool_builder"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["musique"], default="musique")
    ap.add_argument("--split", choices=["train", "dev"], default="train")
    ap.add_argument("--musique-dir", type=Path, default=MUSIQUE_DIR)
    ap.add_argument("--decompose-mode", choices=["gt"], default="gt",
                     help="gt only: pools are keyed to gold hop order, same reason "
                          "reader_ceiling_probe requires gt for its direct-lookup path.")
    ap.add_argument("--decompose-file", type=Path, default=None)
    ap.add_argument("--retrieve-k", type=int, default=20)
    ap.add_argument("--cos-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--answerable-only", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.decompose_file = resolve_decompose_file(
        args.decompose_mode, args.split, args.decompose_file, args.dataset
    )
    out_path = args.out or (HERE / "data" / f"{args.dataset}_{args.split}_pools.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds_args = argparse.Namespace(dataset=args.dataset, musique_dir=args.musique_dir, split=args.split)
    records = load_dataset_records(ds_args)
    decompose_idx = load_decompose_index(args.decompose_file)
    ids = sorted(set(records) & set(decompose_idx))
    if args.answerable_only:
        ids = [eid for eid in ids if records[eid].get("answerable", True)]
    ids = select_example_ids(ids, decompose_idx, limit=args.limit, limit_per_k=0, seed=args.sample_seed)

    (expand_hop_template, *_rest) = _get_pipeline_helpers()

    print(f"Examples: {len(ids)}  decompose={args.decompose_file}")

    examples = []
    gold_answers_by_id: dict[str, list[str]] = {}
    for eid in ids:
        row = records[eid]
        sub_qs = prepare_sub_questions(decompose_idx.get(eid) or [], decompose_mode=args.decompose_mode)
        if not sub_qs:
            continue
        examples.append(init_example_state(eid, row, sub_qs, [METHOD], []))
        gold_answers_by_id[eid] = gold_hop_answers_musique(row)

    max_hops = max((ex.K for ex in examples), default=0)
    n_pools = 0
    n_gold_in_pool = 0

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    with out_path.open("w", encoding="utf-8") as f:
        for hop_j in range(1, max_hops + 1):
            print(f"=== hop {hop_j}/{max_hops} ===", flush=True)
            ex_iter = examples
            if tqdm is not None:
                ex_iter = tqdm(examples, desc=f"hop{hop_j} pools", unit="ex",
                                file=sys.stderr, dynamic_ncols=True)
            for ex in ex_iter:
                if hop_j > ex.K:
                    continue
                hop_row = ex.hop_results[hop_j - 1]
                gold_pi = hop_row.get("gold_para_idx")
                if not (hop_row.get("has_gold_hop") and gold_pi is not None):
                    continue

                gold_answers = gold_answers_by_id[ex.eid]
                prior = gold_answers[: hop_j - 1]
                raw_sq = ex.sub_questions[hop_j - 1]
                expanded_q = expand_hop_template(raw_sq, prior)

                hop_steps = []
                ok = True
                for gj in range(hop_j - 1):
                    prior_pi = ex.hop_results[gj].get("gold_para_idx")
                    prior_para = next((p for p in ex.paragraphs if int(p.get("idx", -1)) == prior_pi), None)
                    if prior_para is None:
                        ok = False
                        break
                    hop_steps.append((ex.sub_questions[gj], (prior_para.get("paragraph_text") or "").strip()))
                if not ok:
                    continue
                prefix_before = build_trace_prefix(ex.q_main, hop_steps)

                candidates_all = embed_retrieval(expanded_q, ex.paragraphs, args.retrieve_k, args.cos_model)
                if not candidates_all:
                    continue

                gold_pos = None
                cand_rows = []
                for pos, (p, score) in enumerate(candidates_all):
                    is_gold = int(p.get("idx", -1)) == int(gold_pi)
                    if is_gold:
                        gold_pos = pos
                    cand_rows.append({
                        "idx": int(p.get("idx", -1)),
                        "title": p.get("title") or "",
                        "paragraph_text": p.get("paragraph_text") or "",
                        "emb_score": round(float(score), 6),
                        "is_gold": is_gold,
                    })

                n_pools += 1
                if gold_pos is not None:
                    n_gold_in_pool += 1

                f.write(json.dumps({
                    "pool_id": f"{ex.eid}__K{ex.K}__j{hop_j - 1}",
                    "source_id": ex.eid, "K": ex.K, "j": hop_j - 1,
                    "prefix_before": prefix_before, "expanded_q": expanded_q,
                    "gold_pos": gold_pos, "candidates": cand_rows,
                }, ensure_ascii=False) + "\n")

    print(f"\nWrote {n_pools} pools -> {out_path}")
    print(f"Gold-in-top-{args.retrieve_k} coverage: {n_gold_in_pool}/{n_pools} "
          f"({n_gold_in_pool / n_pools:.4f})" if n_pools else "no pools")


if __name__ == "__main__":
    main()
