#!/usr/bin/env python3
"""
Step 0+1 test (see update_doc/0907/0907update.md section 9): load a handful of real MuSiQue
train examples (GT decompose) and run hop-1's REAL retrieval (embed_retrieval, no gate, no
short-answer generation yet) -- confirms the basic building blocks (loading + retrieval) work
and produce sane output before anything else gets layered on top.

Usage:
  python step01_load_and_retrieve_test.py --n 8
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
TRACES_DIR = PROJECT_ROOT / "traces"
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"
sys.path.insert(0, str(TRACES_DIR))
sys.path.insert(0, str(RETRIEVAL_DIR))

from dataset_loaders import get_decompose_path, load_decompose_bundle, load_raw_index  # noqa: E402
from trace_evidence import build_hop_queries, gold_evidence_musique, gold_hop_answers_musique  # noqa: E402
from run_retrieval_exp import embed_retrieval  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=8, help="how many example ids to spot-check")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retrieve-k", type=int, default=10)
    ap.add_argument("--cos-model", default="BAAI/bge-base-en-v1.5")
    args = ap.parse_args()

    decompose_path = get_decompose_path("musique", args.split, mode="gt")
    decomp, raw_id_map = load_decompose_bundle(decompose_path, include_enhance=False)
    raw_index = load_raw_index("musique", args.split)
    print(f"decompose entries: {len(decomp)}  raw records: {len(raw_index)}")

    ids = sorted(decomp.keys())
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    picked = ids[: args.n]

    n_hop1_hit = 0
    n_checked = 0
    for did in picked:
        raw_id = raw_id_map.get(did, did)
        record = raw_index.get(raw_id)
        if record is None:
            print(f"[skip] {did}: no raw record for {raw_id}")
            continue
        sub_qs = decomp[did]
        gold_texts, gold_idxs = gold_evidence_musique(record)
        hop_answers = gold_hop_answers_musique(record)
        k = min(len(sub_qs), len(gold_idxs), len(hop_answers))
        if k < 1:
            print(f"[skip] {did}: k<1")
            continue
        expanded_qs = build_hop_queries(sub_qs[:k], hop_answers[:k])
        paragraphs = record.get("paragraphs") or []

        print(f"\n=== {did}  (K={k}) ===")
        print(f"question: {record.get('question')}")
        print(f"hop1 raw sub_q:      {sub_qs[0]!r}")
        print(f"hop1 expanded query: {expanded_qs[0]!r}")
        print(f"gold_idx[hop1]: {gold_idxs[0]}  gold_hop_answer[hop1]: {hop_answers[0]!r}")

        ranked = embed_retrieval(expanded_qs[0], paragraphs, args.retrieve_k, args.cos_model)
        top3 = ranked[:3]
        for rank, (para, score) in enumerate(top3, start=1):
            hit = "  <-- GOLD" if int(para.get("idx", -1)) == gold_idxs[0] else ""
            title = (para.get("title") or "")[:60]
            print(f"  top{rank}: idx={para.get('idx')} score={score:.4f} title={title!r}{hit}")

        n_checked += 1
        if ranked and int(ranked[0][0].get("idx", -1)) == gold_idxs[0]:
            n_hop1_hit += 1

    print(f"\n=== summary: hop1 cosine-top1 hit gold on {n_hop1_hit}/{n_checked} examples ===")


if __name__ == "__main__":
    main()
