#!/usr/bin/env python3
"""
Step 3 test (see update_doc/0907/0907update.md section 9): run recipe2_core.run_recipe2_trace
on several real MuSiQue train examples, print the FULL resulting trace (all hops, labels,
pairs) for manual inspection -- confirms query expansion / propagation / post-hoc labeling /
pair extraction are all correct before adding hidden-state extraction or batching.

Usage:
  CUDA_VISIBLE_DEVICES=2 python step03_multihop_test.py --n 15 --min-k 3
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
TRACES_DIR = PROJECT_ROOT / "traces"
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TRACES_DIR))
sys.path.insert(0, str(RETRIEVAL_DIR))

from dataset_loaders import get_decompose_path, load_decompose_bundle, load_raw_index  # noqa: E402
from trace_evidence import gold_evidence_musique, gold_hop_answers_musique  # noqa: E402
from run_retrieval_exp_wavefront import BatchVllmGenerator  # noqa: E402
from recipe2_core import run_recipe2_trace  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=15, help="how many example ids to try")
    ap.add_argument("--min-k", type=int, default=1, help="only consider examples with K >= this")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--retrieve-k", type=int, default=10)
    ap.add_argument("--cos-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()

    decompose_path = get_decompose_path("musique", args.split, mode="gt")
    decomp, raw_id_map = load_decompose_bundle(decompose_path, include_enhance=False)
    raw_index = load_raw_index("musique", args.split)

    ids = sorted(decomp.keys())
    rng = random.Random(args.seed)
    rng.shuffle(ids)

    cases = []
    for did in ids:
        raw_id = raw_id_map.get(did, did)
        record = raw_index.get(raw_id)
        if record is None:
            continue
        sub_qs = decomp[did]
        gold_texts, gold_idxs = gold_evidence_musique(record)
        hop_answers = gold_hop_answers_musique(record)
        k = min(len(sub_qs), len(gold_idxs), len(hop_answers))
        if k < args.min_k:
            continue
        cases.append((did, record, sub_qs[:k], gold_idxs[:k], gold_texts[:k]))
        if len(cases) >= args.n:
            break

    print(f"Loading vLLM generator ({args.model}) ...")
    gen = BatchVllmGenerator(
        args.model, dtype="bfloat16", tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
    )

    n_all_correct = 0
    n_with_pairs = 0
    total_pairs = 0
    for did, record, sub_qs, gold_idxs, gold_texts in cases:
        trace = run_recipe2_trace(
            case_id=did, question=record.get("question", ""), sub_qs=sub_qs,
            gold_idxs=gold_idxs, gold_texts=gold_texts, paragraphs=record.get("paragraphs") or [],
            generator=gen, cos_model=args.cos_model, retrieve_k=args.retrieve_k,
        )
        print(f"\n=== {trace['trace_id']}  K={trace['K']}  all_correct={trace['all_correct']} "
              f"n_correct={trace['n_correct']}/{trace['K']} ===")
        for h in trace["hops"]:
            mark = "OK  " if h["is_correct"] else "WRONG"
            print(f"  hop{h['hop']} [{mark}] sub_q_expanded={h['sub_question_expanded']!r}")
            print(f"          committed_idx={h['committed_idx']} gold_idx={h['gold_idx']} "
                  f"answer={h['short_answer_generated']!r}")
            if not h["is_correct"]:
                print(f"          -> pair_id={h.get('pair_id')}")
        if trace["all_correct"]:
            n_all_correct += 1
        if trace["pairs"]:
            n_with_pairs += 1
            total_pairs += len(trace["pairs"])

    print(f"\n=== summary: {n_all_correct}/{len(cases)} traces fully correct "
          f"(these are recipe-2-dedup candidates, see step 8); "
          f"{n_with_pairs}/{len(cases)} traces produced >=1 pair, {total_pairs} pairs total ===")


if __name__ == "__main__":
    main()
