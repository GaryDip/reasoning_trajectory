#!/usr/bin/env python3
"""
Step 8 test: recipe 1 (pure gold trace, unified expand_hop_template expansion) + the
recipe1/recipe2 dedup rule. No LLM calls needed for recipe 1 itself (deterministic), but we
still need a real recipe 2 run first to decide whether to skip it.

Usage:
  CUDA_VISIBLE_DEVICES=0 python step08_recipe1_dedup_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACES_DIR = HERE.parent / "traces"
RETRIEVAL_DIR = HERE.parent / "retrieval"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TRACES_DIR))
sys.path.insert(0, str(RETRIEVAL_DIR))

from dataset_loaders import get_decompose_path, load_decompose_bundle, load_raw_index  # noqa: E402
from trace_evidence import gold_evidence_musique, gold_hop_answers_musique  # noqa: E402
from run_retrieval_exp_wavefront import BatchVllmGenerator  # noqa: E402
from recipe2_batch import run_recipe2_batch, build_recipe1_gold_trace, should_skip_recipe1  # noqa: E402

TARGET_IDS = ["3hop1__257997_104557_161232", "4hop1__322597_452185_53858_33265"]  # all_correct, not-all-correct


def main() -> None:
    decompose_path = get_decompose_path("musique", "train", mode="gt")
    decomp, raw_id_map = load_decompose_bundle(decompose_path, include_enhance=False)
    raw_index = load_raw_index("musique", "train")

    cases = []
    hop_answers_by_id = {}
    for did in TARGET_IDS:
        raw_id = raw_id_map.get(did, did)
        record = raw_index[raw_id]
        sub_qs = decomp[did]
        gold_texts, gold_idxs = gold_evidence_musique(record)
        hop_answers = gold_hop_answers_musique(record)
        k = min(len(sub_qs), len(gold_idxs), len(hop_answers))
        cases.append({
            "case_id": did, "question": record.get("question", ""), "sub_qs": sub_qs[:k],
            "gold_idxs": gold_idxs[:k], "gold_texts": gold_texts[:k],
            "paragraphs": record.get("paragraphs") or [],
        })
        hop_answers_by_id[did] = hop_answers[:k]

    print("Loading vLLM generator ...")
    gen = BatchVllmGenerator(
        "meta-llama/Llama-3.1-8B-Instruct", dtype="bfloat16", tensor_parallel_size=1,
        gpu_memory_utilization=0.2, max_model_len=8192,
    )

    r2_traces = run_recipe2_batch(cases, generator=gen)

    for case, r2 in zip(cases, r2_traces):
        did = case["case_id"]
        skip = should_skip_recipe1(r2)
        print(f"\n=== {did} ===")
        print(f"  recipe2 all_correct={r2['all_correct']}  -> should_skip_recipe1={skip}")
        if skip:
            print("  (recipe1 NOT generated, dedup rule applied)")
            continue
        r1 = build_recipe1_gold_trace(case, hop_answers_by_id[did])
        print(f"  recipe1 generated: trace_id={r1['trace_id']}  all_correct={r1['all_correct']}")
        for h in r1["hops"]:
            print(f"    hop{h['hop']}: sub_q_expanded={h['sub_question_expanded']!r}  "
                  f"committed_idx={h['committed_idx']}  is_correct={h['is_correct']}")


if __name__ == "__main__":
    main()
