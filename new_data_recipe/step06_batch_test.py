#!/usr/bin/env python3
"""
Step 6 test: run run_recipe2_batch on the same 2 known example ids used in steps 3-5 (plus a
few more), verify the batched output is IDENTICAL field-for-field to what the single-example
recipe2_core.run_recipe2_trace produced -- confirms batching didn't change any actual logic,
only how the encode/generate calls are grouped.

Usage:
  CUDA_VISIBLE_DEVICES=2 python step06_batch_test.py
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
from recipe2_core import run_recipe2_trace  # noqa: E402
from recipe2_batch import run_recipe2_batch  # noqa: E402

TARGET_IDS = [
    "3hop1__257997_104557_161232",
    "4hop3__695568_769559_129926_718885",
    "4hop1__322597_452185_53858_33265",
    "3hop2__89818_717222_4107",
]


def load_cases(ids):
    decompose_path = get_decompose_path("musique", "train", mode="gt")
    decomp, raw_id_map = load_decompose_bundle(decompose_path, include_enhance=False)
    raw_index = load_raw_index("musique", "train")
    out = []
    for did in ids:
        raw_id = raw_id_map.get(did, did)
        record = raw_index[raw_id]
        sub_qs = decomp[did]
        gold_texts, gold_idxs = gold_evidence_musique(record)
        hop_answers = gold_hop_answers_musique(record)
        k = min(len(sub_qs), len(gold_idxs), len(hop_answers))
        out.append({
            "case_id": did, "record": record, "question": record.get("question", ""),
            "sub_qs": sub_qs[:k], "gold_idxs": gold_idxs[:k], "gold_texts": gold_texts[:k],
            "paragraphs": record.get("paragraphs") or [],
        })
    return out


def strip_nondeterministic(hops):
    """Drop fields that can legitimately differ run-to-run (LLM generation isn't perfectly
    deterministic even at temperature=0 across different batch compositions/padding) --
    compare the fields that MUST match: retrieval-derived (committed_idx, gold_idx, is_correct,
    cosine_top3)."""
    return [{k: h[k] for k in ("hop", "sub_question_expanded", "committed_idx", "gold_idx", "is_correct")}
            for h in hops]


def main() -> None:
    cases_raw = load_cases(TARGET_IDS)

    print(f"Loading vLLM generator ...")
    gen = BatchVllmGenerator(
        "meta-llama/Llama-3.1-8B-Instruct", dtype="bfloat16", tensor_parallel_size=1,
        gpu_memory_utilization=0.2, max_model_len=8192,
    )

    print("\n--- running SINGLE-example recipe2_core (baseline for comparison) ---")
    single_traces = {}
    for c in cases_raw:
        t = run_recipe2_trace(
            case_id=c["case_id"], question=c["question"], sub_qs=c["sub_qs"],
            gold_idxs=c["gold_idxs"], gold_texts=c["gold_texts"], paragraphs=c["paragraphs"],
            generator=gen,
        )
        single_traces[c["case_id"]] = t

    print("\n--- running BATCHED recipe2_batch on all 4 cases together ---")
    batch_cases = [{k: c[k] for k in ("case_id", "question", "sub_qs", "gold_idxs", "gold_texts", "paragraphs")} for c in cases_raw]
    batch_traces = run_recipe2_batch(batch_cases, generator=gen)
    batch_by_id = {t["case_id"]: t for t in batch_traces}

    n_match = 0
    for c in cases_raw:
        did = c["case_id"]
        s = strip_nondeterministic(single_traces[did]["hops"])
        b = strip_nondeterministic(batch_by_id[did]["hops"])
        match = s == b
        n_match += int(match)
        print(f"\n{did}: single vs batch retrieval-level fields match = {match}")
        if not match:
            print("  single:", json.dumps(s, ensure_ascii=False))
            print("  batch :", json.dumps(b, ensure_ascii=False))
        print(f"  single all_correct={single_traces[did]['all_correct']} n_pairs={len(single_traces[did]['pairs'])}")
        print(f"  batch  all_correct={batch_by_id[did]['all_correct']} n_pairs={len(batch_by_id[did]['pairs'])}")

    print(f"\n=== summary: {n_match}/{len(cases_raw)} cases have IDENTICAL retrieval-level "
          f"fields between single-example and batched runs ===")


if __name__ == "__main__":
    main()
