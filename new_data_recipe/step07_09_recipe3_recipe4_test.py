#!/usr/bin/env python3
"""
Step 7 + 9 test: recipe 3 (forced wrong-hop injection, generating a variant per possible
injection position, even for a case recipe 2 already solved fully) and recipe 4 (forced clean
gold prefix + real continuation, for a case recipe 2 did NOT solve).

Usage:
  CUDA_VISIBLE_DEVICES=0 python step07_09_recipe3_recipe4_test.py
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
from recipe2_batch import run_recipe2_batch  # noqa: E402


def load_case(did, decomp, raw_id_map, raw_index):
    raw_id = raw_id_map.get(did, did)
    record = raw_index[raw_id]
    sub_qs = decomp[did]
    gold_texts, gold_idxs = gold_evidence_musique(record)
    hop_answers = gold_hop_answers_musique(record)
    k = min(len(sub_qs), len(gold_idxs), len(hop_answers))
    return {
        "case_id": did, "question": record.get("question", ""), "sub_qs": sub_qs[:k],
        "gold_idxs": gold_idxs[:k], "gold_texts": gold_texts[:k],
        "paragraphs": record.get("paragraphs") or [],
    }, hop_answers[:k]


def print_trace_brief(trace):
    print(f"  trace_id={trace['trace_id']}  all_correct={trace['all_correct']} "
          f"({trace['n_correct']}/{trace['K']})  n_pairs={len(trace['pairs'])}")
    for h in trace["hops"]:
        mark = "OK   " if h["is_correct"] else "WRONG"
        forced = " [FORCED]" if h.get("forced") else ""
        print(f"    hop{h['hop']} [{mark}]{forced} committed_idx={h['committed_idx']} "
              f"gold_idx={h['gold_idx']}  sub_q={h['sub_question_expanded']!r}")


def main() -> None:
    decompose_path = get_decompose_path("musique", "train", mode="gt")
    decomp, raw_id_map = load_decompose_bundle(decompose_path, include_enhance=False)
    raw_index = load_raw_index("musique", "train")

    recipe3_case, _ = load_case("3hop1__257997_104557_161232", decomp, raw_id_map, raw_index)
    recipe4_case, recipe4_gold_answers = load_case("4hop1__322597_452185_53858_33265", decomp, raw_id_map, raw_index)

    print("Loading vLLM generator ...")
    gen = BatchVllmGenerator(
        "meta-llama/Llama-3.1-8B-Instruct", dtype="bfloat16", tensor_parallel_size=1,
        gpu_memory_utilization=0.2, max_model_len=8192,
    )

    # --- Step 7: recipe 3 -- 3 variants of the SAME (already-fully-correct) case, one per
    # forced injection hop (1, 2, 3). Each is its own "case" in the batch (same underlying
    # question/paragraphs, different forced_hops entry), all run together in ONE batch call.
    print("\n" + "=" * 80)
    print("STEP 7: recipe 3 (forced wrong-hop injection) on a case recipe2 already solved")
    print("=" * 80)
    variants = [recipe3_case, recipe3_case, recipe3_case]
    forced_hops = {0: 1, 1: 2, 2: 3}
    r3_traces = run_recipe2_batch(
        variants, generator=gen, forced_hops=forced_hops,
        trace_id_suffix="recipe3_forced", recipe_name="3_forced_wrong",
    )
    # trace_id collides across the 3 variants since case_id is identical -- disambiguate for
    # display/storage using the forced hop (this matches what the real pipeline needs to do:
    # append the forced hop to trace_id when generating multiple recipe-3 variants per case).
    for i, t in enumerate(r3_traces):
        t["trace_id"] = f"{t['trace_id']}_h{forced_hops[i]}"
        print(f"\nforced_hop={forced_hops[i]}:")
        print_trace_brief(t)
        if t["pairs"]:
            print(f"    pair sample: {json.dumps(t['pairs'][0], ensure_ascii=False)[:200]}...")

    # --- Step 9: recipe 4 -- force hops 1..N gold (using gold hop answers for expansion),
    # then let it continue via REAL retrieval from hop N+1 onward. Applied to a case recipe 2
    # did NOT solve (originally failed at hop 2).
    print("\n" + "=" * 80)
    print("STEP 9: recipe 4 (forced clean prefix + real continuation) on a case recipe2 failed")
    print("=" * 80)
    r4_traces = run_recipe2_batch(
        [recipe4_case], generator=gen,
        clean_prefix_upto={0: 2}, clean_prefix_gold_hop_answers={0: recipe4_gold_answers},
        trace_id_suffix="recipe4_cleanprefix2", recipe_name="4_clean_prefix",
    )
    print_trace_brief(r4_traces[0])


if __name__ == "__main__":
    main()
