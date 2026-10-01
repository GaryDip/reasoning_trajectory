#!/usr/bin/env python3
"""
Step 2 test (see update_doc/0907/0907update.md section 9): given hop-1's REAL cosine-top-1
committed evidence (right or wrong, from step 1's logic), generate a real short answer via the
local Llama vLLM server using the same prompt_short_answer production uses -- confirms the
generation piece works and produces sane text before wiring it into the multi-hop loop.

Usage:
  CUDA_VISIBLE_DEVICES=2 python step02_short_answer_test.py --n 8
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
BASE_INFER = PROJECT_ROOT.parent / "multihop_trajectory" / "llama_infer_reasoning"
sys.path.insert(0, str(TRACES_DIR))
sys.path.insert(0, str(RETRIEVAL_DIR))
sys.path.insert(0, str(BASE_INFER))

from dataset_loaders import get_decompose_path, load_decompose_bundle, load_raw_index  # noqa: E402
from trace_evidence import build_hop_queries, gold_evidence_musique, gold_hop_answers_musique  # noqa: E402
from run_retrieval_exp import embed_retrieval  # noqa: E402
from run_retrieval_exp_wavefront import BatchVllmGenerator  # noqa: E402
from run_musique_pipeline import prompt_short_answer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retrieve-k", type=int, default=10)
    ap.add_argument("--cos-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    decompose_path = get_decompose_path("musique", args.split, mode="gt")
    decomp, raw_id_map = load_decompose_bundle(decompose_path, include_enhance=False)
    raw_index = load_raw_index("musique", args.split)

    ids = sorted(decomp.keys())
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    picked = ids[: args.n]

    rows = []  # (did, k, expanded_q1, committed_para, gold_idx, gold_answer)
    for did in picked:
        raw_id = raw_id_map.get(did, did)
        record = raw_index.get(raw_id)
        if record is None:
            continue
        sub_qs = decomp[did]
        gold_texts, gold_idxs = gold_evidence_musique(record)
        hop_answers = gold_hop_answers_musique(record)
        k = min(len(sub_qs), len(gold_idxs), len(hop_answers))
        if k < 1:
            continue
        expanded_qs = build_hop_queries(sub_qs[:k], hop_answers[:k])
        paragraphs = record.get("paragraphs") or []
        ranked = embed_retrieval(expanded_qs[0], paragraphs, args.retrieve_k, args.cos_model)
        if not ranked:
            continue
        committed_para, _score = ranked[0]
        rows.append((did, k, expanded_qs[0], committed_para, gold_idxs[0], hop_answers[0]))

    print(f"Loading vLLM generator ({args.model}) ...")
    gen = BatchVllmGenerator(
        args.model, dtype=args.dtype, tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
    )

    prompts = [prompt_short_answer(eq, para) for (_did, _k, eq, para, _gi, _ga) in rows]
    raw_answers = gen.generate_batch(prompts, args.max_new_tokens, desc="hop1 short-answer")

    n_gold_hit = 0
    for (did, k, eq, para, gold_idx, gold_answer), raw in zip(rows, raw_answers):
        committed_is_gold = int(para.get("idx", -1)) == gold_idx
        print(f"\n=== {did} (K={k}) committed_is_gold={committed_is_gold} ===")
        print(f"  expanded_q: {eq!r}")
        print(f"  committed evidence title: {(para.get('title') or '')[:60]!r}")
        print(f"  generated short answer: {raw.strip()!r}")
        print(f"  gold hop answer:        {gold_answer!r}")
        if committed_is_gold:
            n_gold_hit += 1

    print(f"\n=== summary: {n_gold_hit}/{len(rows)} had gold evidence committed at hop1 "
          f"(generated answer should look reasonable there; for the rest, seeing a "
          f"plausible-but-wrong or NA answer is the EXPECTED/correct behavior) ===")


if __name__ == "__main__":
    main()
