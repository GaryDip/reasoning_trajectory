#!/usr/bin/env python3
"""
Step 4 test (see update_doc/0907/0907update.md section 9): for each recipe-2 trace, generate
the final answer from its OWN committed evidence chain and score EM/F1 against gold -- confirms
the final-reader wiring works and produces sane EM/F1 numbers (traces with all hops correct
should mostly score EM=1; traces with a wrong hop are a mixed bag, which is the whole point --
see 0907update.md's note on this field being a diagnostic signal, not just a metric).

Usage:
  CUDA_VISIBLE_DEVICES=2 python step04_final_answer_test.py --n 15 --min-k 3
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TRACES_DIR))
sys.path.insert(0, str(RETRIEVAL_DIR))

from dataset_loaders import get_decompose_path, load_decompose_bundle, load_raw_index  # noqa: E402
from trace_evidence import gold_evidence_musique, gold_hop_answers_musique  # noqa: E402
from run_retrieval_exp_wavefront import BatchVllmGenerator  # noqa: E402
from recipe2_core import run_recipe2_trace, add_final_answer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--min-k", type=int, default=1)
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

    em_when_all_correct = []
    em_when_not_all_correct = []
    for did, record, sub_qs, gold_idxs, gold_texts in cases:
        trace = run_recipe2_trace(
            case_id=did, question=record.get("question", ""), sub_qs=sub_qs,
            gold_idxs=gold_idxs, gold_texts=gold_texts, paragraphs=record.get("paragraphs") or [],
            generator=gen, cos_model=args.cos_model, retrieve_k=args.retrieve_k,
        )
        add_final_answer(trace, record, gen)

        print(f"\n=== {trace['trace_id']}  K={trace['K']}  all_correct={trace['all_correct']} "
              f"({trace['n_correct']}/{trace['K']}) ===")
        print(f"  gold_answer: {record.get('answer')!r}  gold_aliases: {record.get('answer_aliases')!r}")
        print(f"  final_answer_generated: {trace['final_answer_generated']!r}")
        print(f"  final_answer_em: {trace['final_answer_em']}  final_answer_f1: {trace['final_answer_f1']:.3f}")

        if trace["all_correct"]:
            em_when_all_correct.append(trace["final_answer_em"])
        else:
            em_when_not_all_correct.append(trace["final_answer_em"])

    def rate(xs):
        return f"{sum(xs)}/{len(xs)} ({sum(xs) / len(xs):.2%})" if xs else "n/a"

    print(f"\n=== summary: EM when all hops correct: {rate(em_when_all_correct)}  |  "
          f"EM when NOT all hops correct: {rate(em_when_not_all_correct)} ===")


if __name__ == "__main__":
    main()
