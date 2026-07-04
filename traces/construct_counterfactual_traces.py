#!/usr/bin/env python3
"""
Construct counterfactual traces for LR gate training.

Protocol (MuSiQue train):
  - sub_questions: GT natural-language decompose
  - hops 1..h-1: gold evidence
  - hop h: wrong evidence from cosine top-k \\ gold (hard negative)
  - hops h+1..K: gold evidence (only transition j=h-1 is wrong)

One source example yields up to K * max_wrong_per_hop negative traces.
Use construct_balanced_traces.py to merge with gold and balance by (K, hop).

Usage:
  python construct_counterfactual_traces.py --dataset musique --split train
  python construct_counterfactual_traces.py --cos-topk 10 --max-wrong-per-hop 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dataset_loaders import get_decompose_path, iter_records, load_decompose_index
from trace_evidence import (
    cosine_non_gold_wrong_texts,
    expand_hop_template,
    gold_evidence_musique,
    gold_hop_answers_musique,
)
from trace_format import assemble_trace

PROJECT_ROOT = HERE.parent
OUT_ROOT = PROJECT_ROOT / "traces" / "counterfactual"


def align_subqs_and_gold(
    sub_qs: list[str],
    gold_texts: list[str],
    hop_answers: list[str],
) -> tuple[list[str], list[str], list[str], int]:
    k = min(len(sub_qs), len(gold_texts), len(hop_answers))
    if k < 1:
        return [], [], [], 0
    return sub_qs[:k], gold_texts[:k], hop_answers[:k], k


def main() -> None:
    ap = argparse.ArgumentParser(description="Construct cosine-hard counterfactual traces.")
    ap.add_argument("--dataset", choices=("musique", "2wiki", "hotpot"), default="musique")
    ap.add_argument("--split", choices=("train", "dev"), default="train")
    ap.add_argument("--decompose-mode", choices=("gt", "bart"), default="gt")
    ap.add_argument("--decompose-file", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--cos-topk", type=int, default=10)
    ap.add_argument("--max-wrong-per-hop", type=int, default=3,
                    help="How many distinct cosine non-gold passages per (example, hop)")
    ap.add_argument("--cos-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--limit-examples", type=int, default=0)
    ap.add_argument("--limit-traces", type=int, default=0)
    args = ap.parse_args()

    if args.dataset != "musique":
        sys.exit("Counterfactual cosine construction currently supports musique only.")

    decompose_path = args.decompose_file or get_decompose_path(
        args.dataset, args.split, mode=args.decompose_mode
    )
    if not decompose_path.is_file():
        sys.exit(f"Missing decompose file: {decompose_path}")

    out_path = args.out or (OUT_ROOT / args.dataset / f"{args.split}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    decomp = load_decompose_index(decompose_path)
    n_ex = n_tr = 0
    stats: dict[str, int] = {}

    with open(out_path, "w", encoding="utf-8") as out_f:
        for record in iter_records(args.dataset, args.split):
            rid = str(record.get("id", "")).strip()
            if rid not in decomp:
                continue

            sub_qs = decomp[rid]
            gold_texts, gold_idxs_list = gold_evidence_musique(record)
            hop_answers = gold_hop_answers_musique(record)
            sub_qs, gold_texts, hop_answers, k = align_subqs_and_gold(
                sub_qs, gold_texts, hop_answers
            )
            if k < 1:
                continue

            gold_idxs = set(gold_idxs_list[:k])
            paragraphs = list(record.get("paragraphs") or [])
            q = str(record.get("question", "")).strip()
            ans = str(record.get("answer", "")).strip()
            prior: list[str] = []

            for h in range(1, k + 1):
                query = expand_hop_template(sub_qs[h - 1], prior)
                wrong_cands = cosine_non_gold_wrong_texts(
                    query,
                    paragraphs,
                    gold_idxs,
                    topk=args.cos_topk,
                    max_wrong=args.max_wrong_per_hop,
                    cos_model=args.cos_model,
                )
                for vi, (wrong_text, wrong_pidx, score) in enumerate(wrong_cands):
                    ev_texts = list(gold_texts)
                    ev_texts[h - 1] = wrong_text
                    trace = assemble_trace(q, sub_qs, ev_texts, ans)
                    aug_id = f"{rid}__wrong_h{h}__cos{vi}"
                    row = {
                        "id": aug_id,
                        "dataset": args.dataset,
                        "split": args.split,
                        "trace_type": "error",
                        "wrong_hops": [h],
                        "first_wrong_hop": h,
                        "K": k,
                        "reasoning_trace": trace,
                        "source_id": rid,
                        "variant": f"wrong_h{h}_cos{vi}",
                        "wrong_mode": "cosine_topk_non_gold",
                        "wrong_hop": h,
                        "wrong_paragraph_idx": wrong_pidx,
                        "cosine_score": round(score, 4),
                        "cos_query": query,
                    }
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_tr += 1
                    key = f"K{k}_h{h}"
                    stats[key] = stats.get(key, 0) + 1
                    if args.limit_traces and n_tr >= args.limit_traces:
                        break
                prior.append(hop_answers[h - 1])
                if args.limit_traces and n_tr >= args.limit_traces:
                    break

            n_ex += 1
            if args.limit_examples and n_ex >= args.limit_examples:
                break
            if args.limit_traces and n_tr >= args.limit_traces:
                break

    print(f"Wrote {n_tr} counterfactual traces from {n_ex} examples → {out_path}")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")


if __name__ == "__main__":
    main()
