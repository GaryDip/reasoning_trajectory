#!/usr/bin/env python3
"""
Construct all-gold reasoning traces (positive / no-intervention labels).

For each example: every hop uses gold evidence from dataset annotations.
Output: traces/gold/{dataset}/{split}.jsonl

Usage:
  python construct_gold_traces.py --dataset musique --split train
  python construct_gold_traces.py --dataset musique --split dev --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dataset_loaders import get_decompose_path, iter_records, load_decompose_index
from trace_evidence import gold_evidence_musique, gold_hop_answers_musique
from trace_format import assemble_trace

PROJECT_ROOT = HERE.parent
OUT_ROOT = PROJECT_ROOT / "traces" / "gold"


def align_k(sub_qs, gold_texts, hop_answers):
    k = min(len(sub_qs), len(gold_texts), len(hop_answers))
    if k < 1:
        return [], [], [], 0
    return sub_qs[:k], gold_texts[:k], hop_answers[:k], k


def gold_evidence_2wiki(record: dict) -> tuple[list[str], list[int]]:
    # TODO: map 2Wiki supporting facts to context paragraphs
    raise NotImplementedError("2Wiki gold evidence mapping — implement in next pass")


def gold_evidence_hotpot(record: dict) -> tuple[list[str], list[int]]:
    # TODO: map Hotpot supporting_facts to context titles/sentences
    raise NotImplementedError("Hotpot gold evidence mapping — implement in next pass")


def main() -> None:
    ap = argparse.ArgumentParser(description="Construct all-gold reasoning traces.")
    ap.add_argument("--dataset", choices=("musique", "2wiki", "hotpot"), required=True)
    ap.add_argument("--split", choices=("train", "dev"), default="train")
    ap.add_argument("--decompose-mode", choices=("gt", "bart"), default="gt",
                    help="gt=MuSiQue GT NL (default for trace construction); bart=BART predictions")
    ap.add_argument("--decompose-file", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    decompose_path = args.decompose_file or get_decompose_path(
        args.dataset, args.split, mode=args.decompose_mode
    )
    if not decompose_path.is_file():
        sys.exit(f"Missing decompose file: {decompose_path}")

    out_path = args.out or (OUT_ROOT / args.dataset / f"{args.split}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    decomp = load_decompose_index(decompose_path)
    n = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for record in iter_records(args.dataset, args.split):
            rid = str(record.get("id", "")).strip()
            if rid not in decomp:
                continue
            sub_qs = decomp[rid]
            hop_answers = gold_hop_answers_musique(record) if args.dataset == "musique" else []
            if args.dataset == "musique":
                ev_texts, _ = gold_evidence_musique(record)
            elif args.dataset == "2wiki":
                ev_texts, _ = gold_evidence_2wiki(record)
            else:
                ev_texts, _ = gold_evidence_hotpot(record)

            if args.dataset == "musique":
                sub_qs, ev_texts, hop_answers, k = align_k(sub_qs, ev_texts, hop_answers)
            else:
                k = min(len(sub_qs), len(ev_texts))
                if k < 1:
                    continue
                sub_qs = sub_qs[:k]
                ev_texts = ev_texts[:k]

            if k < 1:
                continue
            q = str(record.get("question", "")).strip()
            ans = str(record.get("answer", "")).strip()
            trace = assemble_trace(q, sub_qs, ev_texts, ans)

            row = {
                "id": rid,
                "dataset": args.dataset,
                "split": args.split,
                "trace_type": "correct",
                "wrong_hops": [],
                "first_wrong_hop": -1,
                "K": k,
                "reasoning_trace": trace,
                "source_id": rid,
                "variant": "gold",
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break

    print(f"Wrote {n} gold traces → {out_path}")


if __name__ == "__main__":
    main()
