#!/usr/bin/env python3
"""
Backfill task2's label (gate_mtl_shared_backbone, see update_doc discussion 2026-09-04): every
hop's `short_answer_generated` was stored in traces_v2, but the GOLD hop answer it should be
compared against was only ever held in-memory during build_dataset.py, never written to disk.
Pure CPU pass, no GPU/LLM calls -- looks the gold hop answers back up via the same
decompose+raw loaders build_dataset.py used, and scores each hop's short_answer_generated
against them with the same EM/F1 metric used for grading final answers.

Writes a NEW file (does not touch the input) -- swap it into place yourself once you've spot-
checked it.

Usage:
  python backfill_hop_answer_labels.py --in-path output_full/traces_v2/musique/train.jsonl \
      --out-path output_full/traces_v2/musique/train_with_hop_labels.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACES_DIR = HERE.parent / "traces"
RETRIEVAL_DIR = HERE.parent / "retrieval"
BASE_INFER = HERE.parent.parent / "multihop_trajectory" / "llama_infer_reasoning"
for p in (HERE, TRACES_DIR, RETRIEVAL_DIR, BASE_INFER):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dataset_loaders import (  # noqa: E402
    get_decompose_path, get_decompose_enhance_path, load_decompose_bundle, load_raw_index,
)
from trace_evidence import gold_hop_answers_musique  # noqa: E402
from run_musique_pipeline import judge_answer_official_from_golds  # noqa: E402


def build_gold_hop_answers_index(dataset: str, split: str) -> dict[str, list[str]]:
    decompose_path = get_decompose_path(dataset, split, mode="gt")
    enhance_path = get_decompose_enhance_path(dataset, split) if dataset == "musique" else None
    decomp, raw_id_map = load_decompose_bundle(decompose_path, enhance_path=enhance_path, include_enhance=True)
    raw_index = load_raw_index(dataset, split)

    out: dict[str, list[str]] = {}
    for did in decomp:
        raw_id = raw_id_map.get(did, did)
        record = raw_index.get(raw_id)
        if record is None:
            continue
        out[did] = gold_hop_answers_musique(record)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="musique")
    ap.add_argument("--split", default="train")
    ap.add_argument("--in-path", type=Path, required=True)
    ap.add_argument("--out-path", type=Path, required=True)
    args = ap.parse_args()

    print("building gold hop-answer index ...")
    gold_hop_answers = build_gold_hop_answers_index(args.dataset, args.split)
    print(f"  {len(gold_hop_answers)} cases indexed")

    n_lines = 0
    n_hops = 0
    n_missing_gold = 0
    n_em1 = 0
    f1_sum = 0.0
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.in_path.open(encoding="utf-8") as fin, args.out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            n_lines += 1
            golds = gold_hop_answers.get(t["case_id"])
            for h in t.get("hops", []):
                n_hops += 1
                j = h["hop"] - 1
                gold = golds[j] if golds and 0 <= j < len(golds) else None
                pred = h.get("short_answer_generated")
                if not gold or pred is None:
                    n_missing_gold += 1
                    h["hop_answer_em"] = None
                    h["hop_answer_f1"] = None
                    continue
                em, f1 = judge_answer_official_from_golds(pred, [gold])
                h["hop_answer_em"] = bool(em)
                h["hop_answer_f1"] = f1
                n_em1 += int(em)
                f1_sum += f1
            fout.write(json.dumps(t, ensure_ascii=False) + "\n")

    n_scored = n_hops - n_missing_gold
    print(f"\n{n_lines} traces / {n_hops} hops processed")
    print(f"  {n_missing_gold} hops missing gold/pred (left as null)")
    if n_scored:
        print(f"  hop_answer_em rate: {n_em1}/{n_scored} = {n_em1 / n_scored:.1%}")
        print(f"  hop_answer_f1 mean: {f1_sum / n_scored:.3f}")
    print(f"\nwrote -> {args.out_path}")


if __name__ == "__main__":
    main()
