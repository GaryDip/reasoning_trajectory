#!/usr/bin/env python3
"""
Merge gold + counterfactual trace jsonl into traces/merged/ with manifest.

Usage:
  python merge_traces.py --dataset musique --split train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--gold", type=Path, default=None)
    ap.add_argument("--counterfactual", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    gold_path = args.gold or PROJECT_ROOT / "traces" / "gold" / args.dataset / f"{args.split}.jsonl"
    cf_path = args.counterfactual or PROJECT_ROOT / "traces" / "counterfactual" / args.dataset / f"{args.split}.jsonl"
    out_path = args.out or PROJECT_ROOT / "traces" / "merged" / args.dataset / f"{args.split}.jsonl"
    manifest_path = out_path.with_suffix(".manifest.jsonl")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_gold = n_cf = 0
    with open(out_path, "w", encoding="utf-8") as out_f, open(
        manifest_path, "w", encoding="utf-8"
    ) as man_f:
        for path, label in ((gold_path, "gold"), (cf_path, "counterfactual")):
            if not path.is_file():
                print(f"Skip missing {path}")
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    man_f.write(json.dumps({"source": label, "id": row["id"]}, ensure_ascii=False) + "\n")
                    if label == "gold":
                        n_gold += 1
                    else:
                        n_cf += 1

    print(f"Merged {n_gold} gold + {n_cf} counterfactual → {out_path}")


if __name__ == "__main__":
    main()
