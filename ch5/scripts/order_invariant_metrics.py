#!/usr/bin/env python3
"""Chapter 5's view of the gold-order protocol: main (λ=0.60) vs control (λ=0), both ways.

The protocol itself lives in `retrieval/goldmatch.py` (see `ORDER_INVARIANT_TASK.md`); the full
32-run Chapter 4 recomputation is `retrieval/compute_order_invariant_v2.py`. This script keeps
the small, Chapter-5-facing table next to the rest of ch5/, and is the self-check that MuSiQue --
where order-invariance does not apply -- comes out bit-identical under both protocols.

  chain@1    every aligned hop hit AND len(para_ids) == len(gold_idxs)
  recall@1   correct hops / all gold hops (full denominator: a missing hop counts as a miss)
  order-invariant  para_ids[j] in set(gold_idxs) and not already claimed by an earlier hop

EM and F1 are untouched by any of this: they never read gold_idxs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNS = ROOT / "retrieval/results"
sys.path.insert(0, str(ROOT / "retrieval"))
from goldmatch import (  # noqa: E402
    chain_at1, hop_hits, order_invariant_applies, selected_a_gold_twice,
)

ARMS = {
    "musique": {"gate_lambda0p60": "20261008_071635_l15_musique_lambda0p60_dev",
                "control_lambda0": "20261003_070503_nogate_musique_lambda0_dev"},
    "2wiki": {"gate_lambda0p60": "20261008_085006_l15_2wiki_lambda0p60_dev",
              "control_lambda0": "20261003_070301_nogate_2wiki_lambda0_dev"},
    "hotpot": {"gate_lambda0p60": "20261008_074208_l15_hotpot_lambda0p60_dev",
               "control_lambda0": "20261003_070253_nogate_hotpot_lambda0_dev"},
}
METRICS = ("em", "chain@1_positional", "chain@1_order_invariant",
           "recall@1_positional", "recall@1_order_invariant")


def load_2wiki_types() -> dict[str, str]:
    return {str(r["_id"]): r.get("type", "")
            for r in json.loads((ROOT / "data/raw/2wikimultihopqa/dev.json").read_text())}


def score(ds: str, run: str, types: dict[str, str]) -> dict:
    cases = [json.loads(l) for l in next((RUNS / run).glob("retrieval_cases_*.jsonl")).open()
             if l.strip()]
    n = len(cases)
    chain = {"positional": 0, "order_invariant": 0}
    hit = {"positional": 0, "order_invariant": 0}
    gold = em = n_oi = n_dup = 0
    for c in cases:
        p, g = c["para_ids"], c["gold_idxs"]
        oi = order_invariant_applies(ds, types.get(c["id"], "") if ds == "2wiki" else "")
        n_oi += oi
        n_dup += oi and selected_a_gold_twice(p, g)
        gold += len(g); em += c["em"]
        for proto, use_oi in (("positional", False), ("order_invariant", oi)):
            hit[proto] += sum(hop_hits(p, g, use_oi))
            chain[proto] += chain_at1(p, g, use_oi)[1]
    return {"run": run, "n": n, "n_order_invariant": n_oi,
            "n_selected_a_gold_twice": n_dup, "em": round(em / n, 4),
            "chain@1_positional": round(chain["positional"] / n, 4),
            "chain@1_order_invariant": round(chain["order_invariant"] / n, 4),
            "recall@1_positional": round(hit["positional"] / gold, 4),
            "recall@1_order_invariant": round(hit["order_invariant"] / gold, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "ch5/results/order_invariant_metrics.json")
    args = ap.parse_args()
    types = load_2wiki_types()
    out: dict = {"_protocol": {"module": "retrieval/goldmatch.py",
                               "full_chapter4_recomputation":
                                   "retrieval/results/order_invariant_v2.json"}}
    hdr = (f"{'dataset/arm':28s} {'EM':>7s} {'chain pos':>9s} {'chain oi':>9s} "
           f"{'rec pos':>8s} {'rec oi':>7s} {'dup gold':>9s}")
    print(hdr); print("-" * len(hdr))
    for ds, arms in ARMS.items():
        out[ds] = {k: score(ds, r, types) for k, r in arms.items()}
        for k, v in out[ds].items():
            print(f"{ds + '/' + k:28s} {v['em']:7.4f} {v['chain@1_positional']:9.4f} "
                  f"{v['chain@1_order_invariant']:9.4f} {v['recall@1_positional']:8.4f} "
                  f"{v['recall@1_order_invariant']:7.4f} {v['n_selected_a_gold_twice']:9d}")
        g, c = out[ds]["gate_lambda0p60"], out[ds]["control_lambda0"]
        out[ds]["gate_minus_control_points"] = {m: round(100 * (g[m] - c[m]), 2) for m in METRICS}
        print(f"{ds + '/  Δ (points)':28s} "
              + "  ".join(f"{v:+.2f}" for v in out[ds]["gate_minus_control_points"].values()))
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
