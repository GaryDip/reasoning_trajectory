#!/usr/bin/env python3
"""
Strict chain@1 (and a denominator-fair recall@1) recomputed from the per-question case logs.

Two metrics in the main pipeline are *lenient* about hop-count mismatches between the predicted
decomposition and the annotation:

  chain@1   `all(pid == gid for pid, gid in zip(para_ids, gold_idxs))` -- `zip` truncates to the
            shorter side, so predicting MORE hops than annotated is free, and predicting FEWER
            means the missing gold hops are never checked.
  recall@1  scored only at hops where both sides exist (the scoring loop runs over predicted hops
            and skips those with no gold at the same index), so the denominator is
            min(len(pred), len(gold)).

Both biases favour a decomposer that predicts SHORT chains, which matters for the decomposer
ablation (WS 3.8): the MuSiQue-only model predicts fewer hops, so part of its apparent gain can
be "fewer hops were compared" rather than better retrieval.

Reported here:
  chain@1_loose   as in the run summary (reproduced, as a check)
  chain@1_strict  additionally requires len(para_ids) == len(gold_idxs)
  recall@1_full   hits / ALL annotated gold hops -- a gold hop that was never scored counts as a
                  miss, so the denominator no longer depends on the predicted hop count
  recall@1_aligned  hits / aligned hops, i.e. what the run summary reports
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def score(run_dir: Path) -> dict:
    cases = [json.loads(l) for l in next(run_dir.glob("retrieval_cases_*.jsonl")).open() if l.strip()]
    n = len(cases)
    loose = strict = 0
    hit = aligned = full = 0
    for c in cases:
        g, p = c["gold_idxs"], c["para_ids"]
        hm = [a == b for a, b in zip(p, g)]
        if hm and all(hm):
            loose += 1
            if len(p) == len(g):
                strict += 1
        hit += sum(hm)
        aligned += len(hm)
        full += len(g)
    return {"n": n,
            "chain@1_loose": round(loose / n, 4),
            "chain@1_strict": round(strict / n, 4),
            "recall@1_aligned": round(hit / aligned, 4),
            "recall@1_full": round(hit / full, 4),
            "n_hops_aligned": aligned, "n_hops_gold": full}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="LABEL=run_dir_name")
    ap.add_argument("--out", type=Path, default=HERE / "results/strict_chain_at1.json")
    args = ap.parse_args()
    out = {}
    print(f"{'run':46s} {'chain loose':>11s} {'chain strict':>12s} {'Δ':>7s} "
          f"{'r@1 aligned':>11s} {'r@1 full':>9s} {'Δ':>7s}")
    for kv in args.runs:
        label, name = kv.split("=", 1)
        d = HERE / "results" / name
        r = score(d)
        out[label] = {**r, "run_dir": name}
        print(f"{label:46s} {r['chain@1_loose']:>11.4f} {r['chain@1_strict']:>12.4f} "
              f"{100*(r['chain@1_strict']-r['chain@1_loose']):>+7.2f} "
              f"{r['recall@1_aligned']:>11.4f} {r['recall@1_full']:>9.4f} "
              f"{100*(r['recall@1_full']-r['recall@1_aligned']):>+7.2f}")
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
