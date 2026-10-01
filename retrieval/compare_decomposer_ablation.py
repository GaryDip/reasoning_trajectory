#!/usr/bin/env python3
"""Decomposer ablation table: D group (gate v3, λ=0.60) end-to-end with a swapped BART decomposer
vs the D-group row of the 0831 report (section 1.5 -- production MuSiQue+2Wiki decomposer,
otherwise identical config). Same metrics and names as that row: position-strict recall@1 /
recall@3 over the full reranked pool, chain@1, EM, F1. n_hops (hops scored) is shown last since a
different decomposer can split questions into a different number of hops.

Usage:
  python compare_decomposer_ablation.py --tag-suffix musiqueonly_decomposer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
BASELINE = {  # the 0831 report's D-group row
    "musique": "20260828_121002_gate_v3_rawprefix_musique_lambda060_dev",
    "2wiki": "20260828_123331_gate_v3_rawprefix_2wiki_lambda060_dev",
    "hotpot": "20260828_142420_gate_v3_rawprefix_hotpot_lambda060_dev",
}


def load_summary(run_dir: Path) -> dict:
    return json.loads(next(run_dir.glob("retrieval_exp_*.json")).read_text())


def row(s: dict) -> dict:
    ps = s["full_pool_gold_rank_overall"]
    return {
        "recall@1": ps["recall@1"], "recall@3": ps["recall@3"],
        "chain@1": s["final_beam_chain_match_rate_overall"],
        "EM": s["answer_overall"]["answer_em"], "F1": s["answer_overall"]["answer_f1"],
        "n_hops": ps["n_hops"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag-suffix", required=True)
    ap.add_argument("--datasets", nargs="+", default=["musique", "2wiki", "hotpot"])
    args = ap.parse_args()

    cols = ["recall@1", "recall@3", "chain@1", "EM", "F1", "n_hops"]
    out: dict[str, dict] = {}
    for ds in args.datasets:
        cand = sorted(RESULTS.glob(f"*_gate_v3_rawprefix_{ds}_lambda060_dev_{args.tag_suffix}"))
        cand = [c for c in cand if any(c.glob("retrieval_exp_*.json"))]
        if not cand:
            print(f"\n=== {ds}: no finished run for tag suffix '{args.tag_suffix}' yet")
            continue
        base, new = row(load_summary(RESULTS / BASELINE[ds])), row(load_summary(cand[-1]))
        out[ds] = {"baseline": base, args.tag_suffix: new, "run_dir": str(cand[-1])}
        print(f"\n=== {ds}   ({cand[-1].name})")
        print(f"{'':<16}" + "".join(f"{c:>10}" for c in cols))
        print(f"{'MuSiQue+2Wiki':<16}" + "".join(f"{base[c]:>10}" if c == "n_hops" else f"{base[c]:>10.4f}" for c in cols))
        print(f"{'MuSiQue-only':<16}" + "".join(f"{new[c]:>10}" if c == "n_hops" else f"{new[c]:>10.4f}" for c in cols))
        print(f"{'Δ':<16}" + "".join(f"{new[c] - base[c]:>+10d}" if c == "n_hops" else f"{new[c] - base[c]:>+10.4f}" for c in cols))

    if out:
        dst = RESULTS / f"decomposer_ablation_{args.tag_suffix}.json"
        dst.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\nsaved -> {dst}")


if __name__ == "__main__":
    main()
