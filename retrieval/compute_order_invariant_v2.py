#!/usr/bin/env python3
"""Recompute every Chapter 4 retrieval metric under both gold-matching protocols.

See `ORDER_INVARIANT_TASK.md` for the decision and `retrieval/goldmatch.py` for the protocol.
This script does NOT touch `compute_strict_chain.py` or any existing result file: the old numbers
stay where they are so the two can be compared.

Per run it reports recall@1 / recall@3 (aligned and full denominator) and chain@1 (loose and
strict), each positionally and order-invariantly, plus two diagnostics: how many questions picked
the same gold passage at two hops (what the first order-invariant implementation let through) and,
for 2Wiki, the bridge_comparison precedence variant.

Output: retrieval/results/order_invariant_v2.json      (CPU only, ~2 min)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE))
from goldmatch import (  # noqa: E402
    bridge_comparison_precedence_ok, chain_at1, hop_hits, hop_hits_at_k,
    order_invariant_applies, selected_a_gold_twice,
)

# (label, dataset, run_dir) grouped by the Chapter 4 table they feed.
TABLES: dict[str, list[tuple[str, str, str]]] = {
    "4.6_main_and_control": [
        ("musique/gate_lambda0p60", "musique", "20261008_071635_l15_musique_lambda0p60_dev"),
        ("musique/control_lambda0", "musique", "20261003_070503_nogate_musique_lambda0_dev"),
        ("2wiki/gate_lambda0p60", "2wiki", "20261008_085006_l15_2wiki_lambda0p60_dev"),
        ("2wiki/control_lambda0", "2wiki", "20261003_070301_nogate_2wiki_lambda0_dev"),
        ("hotpot/gate_lambda0p60", "hotpot", "20261008_074208_l15_hotpot_lambda0p60_dev"),
        ("hotpot/control_lambda0", "hotpot", "20261003_070253_nogate_hotpot_lambda0_dev"),
    ],
    "4.9_feature_ablation": [
        ("musique/B_delta_only", "musique", "20261008_204721_l15_B_deltaonly_musique_dev"),
        ("musique/C_h_only", "musique", "20261008_213411_l15_C_honly_musique_dev"),
        ("hotpot/B_delta_only", "hotpot", "20261008_221555_l15_B_deltaonly_hotpot_dev"),
        ("hotpot/C_h_only", "hotpot", "20261008_233432_l15_C_honly_hotpot_dev"),
        ("2wiki/B_delta_only", "2wiki", "20261009_002550_l15_B_deltaonly_2wiki_dev"),
        ("2wiki/C_h_only", "2wiki", "20261009_021558_l15_C_honly_2wiki_dev"),
    ],
    "4.10_decomposer_ablation": [
        ("musique/musique_only_decomposer", "musique",
         "20261008_154506_l15_musique_musiqueonly_decomposer"),
        ("hotpot/musique_only_decomposer", "hotpot",
         "20261008_162924_l15_hotpot_musiqueonly_decomposer"),
        ("2wiki/musique_only_decomposer", "2wiki",
         "20261008_175915_l15_2wiki_musiqueonly_decomposer"),
    ],
    "4.11_colbert": [
        ("musique/colbert_nogate", "musique", "20260818_154358_colbert_musique_nogate_full"),
        ("musique/colbert_gate", "musique", "20261008_112624_l15_colbert_musique_gate"),
        ("2wiki/colbert_nogate", "2wiki", "20260818_170355_colbert_2wiki_nogate_full"),
        ("2wiki/colbert_gate", "2wiki", "20261008_124446_l15_colbert_2wiki_gate"),
        ("hotpot/colbert_nogate", "hotpot", "20260819_003311_colbert_hotpot_nogate_full"),
        ("hotpot/colbert_gate", "hotpot", "20261008_114748_l15_colbert_hotpot_gate"),
    ],
    "4.12_global_corpus": [
        ("musique/global_k10_lambda0", "musique", "20261008_161412_l15_nogate_global_musique_k10"),
        ("musique/global_k10_lambda0p60", "musique", "20261009_061318_l15_global_musique_k10"),
        ("musique/global_k50_lambda0", "musique", "20261009_035050_l15_nogate_global_musique_k50"),
        ("musique/global_k50_lambda0p60", "musique", "20261008_161903_l15_global_musique_k50"),
        ("2wiki/global_k10_lambda0", "2wiki", "20261008_201547_l15_nogate_global_2wiki_k10"),
        ("2wiki/global_k10_lambda0p60", "2wiki", "20261009_071742_l15_global_2wiki_k10"),
        ("2wiki/global_k50_lambda0", "2wiki", "20261009_134820_l15_nogate_global_2wiki_k50"),
        ("2wiki/global_k50_lambda0p60", "2wiki", "20261008_211248_l15_global_2wiki_k50"),
        ("hotpot/global_k10_lambda0", "hotpot", "20261008_172723_l15_nogate_global_hotpot_k10"),
        ("hotpot/global_k10_lambda0p60", "hotpot", "20261009_063306_l15_global_hotpot_k10"),
        ("hotpot/global_k50_lambda0", "hotpot", "20261009_082043_l15_nogate_global_hotpot_k50"),
        ("hotpot/global_k50_lambda0p60", "hotpot", "20261008_174752_l15_global_hotpot_k50"),
    ],
}


def load_types() -> dict[str, str]:
    return {str(r["_id"]): r.get("type", "")
            for r in json.loads((ROOT / "data/raw/2wikimultihopqa/dev.json").read_text())}


def load_cases(run: str) -> list[dict]:
    d = RESULTS / run
    f = next(d.glob("retrieval_cases_*.jsonl"), None)
    if f is None:
        raise FileNotFoundError(f"no retrieval_cases_*.jsonl in {d}")
    return [json.loads(l) for l in f.open() if l.strip()]


def score(dataset: str, run: str, types: dict[str, str]) -> dict:
    cases = load_cases(run)
    n = len(cases)
    acc = {p: {"hit1": 0, "hit3": 0, "aligned": 0, "loose": 0, "strict": 0}
           for p in ("positional", "order_invariant")}
    gold_total = 0
    em = f1 = 0.0
    n_oi = n_dup = strict_nodedup = 0
    bc_n = bc_ok = bc_perm = 0
    have_ranked = True
    for c in cases:
        p, g = c["para_ids"], c["gold_idxs"]
        qtype = types.get(c["id"], "") if dataset == "2wiki" else ""
        oi = order_invariant_applies(dataset, qtype)
        n_oi += oi
        gold_total += len(g)
        em += c["em"]; f1 += c.get("f1") or 0.0
        ranked = c.get("hop_ranked_para_ids")
        if not ranked:
            have_ranked = False
        for proto, use_oi in (("positional", False), ("order_invariant", oi)):
            a = acc[proto]
            hits = hop_hits(p, g, use_oi)
            a["hit1"] += sum(hits); a["aligned"] += len(hits)
            h3 = hop_hits_at_k(p, g, ranked, use_oi, k=3)
            if h3 is not None:
                a["hit3"] += sum(h3)
            loose, strict = chain_at1(p, g, use_oi)
            a["loose"] += loose; a["strict"] += strict
        if oi:
            n_dup += selected_a_gold_twice(p, g)
            loose_nd, _ = chain_at1(p, g, True, dedup=False)
            strict_nodedup += loose_nd and len(p) == len(g)
        else:
            _, s_pos = chain_at1(p, g, False)
            strict_nodedup += s_pos
        if dataset == "2wiki" and qtype == "bridge_comparison":
            v = bridge_comparison_precedence_ok(p, g)
            if v is not None:
                bc_n += 1; bc_ok += v
                bc_perm += (set(p) == set(g) and len(set(p)) == len(p))

    def block(proto: str) -> dict:
        a = acc[proto]
        d = {"recall@1_aligned": round(a["hit1"] / max(a["aligned"], 1), 4),
             "recall@1_full": round(a["hit1"] / max(gold_total, 1), 4),
             "chain@1_loose": round(a["loose"] / n, 4),
             "chain@1_strict": round(a["strict"] / n, 4)}
        if have_ranked:
            d["recall@3_aligned"] = round(a["hit3"] / max(a["aligned"], 1), 4)
            d["recall@3_full"] = round(a["hit3"] / max(gold_total, 1), 4)
        else:
            d["recall@3_aligned"] = d["recall@3_full"] = None
        return d

    out = {"run": run, "dataset": dataset, "n": n,
           "n_order_invariant": n_oi, "n_hops_gold": gold_total,
           "n_hops_aligned": acc["positional"]["aligned"],
           "em": round(em / n, 4), "f1": round(f1 / n, 4),
           "has_candidate_lists": have_ranked,
           "positional": block("positional"),
           "order_invariant": block("order_invariant"),
           "diagnostics": {
               "n_selected_a_gold_twice": n_dup,
               "share_of_order_invariant_questions": round(n_dup / max(n_oi, 1), 4),
               "chain@1_strict_order_invariant_without_dedup": round(strict_nodedup / n, 4),
               "chain@1_strict_overstated_by_old_rule_points": round(
                   100 * (strict_nodedup - acc["order_invariant"]["strict"]) / n, 2)}}
    if bc_n:
        out["diagnostics"]["bridge_comparison"] = {
            "n": bc_n,
            "chain@1_strict_permutation": round(bc_perm / bc_n, 4),
            "chain@1_strict_precedence": round(bc_ok / bc_n, 4),
            "note": "precedence keeps g1<g3 and g2<g4 (each film before its own director); "
                    "permutation is the headline rule, see goldmatch.py"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RESULTS / "order_invariant_v2.json")
    ap.add_argument("--tables", nargs="*", default=None)
    args = ap.parse_args()
    types = load_types()
    out: dict = {"_protocol": {
        "module": "retrieval/goldmatch.py",
        "task": "ORDER_INVARIANT_TASK.md",
        "order_invariant_scope": "HotpotQA: all questions; 2Wiki: comparison + "
                                 "bridge_comparison; MuSiQue and 2Wiki compositional/inference "
                                 "stay positional (both protocols coincide there)",
        "per_hop_rule": "para_ids[j] in set(gold_idxs) and para_ids[j] not in para_ids[:j]",
        "chain@1_strict_rule": "len(para_ids) == len(gold_idxs) and para_ids is a permutation "
                               "of gold_idxs",
        "bridge_comparison": "ORDER_INVARIANT_TASK.md assumed the dedup'd supporting_facts order "
                             "is branch1's two passages then branch2's; it is NOT -- it is "
                             "interleaved (branch1 hop1, branch2 hop1, branch1 hop2, branch2 "
                             "hop2), verified on all 2,751 dev questions against the evidences "
                             "triples. Per that document's fallback the headline rule is plain "
                             "permutation-without-reuse; the branch-precedence variant is "
                             "reported as a diagnostic only.",
        "recall@3": "null where the run's case log predates hop_ranked_para_ids "
                    "(the B/C feature arms and the 2026-08 ColBERT no-gate runs)"}}
    for table, runs in TABLES.items():
        if args.tables and table not in args.tables:
            continue
        out[table] = {}
        print(f"=== {table}")
        for label, ds, run in runs:
            r = score(ds, run, types)
            out[table][label] = r
            p, o = r["positional"], r["order_invariant"]
            r3 = "   n/a" if p["recall@3_full"] is None else f"{p['recall@3_full']:6.4f}"
            o3 = "   n/a" if o["recall@3_full"] is None else f"{o['recall@3_full']:6.4f}"
            print(f"  {label:34s} EM {r['em']:.4f} | chain@1s {p['chain@1_strict']:.4f}"
                  f" -> {o['chain@1_strict']:.4f} | r@1full {p['recall@1_full']:.4f}"
                  f" -> {o['recall@1_full']:.4f} | r@3full {r3} -> {o3}"
                  f" | dupgold {r['diagnostics']['n_selected_a_gold_twice']}")
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
