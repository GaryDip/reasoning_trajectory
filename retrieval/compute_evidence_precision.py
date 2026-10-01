#!/usr/bin/env python3
"""Evidence-set precision for the D-group / ChainRAG / GRITHopper comparison (0831 report,
section 2.4), computed from the three methods' existing per-question outputs -- no rerun.

Per question, the "retrieved set" is the same set the 0831 gold-coverage metric used:
  D group     -- titles of the committed paragraph at each hop (case log `para_ids`)
  GRITHopper  -- titles of the top-1 pick at each iterative step (`retrieved_titles`)
  ChainRAG    -- union over sub-questions of the titles behind every sentence that ended up
                 in the final answer context after graph expansion (`retrieved_titles_by_subq`)
Gold set = titles of the dataset's supporting paragraphs (`gold_titles_for`, shared by all three).

  precision = |retrieved ∩ gold| / |retrieved|     (set-based, title level)
  coverage  = |retrieved ∩ gold| / |gold|          (== 0831 "gold 覆盖率", recomputed as a check)

Reported both macro (mean of per-question values) and micro (pooled counts), plus the mean
retrieved-set size so precision can be read against how many passages each method keeps.

Per-hop ranking metrics, over each retrieval decision unit (D hop / GRITHopper step / ChainRAG
sub-question), on that unit's own ranked list (titles, deduplicated in rank order):
  recall@1 / recall@3 -- any gold title within top-1 / top-3 (== 0831 order-invariant recall,
                         recomputed as a check)
  precision@3         -- (# gold titles within top-3) / 3
D group needs a run that logged `hop_ranked_para_ids` (pass it via --d-runs); its hops are
truncated to len(gold_idxs), the same hop set its own recall@k was accumulated over.

Usage:
  python compute_evidence_precision.py            # all three datasets
  python compute_evidence_precision.py --datasets musique
  python compute_evidence_precision.py --d-runs musique=<run_dir> 2wiki=<run_dir> hotpot=<run_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from run_retrieval_exp import load_dataset_records  # noqa: E402

D_RUNS = {
    "musique": "20260828_155235_gate_v3_rawprefix_musique_lambda060_dev_orderinvariant",
    "2wiki": "20260828_161558_gate_v3_rawprefix_2wiki_lambda060_dev_orderinvariant",
    "hotpot": "20260828_180622_gate_v3_rawprefix_hotpot_lambda060_dev_orderinvariant",
}
CHAINRAG_NAME = {"musique": "musique", "2wiki": "2wikimqa", "hotpot": "hotpotqa"}


def gold_titles_for(row: dict) -> set[str]:
    # same definition as grithopper/run_retrieval.py::gold_titles_for
    gold_idxs = {qd.get("paragraph_support_idx") for qd in (row.get("question_decomposition") or [])}
    return {p["title"] for p in (row.get("paragraphs") or []) if p.get("idx") in gold_idxs}


def load_concat_json(path: Path) -> list[dict]:
    """ChainRAG's results.jsonl is pretty-printed JSON objects back to back, not one per line."""
    s, dec, i, out = path.read_text(encoding="utf-8"), json.JSONDecoder(), 0, []
    while i < len(s):
        while i < len(s) and s[i].isspace():
            i += 1
        if i >= len(s):
            break
        obj, i = dec.raw_decode(s, i)
        out.append(obj)
    return out


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dedup(seq: list[str]) -> list[str]:
    return list(dict.fromkeys(seq))


def load_method_outputs(dataset: str, records: dict[str, dict], d_run: str):
    """Returns ({method: {example_id: retrieved title set}},
                {method: {example_id: [ranked title list per hop]}}) -- a method is absent from
    the second dict when its outputs carry no per-hop ranking (old D runs)."""
    out: dict[str, dict[str, set[str]]] = {}
    ranked: dict[str, dict[str, list[list[str]]]] = {}

    d_dir = Path(d_run) if Path(d_run).is_absolute() else HERE / "results" / d_run
    d_cases = load_jsonl(next(d_dir.glob("retrieval_cases_*.jsonl")))
    out["D 组"] = {}
    if all("hop_ranked_para_ids" in c for c in d_cases):
        ranked["D 组"] = {}
    for c in d_cases:
        idx_to_title = {p["idx"]: p["title"] for p in records[c["id"]]["paragraphs"]}
        out["D 组"][c["id"]] = {idx_to_title[i] for i in c["para_ids"] if i in idx_to_title}
        if "D 组" in ranked:
            hops = c["hop_ranked_para_ids"][: len(c["gold_idxs"])]
            ranked["D 组"][c["id"]] = [dedup([idx_to_title[i] for i in h if i in idx_to_title]) for h in hops]

    # ChainRAG: question_id is the line index into its converted input file, which carries _id
    cr_name = CHAINRAG_NAME[dataset]
    cr_inputs = load_jsonl(ROOT / "chainrag" / "data" / f"{cr_name}.jsonl")
    cr_results = load_concat_json(ROOT / "chainrag" / "processed_data" / cr_name / "results.jsonl")
    out["ChainRAG"] = {}
    for r in cr_results:
        inp = cr_inputs[r["question_id"]]
        assert inp["input"].strip() == r["question"].strip(), f"id misalignment at {r['question_id']}"
        titles: set[str] = set()
        for sq in r.get("retrieved_titles_by_subq") or []:
            titles.update(sq.get("retrieved_titles") or [])
        out["ChainRAG"][inp["_id"]] = titles
        ranked.setdefault("ChainRAG", {})[inp["_id"]] = [
            dedup(sq.get("ranked_titles") or []) for sq in r.get("retrieved_titles_by_subq") or []]

    grit = load_jsonl(ROOT / "grithopper" / "results" / dataset / "retrieval.jsonl")
    out["GRITHopper"] = {r["id"]: set(r.get("retrieved_titles") or []) for r in grit}
    ranked["GRITHopper"] = {r["id"]: [dedup(h.get("ranked_titles") or []) for h in r.get("hops") or []]
                            for r in grit}
    return out, ranked


def score_ranked(ranked: dict[str, list[list[str]]], gold: dict[str, set[str]], k: int = 3) -> dict[str, float]:
    n_hops = r1 = r3 = 0
    p3_sum = 0.0
    for eid, g in gold.items():
        if not g:
            continue
        for hop in ranked.get(eid, []):
            n_hops += 1
            r1 += any(t in g for t in hop[:1])
            r3 += any(t in g for t in hop[:3])
            p3_sum += sum(t in g for t in hop[:k]) / k
    return {"n_hops": n_hops, "recall@1": r1 / n_hops, "recall@3": r3 / n_hops,
            "precision@3": p3_sum / n_hops, "gold_in_top3": p3_sum * k / n_hops}


def score(retrieved: dict[str, set[str]], gold: dict[str, set[str]]) -> dict[str, float]:
    n = 0
    p_sum = c_sum = f_sum = size_sum = 0.0
    hit_tot = ret_tot = gold_tot = 0
    for eid, g in gold.items():
        if not g:
            continue
        r = retrieved.get(eid, set())
        hit = len(r & g)
        p = hit / len(r) if r else 0.0
        c = hit / len(g)
        n += 1
        p_sum += p
        c_sum += c
        f_sum += 2 * p * c / (p + c) if p + c else 0.0
        size_sum += len(r)
        hit_tot, ret_tot, gold_tot = hit_tot + hit, ret_tot + len(r), gold_tot + len(g)
    return {
        "n": n,
        "precision_macro": p_sum / n,
        "precision_micro": hit_tot / ret_tot if ret_tot else 0.0,
        "coverage_macro": c_sum / n,
        "set_f1_macro": f_sum / n,
        "avg_retrieved": size_sum / n,
        "avg_gold": gold_tot / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["musique", "2wiki", "hotpot"])
    ap.add_argument("--d-runs", nargs="*", default=[], metavar="DATASET=RUN_DIR",
                     help="override the D-group run per dataset (default: the 0828 orderinvariant runs, "
                          "which have no per-hop ranking, so D gets no precision@3)")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "evidence_precision_3way.json")
    args = ap.parse_args()
    d_runs = {**D_RUNS, **dict(kv.split("=", 1) for kv in args.d_runs)}

    all_results: dict[str, dict] = {}
    for ds in args.datasets:
        ds_args = argparse.Namespace(
            dataset=ds, split="dev",
            musique_dir=ROOT / "data" / "raw" / "musique",
            twowiki_file=ROOT / "data" / "raw" / "2wikimultihopqa" / "dev.json",
            hotpot_file=ROOT / "data" / "raw" / "hotpotqa" / "hotpot_dev_distractor_v1.json",
        )
        records = load_dataset_records(ds_args)
        sets, ranked = load_method_outputs(ds, records, d_runs[ds])
        ids = set.intersection(*(set(s) for s in sets.values()))
        gold = {eid: gold_titles_for(records[eid]) for eid in ids}
        print(f"\n=== {ds}  (n={len(ids)} questions present in all three) ===")
        print(f"{'method':<12} {'P_macro':>8} {'P_micro':>8} {'cov':>7} {'setF1':>7} {'|ret|':>6} {'|gold|':>6}")
        all_results[ds] = {}
        for m, s in sets.items():
            res = score(s, gold)
            all_results[ds][m] = res
            print(f"{m:<12} {res['precision_macro']:>8.4f} {res['precision_micro']:>8.4f} "
                  f"{res['coverage_macro']:>7.4f} {res['set_f1_macro']:>7.4f} "
                  f"{res['avg_retrieved']:>6.2f} {res['avg_gold']:>6.2f}")
        print(f"\n{'method':<12} {'n_hops':>7} {'R@1':>7} {'R@3':>7} {'P@3':>7} {'gold/top3':>9}")
        for m in sets:
            if m not in ranked:
                print(f"{m:<12} {'(no per-hop ranking in this run -- pass --d-runs)':>40}")
                continue
            rr = score_ranked(ranked[m], gold)
            all_results[ds][m].update(rr)
            print(f"{m:<12} {rr['n_hops']:>7} {rr['recall@1']:>7.4f} {rr['recall@3']:>7.4f} "
                  f"{rr['precision@3']:>7.4f} {rr['gold_in_top3']:>9.2f}")

    args.out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
