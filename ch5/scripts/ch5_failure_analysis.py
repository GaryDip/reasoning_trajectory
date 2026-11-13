#!/usr/bin/env python3
"""
Chapter 5 failure analysis (CH5_PLAN.md task 1). CPU only.

Everything is recomputed from the per-question case logs of the uniform-layer-15 runs, so the
numbers are on the same footing as Chapter 4. Answer normalization is imported from MuSiQue's
official `metrics.answer`, the same module `judge_answer_official` uses, so "the prediction equals
a hop answer" means equal under exactly the EM normalization.

  1a  every question into one of four mutually exclusive buckets (answered / wrong hop count /
      retrieval error / retrieved everything yet answered wrongly)
  1b  NA cascades: a hop answers "NA" and a later sub-question refers to it
  1c  the final answer is one of the intermediate hop answers (the comparison was never performed)
  1d  within bucket D, how often the prediction only differs in wording
  1e  what the gate changed, for the questions each system gets right and the other does not
  1f  bucket breakdown by annotated hop count (MuSiQue)
  plus k_match per dataset, recomputed on the new runs (CH5_PLAN task 4.2)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent          # ch5/scripts/
ROOT = HERE.parents[1]                          # the repo root
RETRIEVAL = ROOT / "retrieval"                  # where the run directories live
# MuSiQue's official scorer: `judge_answer_official` in the base pipeline imports exactly this
# module, so "equal after normalization" here means equal under the EM used in Chapters 4 and 5.
# answer.py does `from metrics.metric import Metric`, so its own package dir has to be on the path.
_MQ = ROOT.parent / "multihop_trajectory" / "decompose-musique" / "musique"
sys.path.insert(0, str(_MQ))
sys.path.insert(0, str(_MQ.parent))
from musique.metrics.answer import (  # noqa: E402
    compute_f1, metric_max_over_ground_truths, normalize_answer,
)
sys.path.insert(0, str(RETRIEVAL))
from goldmatch import hop_hits, order_invariant_applies, selected_a_gold_twice  # noqa: E402

RUNS = {
    "main": {"musique": "20261008_071635_l15_musique_lambda0p60_dev",
             "hotpot": "20261008_074208_l15_hotpot_lambda0p60_dev",
             "2wiki": "20261008_085006_l15_2wiki_lambda0p60_dev"},
    "control": {"musique": "20261003_070503_nogate_musique_lambda0_dev",
                "2wiki": "20261003_070301_nogate_2wiki_lambda0_dev",
                "hotpot": "20261003_070253_nogate_hotpot_lambda0_dev"},
    "main_default_prompt": {"musique": "20261010_074742_l15_lambda0p60_musique_defaultprompt",
                            "2wiki": "20261010_075104_l15_lambda0p60_2wiki_defaultprompt",
                            "hotpot": "20261010_080204_l15_lambda0p60_hotpot_defaultprompt"},
}
DATASETS = ["musique", "2wiki", "hotpot"]


def load_cases(run_dir: str) -> dict[str, dict]:
    d = RETRIEVAL / "results" / run_dir
    f = next(d.glob("retrieval_cases_*.jsonl"))
    return {json.loads(l)["id"]: json.loads(l) for l in f.open() if l.strip()}


def load_meta(ds: str):
    """gold answers (+ aliases), question type, and the raw BART sub-questions."""
    golds, qtype = {}, {}
    if ds == "musique":
        for l in (ROOT / "data/raw/musique/musique_ans_v1.0_dev.jsonl").open():
            r = json.loads(l)
            golds[r["id"]] = [r["answer"]] + [a for a in (r.get("answer_aliases") or []) if a]
    elif ds == "2wiki":
        for r in json.loads((ROOT / "data/raw/2wikimultihopqa/dev.json").read_text()):
            golds[str(r["_id"])] = [r["answer"]]
            qtype[str(r["_id"])] = r.get("type", "?")
    else:
        for r in json.loads((ROOT / "data/raw/hotpotqa/hotpot_dev_distractor_v1.json").read_text()):
            golds[str(r["_id"])] = [r["answer"]]
            qtype[str(r["_id"])] = r.get("type", "?")
    subqs = {}
    for l in (ROOT / f"data/decompose/{ds}/bart/dev_nl.jsonl").open():
        r = json.loads(l)
        subqs[str(r["id"])] = r.get("sub_questions") or []
    return golds, qtype, subqs


def hop_ok(c: dict, j: int, oi: bool = False) -> bool:
    """Was the passage selected at hop j the right one?

    Both protocols live in `retrieval/goldmatch.py`; order-invariant means "a gold passage that
    no earlier hop already claimed", so it needs the whole prefix, not just hop j.
    """
    hits = hop_hits(c["para_ids"], c["gold_idxs"], oi)
    return j < len(hits) and hits[j]


def bucket(c: dict, oi: bool = False) -> str:
    if c["em"] == 1:
        return "A_answered"
    if len(c["para_ids"]) != len(c["gold_idxs"]):
        return "B_wrong_hop_count"
    got_everything = (sorted(c["para_ids"]) == sorted(c["gold_idxs"]) if oi
                      else all(p == g for p, g in zip(c["para_ids"], c["gold_idxs"])))
    return "D_retrieved_all_wrong_answer" if got_everything else "C_retrieval_error"


REF = re.compile(r"\[Answer\s*(\d+)\]", re.I)


def na_cascade(c: dict, subqs: list[str]) -> bool:
    """A hop answered NA and some LATER sub-question refers to that hop."""
    prior = c.get("prior") or []
    for j, ans in enumerate(prior[:-1]):                 # the last hop has no consumer
        if normalize_answer(str(ans)) != "na":
            continue
        hop = j + 1                                      # [Answer i] is 1-indexed
        for later in subqs[j + 1:]:
            if any(int(m) == hop for m in REF.findall(later)):
                return True
    return False


def answer_is_an_intermediate_hop_answer(c: dict) -> bool:
    """The final answer repeats a NON-FINAL hop answer.

    Matching the LAST hop's answer is the expected behaviour of a bridge question, so it is not a
    failure mode; the failure the comparison instruction targets is answering with one of the
    values that were only meant to be compared, i.e. an earlier hop's answer.
    """
    fa = normalize_answer(str(c.get("final_answer") or ""))
    prior = c.get("prior") or []
    if not fa or fa == "na" or len(prior) < 2:
        return False                                     # "NA" is an abstention, not a comparison
    return any(normalize_answer(str(p)) == fa for p in prior[:-1])


def wording_level(c: dict, golds: list[str]) -> str:
    """How confidently is the prediction only a different wording of the gold answer?

    Returns "lower" (conservative), "upper" (the permissive rule only) or "" (not wording).

    The permissive rule is substring containment in either direction, or token F1 >= 0.5. It
    also fires on predictions that answer a DIFFERENT question than the one asked -- e.g.
    "Ondine" against the gold "for Best Performance by a Leading Actress in a Play in Ondine",
    where the play's name is contained in, but is not, the award asked for. So it can only be
    read as an upper bound. The conservative rule asks for high two-sided token overlap
    (F1 >= 0.8), which none of the observed mislabelled cases reaches, and is therefore a lower
    bound. The truth is in between; `--audit-sample` dumps a sample for manual calibration.
    """
    raw = str(c.get("final_answer") or "")
    pred = normalize_answer(raw)
    if not pred or pred == "na":                         # an abstention is not a wording issue
        return ""
    f1 = metric_max_over_ground_truths(compute_f1, raw, golds)
    if f1 >= 0.8:
        return "lower"
    if f1 >= 0.5:
        return "upper"
    for g in golds:
        gn = normalize_answer(str(g))
        if gn and (pred in gn or gn in pred):
            return "upper"
    return ""


def first_divergent_hop(a: dict, b: dict):
    for j, (pa, pb) in enumerate(zip(a["para_ids"], b["para_ids"])):
        if pa != pb:
            return j
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "ch5/results/ch5_failure_analysis.json")
    ap.add_argument("--n-examples", type=int, default=10)
    ap.add_argument("--audit-sample", type=int, default=50,
                    help="per dataset, how many bucket-D wording cases to dump for manual review")
    ap.add_argument("--audit-out", type=Path,
                    default=ROOT / "ch5/results/bucketD_wording_audit_sample.jsonl")
    args = ap.parse_args()
    rng = random.Random(0)
    out: dict = {}
    audit: list[dict] = []

    for ds in DATASETS:
        golds, qtype, subqs = load_meta(ds)
        main_c = load_cases(RUNS["main"][ds])
        ctrl_c = load_cases(RUNS["control"][ds])
        dflt_c = load_cases(RUNS["main_default_prompt"][ds])
        n = len(main_c)
        res: dict = {"n_questions": n, "run": RUNS["main"][ds]}

        # ---- 1a ----  computed twice: positionally, and ignoring the order of the gold
        # passages wherever that order is an annotation artefact (see `order_invariant`).
        OI = {i: order_invariant_applies(ds, qtype.get(i, "")) for i in main_c}
        res["order_invariant_questions"] = {
            "n": sum(OI.values()), "share": round(sum(OI.values()) / n, 4),
            "rule": ("all questions" if ds == "hotpot" else
                     "type in (comparison, bridge_comparison)" if ds == "2wiki" else "none"),
            "n_selected_a_gold_twice": sum(
                1 for i, c in main_c.items()
                if OI[i] and selected_a_gold_twice(c["para_ids"], c["gold_idxs"]))}
        buckets = {i: bucket(c) for i, c in main_c.items()}
        buckets_oi = {i: bucket(c, OI[i]) for i, c in main_c.items()}

        def bucket_shares(bk):
            cnt = Counter(bk.values())
            return {k: {"n": cnt.get(k, 0), "share": round(cnt.get(k, 0) / n, 4)}
                    for k in ("A_answered", "B_wrong_hop_count", "C_retrieval_error", "D_retrieved_all_wrong_answer")}
        res["1a_buckets"] = bucket_shares(buckets)
        res["1a_buckets_order_invariant"] = bucket_shares(buckets_oi)
        # task 4.2: k_match recomputed on this run
        km = sum(1 for c in main_c.values() if len(c["para_ids"]) == len(c["gold_idxs"]))
        res["k_match_this_run"] = round(km / n, 4)

        # ---- 1b ----
        na_ids = {i for i, c in main_c.items() if na_cascade(c, subqs.get(i, []))}

        def hop_acc(oi_map):
            hit_a = tot_a = hit_r = tot_r = 0
            for i, c in main_c.items():
                prior = c.get("prior") or []
                na_pos = [j for j, a in enumerate(prior[:-1]) if normalize_answer(str(a)) == "na"]
                hits = hop_hits(c["para_ids"], c["gold_idxs"], oi_map[i])
                for j, ok in enumerate(hits):
                    if na_pos and j > min(na_pos):
                        tot_a += 1; hit_a += ok
                    elif not na_pos:
                        tot_r += 1; hit_r += ok
            return {"hop_accuracy_after_na": round(hit_a / max(tot_a, 1), 4),
                    "n_hops_after_na": tot_a,
                    "hop_accuracy_in_questions_without_na": round(hit_r / max(tot_r, 1), 4),
                    "n_hops_reference": tot_r}

        # An NA cascade can land in any bucket, so both directions are reported explicitly:
        #   share_of_bucket = NA cascades in this bucket / all questions in this bucket
        #   share_of_na     = NA cascades in this bucket / all NA cascades
        def na_crosstab(bk):
            t = {}
            for b in ("A_answered", "B_wrong_hop_count", "C_retrieval_error", "D_retrieved_all_wrong_answer"):
                ids = {i for i, v in bk.items() if v == b}
                k = len(ids & na_ids)
                t[b] = {"n_na_cascade": k, "n_bucket": len(ids),
                        "share_of_bucket": round(k / max(len(ids), 1), 4),
                        "share_of_na": round(k / max(len(na_ids), 1), 4)}
            return t
        res["1b_na_cascade"] = {
            "n": len(na_ids), "share_of_all": round(len(na_ids) / n, 4),
            "by_bucket": na_crosstab(buckets),
            "by_bucket_order_invariant": na_crosstab(buckets_oi),
            "positional": hop_acc({i: False for i in main_c}),
            "order_invariant": hop_acc(OI)}

        # ---- 1c / 1d ---- bucket D split into three MUTUALLY EXCLUSIVE causes, in the order
        # used by the 0817 case study (wording first, then the repeated hop answer, then the rest),
        # so the shares are comparable with the 77 / 8 / 15 reported there.
        D_ids = [i for i, b in buckets_oi.items() if b == "D_retrieved_all_wrong_answer"]
        D_ids_positional = [i for i, b in buckets.items() if b == "D_retrieved_all_wrong_answer"]

        def split_D(cases, ids):
            """Four mutually exclusive causes, in the order of the 0817 case study."""
            ids = [i for i in ids if i in cases]
            lv = {i: wording_level(cases[i], golds.get(i, [])) for i in ids}
            na = [i for i in ids
                  if normalize_answer(str(cases[i].get("final_answer") or "")) == "na"]
            rest = [i for i in ids if i not in set(na)]
            w = [i for i in rest if lv[i]]
            w_lo = [i for i in w if lv[i] == "lower"]
            rest = [i for i in rest if i not in set(w)]
            hop = [i for i in rest if answer_is_an_intermediate_hop_answer(cases[i])]
            other = [i for i in rest if i not in set(hop)]
            return ids, na, w, w_lo, hop, other

        def hop_answer_rate(cases, ids):
            allids, _na, _w, _wlo, hop, _o = split_D(cases, ids)
            if not allids:
                return None, 0
            return round(len(hop) / len(allids), 4), len(allids)
        rate_hint, nD = hop_answer_rate(main_c, D_ids)
        # The default-prompt arm is a replay of the SAME chains, so its case log stores only the
        # regenerated answer (no para_ids / prior). Retrieval is bit-for-bit identical, so the
        # per-hop answers come from the main run and the bucket is recomputed with the replayed EM:
        # a question is in D iff every hop was retrieved correctly and the answer is still wrong.
        merged_dflt = {}
        for i, c in dflt_c.items():
            if i not in main_c:
                continue
            m = main_c[i]
            merged_dflt[i] = {**m, "em": c["em"], "f1": c["f1"],
                              "final_answer": c.get("final_answer")}
        d_buckets = {i: bucket(c, OI.get(i, False)) for i, c in merged_dflt.items()}
        D_ids_dflt = [i for i, b in d_buckets.items() if b == "D_retrieved_all_wrong_answer"]
        rate_dflt, nD_dflt = hop_answer_rate(merged_dflt, D_ids_dflt)
        res["1c_answer_is_hop_answer"] = {
            "with_comparison_hint": {"rate": rate_hint, "n_in_bucket_D": nD},
            "with_default_prompt": {"rate": rate_dflt, "n_in_bucket_D": nD_dflt}}
        if qtype:
            by_t = {}
            for t in sorted(set(qtype.values())):
                ids_t = [i for i in D_ids if qtype.get(i) == t]
                r, m = hop_answer_rate(main_c, ids_t)
                by_t[t] = {"rate": r, "n_in_bucket_D": m}
            res["1c_by_question_type"] = by_t

        # ---- 1d ----
        def breakdown(ids):
            allD, na_ids_D, word_ids, word_lo, hop_ids, other_ids = split_D(main_c, ids)
            d = max(len(allD), 1)
            return {"n_in_bucket_D": len(allD),
                    "abstained_na": {"n": len(na_ids_D),
                                     "share": round(len(na_ids_D) / d, 4)},
                    "only_wording_upper_bound": {"n": len(word_ids),
                                                 "share": round(len(word_ids) / d, 4)},
                    "only_wording_lower_bound": {"n": len(word_lo),
                                                 "share": round(len(word_lo) / d, 4)},
                    "repeats_an_intermediate_hop_answer": {
                        "n": len(hop_ids), "share": round(len(hop_ids) / d, 4)},
                    "other": {"n": len(other_ids),
                              "share": round(len(other_ids) / d, 4)}}, word_ids
        bd, word_ids = breakdown(D_ids)
        samples = rng.sample(word_ids, min(args.n_examples, len(word_ids))) if word_ids else []
        bd["wording_examples"] = [{"id": i, "prediction": main_c[i].get("final_answer"),
                                   "gold": golds.get(i, [])} for i in samples]
        res["1d_bucket_D_breakdown"] = bd
        res["1d_bucket_D_breakdown_positional"] = breakdown(D_ids_positional)[0]
        # a sample for the manual calibration of the wording rule (CH5 review point 2)
        for i in (rng.sample(word_ids, min(args.audit_sample, len(word_ids)))
                  if word_ids else []):
            audit.append({"dataset": ds, "id": i,
                          "question": main_c[i].get("question"),
                          "prediction": main_c[i].get("final_answer"),
                          "gold": golds.get(i, []),
                          "rule": wording_level(main_c[i], golds.get(i, [])),
                          "f1": round(metric_max_over_ground_truths(
                              compute_f1, str(main_c[i].get("final_answer") or ""),
                              golds.get(i, [])), 3),
                          "verdict_fill_in": ""})

        # ---- 1e ----
        def diag(ids, oi_map):
            d = Counter()
            for i in ids:
                a, b = main_c[i], ctrl_c[i]
                j = first_divergent_hop(a, b)
                if j is None:
                    d["iv_same_selection_generation_only"] += 1
                    continue
                oi = oi_map[i]
                a_ok, b_ok = hop_ok(a, j, oi), hop_ok(b, j, oi)
                if b_ok and not a_ok:
                    d["i_gate_demoted_a_correct_top1"] += 1
                elif a_ok and not b_ok:
                    d["ii_gate_fixed_an_error"] += 1
                else:
                    d["iii_neither_is_gold"] += 1
            return dict(d)
        shared = set(main_c) & set(ctrl_c)
        gate_only = [i for i in shared if main_c[i]["em"] == 1 and ctrl_c[i]["em"] == 0]
        ctrl_only = [i for i in shared if main_c[i]["em"] == 0 and ctrl_c[i]["em"] == 1]
        flat = {i: False for i in main_c}
        res["1e_paired_with_control"] = {
            "gate_only_correct": {"n": len(gate_only), **diag(gate_only, flat)},
            "control_only_correct": {"n": len(ctrl_only), **diag(ctrl_only, flat)}}
        res["1e_paired_with_control_order_invariant"] = {
            "gate_only_correct": {"n": len(gate_only), **diag(gate_only, OI)},
            "control_only_correct": {"n": len(ctrl_only), **diag(ctrl_only, OI)}}

        # ---- 1f ----
        if ds == "musique":
            byK = defaultdict(Counter)
            for i, c in main_c.items():
                byK[len(c["gold_idxs"])][buckets_oi[i]] += 1
            res["1f_by_annotated_hops"] = {
                str(k): {b: {"n": v.get(b, 0), "share": round(v.get(b, 0) / sum(v.values()), 4)}
                         for b in ("A_answered", "B_wrong_hop_count", "C_retrieval_error",
                                   "D_retrieved_all_wrong_answer")}
                for k, v in sorted(byK.items())}
        out[ds] = res

        print(f"=== {ds}  n={n}  k_match={res['k_match_this_run']:.4f}  "
              f"order-invariant on {res['order_invariant_questions']['n']} questions")
        for k in ("1a_buckets", "1a_buckets_order_invariant"):
            print(f"  {k}")
            for kk, v in res[k].items():
                print(f"    {kk:30s} {v['n']:6d}  {v['share']:.2%}")
        nb = res["1b_na_cascade"]
        print(f"    NA cascade {nb['n']} ({nb['share_of_all']:.2%}); of those "
              f"{nb['by_bucket_order_invariant']['C_retrieval_error']['share_of_na']:.1%} are in C, "
              f"and they are {nb['by_bucket_order_invariant']['C_retrieval_error']['share_of_bucket']:.1%} of C")
        for k in ("positional", "order_invariant"):
            print(f"    hop acc after NA ({k}) {nb[k]['hop_accuracy_after_na']:.4f} vs "
                  f"{nb[k]['hop_accuracy_in_questions_without_na']:.4f}")
        print(f"    answer==a hop answer (bucket D): hint {rate_hint} (n={nD}) | "
              f"default {rate_dflt} (n={nD_dflt})")
        bd = res["1d_bucket_D_breakdown"]
        print(f"    bucket D split: NA {bd['abstained_na']['share']:.1%} | wording "
              f"{bd['only_wording_lower_bound']['share']:.1%}-{bd['only_wording_upper_bound']['share']:.1%} | "
              f"repeats a hop answer {bd['repeats_an_intermediate_hop_answer']['share']:.1%} | "
              f"other {bd['other']['share']:.1%}")
        for k in ("1e_paired_with_control", "1e_paired_with_control_order_invariant"):
            print(f"    {k}: gate-only {res[k]['gate_only_correct']}")
            print(f"    {' ' * len(k)}  ctrl-only {res[k]['control_only_correct']}")
        print()

    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"-> {args.out}")
    if audit:
        args.audit_out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                          for r in audit))
        print(f"-> {args.audit_out}  ({len(audit)} cases to review by hand)")


if __name__ == "__main__":
    main()
