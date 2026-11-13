#!/usr/bin/env python3
"""
Experiment 4, step 1 (CPU) -- rebuild reasoning chains that contain NATURAL retrieval errors.

Experiments 1-3 all use *constructed* errors (one hop's passage swapped on purpose). Experiment 4
asks whether the same signal shows up when the pipeline makes its own mistakes, so it needs the
chains the system actually produced. Source: the λ=0 (no-gate) dev run, whose per-question case
log stores the selected passage ids and the per-hop short answers:
    retrieval/results/20261003_070503_nogate_musique_lambda0_dev/

Only MuSiQue is usable, because only it annotates the supporting paragraph of every hop, which is
what makes a hop "wrong" decidable. Questions whose BART decomposition has a different number of
hops than the annotation are dropped (hops cannot be aligned): 2,022 of 2,417 survive (83.7%).

A chain is written with trace_type="correct" ON PURPOSE. The extractor truncates chains labelled
"error" at the injected hop (n_take = wrong_hop + 1), but here we need every transition --
including the ones after the first mistake -- so the chain must not be truncated. The real wrong
hops are carried in `wrong_hops`, which the extractor copies into the npz untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "traces"))
from trace_format import assemble_trace  # noqa: E402

RUN = ROOT / "retrieval/results/20261003_070503_nogate_musique_lambda0_dev"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "results/exp4_natural_traces.jsonl")
    args = ap.parse_args()

    raw = {}
    for l in (ROOT / "data/raw/musique/musique_ans_v1.0_dev.jsonl").open():
        r = json.loads(l)
        raw[r["id"]] = r
    dec = {}
    for l in (ROOT / "data/decompose/musique/bart/dev_nl.jsonl").open():
        r = json.loads(l)
        dec[r["id"]] = r.get("sub_questions") or []
    cases = [json.loads(l) for l in next(args.run.glob("retrieval_cases_*.jsonl")).open() if l.strip()]
    print(f"cases={len(cases)}  raw={len(raw)}  decompose={len(dec)}")

    kept, dropped, stats = [], 0, {"n_wrong": {}}
    for c in cases:
        ex = raw.get(c["id"])
        sub = dec.get(c["id"])
        if ex is None or not sub:
            dropped += 1
            continue
        gold_idx = [d["paragraph_support_idx"] for d in ex["question_decomposition"]]
        if len(sub) != len(gold_idx):
            dropped += 1                      # hop counts not alignable
            continue
        pids = c.get("para_ids") or []
        if len(pids) != len(gold_idx):
            dropped += 1
            continue
        paras = {p["idx"]: p for p in ex["paragraphs"]}
        if any(p not in paras for p in pids):
            dropped += 1
            continue
        ev = [(paras[p].get("paragraph_text") or "").strip() for p in pids]
        wrong = [j + 1 for j, p in enumerate(pids) if p != gold_idx[j]]
        kept.append({
            "id": f"nat__{c['id']}",
            "dataset": "musique", "split": "dev",
            "trace_type": "correct",          # see module docstring: keeps every prefix
            "wrong_hops": wrong,
            "first_wrong_hop": wrong[0] if wrong else None,
            "K": len(gold_idx),
            "reasoning_trace": assemble_trace(c["question"], sub, ev,
                                              c.get("final_answer") or "NA"),
            "source_id": c["id"],
            "selected_para_ids": pids, "gold_para_ids": gold_idx,
        })
        stats["n_wrong"][str(len(wrong))] = stats["n_wrong"].get(str(len(wrong)), 0) + 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    byK = {}
    for r in kept:
        byK.setdefault(r["K"], [0, 0])
        byK[r["K"]][0] += 1
        byK[r["K"]][1] += 1 if r["wrong_hops"] else 0
    print(f"kept {len(kept)}  dropped {dropped}")
    print("  by K (total / with >=1 wrong hop):",
          {k: f"{v[0]} / {v[1]}" for k, v in sorted(byK.items())})
    print("  chains by number of wrong hops:", dict(sorted(stats["n_wrong"].items())))
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
