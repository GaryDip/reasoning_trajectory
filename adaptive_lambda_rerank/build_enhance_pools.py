#!/usr/bin/env python3
"""
Extra pools for the underrepresented K=3/K=4 buckets, built from LLM-paraphrased
sub-questions in data/decompose/musique/gt/train_nl_enhance.jsonl (its own summary
confirms target_k=[3,4] — it was built for exactly this balancing purpose, just never
consumed anywhere yet).

This is NOT synthetic duplication: each paraphrase is a differently-worded version of
the SAME underlying K-hop question (same gold hop order/answers, from the record's own
question_decomposition — only the sub-question TEXT changes), so running it through
embed_retrieval for real gives a genuinely different BGE top-k pool (different query
embedding -> different emb_score distribution, possibly different candidates), not a
copy of an existing pool. Only verify_pass=True rows are used (~74% of the file).

Written to its OWN file — rerank_listwise_contrastive/build_topk_pools.py and its
existing musique_train_pools.jsonl are never touched. build_training_data.py then reads
both files together via --extra-pools.

Usage:
  python build_enhance_pools.py --split train
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"
TRACES_DIR = PROJECT_ROOT / "traces"
sys.path.insert(0, str(RETRIEVAL_DIR))
sys.path.insert(0, str(TRACES_DIR))

from run_retrieval_exp import (  # noqa: E402
    MUSIQUE_DIR,
    _get_pipeline_helpers,
    build_trace_prefix,
    embed_retrieval,
    load_dataset_records,
)
from run_retrieval_exp_wavefront import init_example_state  # noqa: E402
from trace_evidence import gold_hop_answers_musique  # noqa: E402

METHOD = "pool_builder_enhance"


def load_enhance_rows(path: Path, target_k: set[int]) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if not d.get("verify_pass"):
                continue
            if int(d["K"]) not in target_k:
                continue
            rows.append(d)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["musique"], default="musique")
    ap.add_argument("--split", choices=["train"], default="train",
                     help="Only train has an enhance file — dev's own K distribution is "
                          "already less skewed and should stay untouched as a clean holdout.")
    ap.add_argument("--musique-dir", type=Path, default=MUSIQUE_DIR)
    ap.add_argument("--enhance-file", type=Path,
                     default=PROJECT_ROOT / "data" / "decompose" / "musique" / "gt" / "train_nl_enhance.jsonl")
    ap.add_argument("--target-k", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--retrieve-k", type=int, default=20)
    ap.add_argument("--cos-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.enhance_file.is_file():
        sys.exit(f"Missing enhance file: {args.enhance_file}")
    out_path = args.out or (HERE / "data" / f"{args.dataset}_{args.split}_pools_enhance.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds_args = argparse.Namespace(dataset=args.dataset, musique_dir=args.musique_dir, split=args.split)
    records = load_dataset_records(ds_args)

    enhance_rows = load_enhance_rows(args.enhance_file, set(args.target_k))
    print(f"Enhance rows (verify_pass, K in {args.target_k}): {len(enhance_rows)} / "
          f"file has target_k={args.target_k}")

    (expand_hop_template, *_rest) = _get_pipeline_helpers()

    examples = []
    gold_answers_by_id: dict[str, list[str]] = {}
    variant_by_id: dict[str, int] = {}
    for row_d in enhance_rows:
        source_id = row_d["source_id"]
        record = records.get(source_id)
        if record is None:
            continue
        sub_qs = [str(s).strip() for s in row_d.get("sub_questions") or [] if str(s).strip()]
        if len(sub_qs) != int(row_d["K"]):
            continue  # paraphrase dropped/garbled a hop — skip, don't guess alignment
        examples.append(init_example_state(source_id, record, sub_qs, [METHOD], []))
        gold_answers_by_id[source_id] = gold_hop_answers_musique(record)
        variant_by_id[source_id] = int(row_d.get("variant", 0))

    print(f"Usable paraphrased examples: {len(examples)}")
    max_hops = max((ex.K for ex in examples), default=0)
    n_pools = 0
    n_gold_in_pool = 0

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    with out_path.open("w", encoding="utf-8") as f:
        for hop_j in range(1, max_hops + 1):
            print(f"=== hop {hop_j}/{max_hops} ===", flush=True)
            ex_iter = examples
            if tqdm is not None:
                ex_iter = tqdm(examples, desc=f"hop{hop_j} enhance-pools", unit="ex",
                                file=sys.stderr, dynamic_ncols=True)
            for ex in ex_iter:
                if hop_j > ex.K:
                    continue
                hop_row = ex.hop_results[hop_j - 1]
                gold_pi = hop_row.get("gold_para_idx")
                if not (hop_row.get("has_gold_hop") and gold_pi is not None):
                    continue

                gold_answers = gold_answers_by_id[ex.eid]
                prior = gold_answers[: hop_j - 1]
                raw_sq = ex.sub_questions[hop_j - 1]
                expanded_q = expand_hop_template(raw_sq, prior)

                hop_steps = []
                ok = True
                for gj in range(hop_j - 1):
                    prior_pi = ex.hop_results[gj].get("gold_para_idx")
                    prior_para = next((p for p in ex.paragraphs if int(p.get("idx", -1)) == prior_pi), None)
                    if prior_para is None:
                        ok = False
                        break
                    hop_steps.append((ex.sub_questions[gj], (prior_para.get("paragraph_text") or "").strip()))
                if not ok:
                    continue
                prefix_before = build_trace_prefix(ex.q_main, hop_steps)

                candidates_all = embed_retrieval(expanded_q, ex.paragraphs, args.retrieve_k, args.cos_model)
                if not candidates_all:
                    continue

                gold_pos = None
                cand_rows = []
                for pos, (p, score) in enumerate(candidates_all):
                    is_gold = int(p.get("idx", -1)) == int(gold_pi)
                    if is_gold:
                        gold_pos = pos
                    cand_rows.append({
                        "idx": int(p.get("idx", -1)),
                        "title": p.get("title") or "",
                        "paragraph_text": p.get("paragraph_text") or "",
                        "emb_score": round(float(score), 6),
                        "is_gold": is_gold,
                    })

                n_pools += 1
                if gold_pos is not None:
                    n_gold_in_pool += 1

                variant = variant_by_id[ex.eid]
                f.write(json.dumps({
                    "pool_id": f"{ex.eid}__K{ex.K}__j{hop_j - 1}__enh{variant}",
                    "source_id": ex.eid, "K": ex.K, "j": hop_j - 1,
                    "prefix_before": prefix_before, "expanded_q": expanded_q,
                    "gold_pos": gold_pos, "candidates": cand_rows,
                }, ensure_ascii=False) + "\n")

    print(f"\nWrote {n_pools} enhance pools -> {out_path}")
    if n_pools:
        print(f"Gold-in-top-{args.retrieve_k} coverage: {n_gold_in_pool}/{n_pools} "
              f"({n_gold_in_pool / n_pools:.4f})")


if __name__ == "__main__":
    main()
