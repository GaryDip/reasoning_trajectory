#!/usr/bin/env python3
"""
Build gold + cosine counterfactual traces with per-(K, hop) balancing.

Positive: GT NL sub_questions + all gold evidence (1 per example).
Negative: hop h wrong from cosine top-k non-gold; up to max_wrong_per_hop variants.

Balancing:
  For each K, target pos count = all examples with that K.
  For each (K, hop h), target neg count = pos_count[K] * balance_ratio
  Rare buckets (e.g. K=4) are upsampled by repeating traces with different
  cosine wrong passages until target is met (or source exhausted).

Output (one pass, three files):
  traces/gold/{dataset}/{split}.jsonl           — all positives
  traces/counterfactual/{dataset}/{split}.jsonl — all negatives (unbalanced)
  traces/merged/{dataset}/{split}.jsonl         — pos + balanced neg (for hidden states)
  traces/merged/{dataset}/{split}.stats.json

Usage:
  python construct_balanced_traces.py --dataset musique --split train
  python construct_balanced_traces.py --balance-ratio 1.0 --max-wrong-per-hop 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dataset_loaders import (
    get_decompose_enhance_path,
    get_decompose_path,
    load_decompose_bundle,
    load_raw_index,
)
from trace_evidence import (
    CosinePoolRanker,
    ExampleRetrieval,
    build_hop_queries,
    gold_evidence_musique,
    gold_hop_answers_musique,
    resolve_device,
)
from trace_format import assemble_trace

PROJECT_ROOT = HERE.parent
GOLD_ROOT = PROJECT_ROOT / "traces" / "gold"
CF_ROOT = PROJECT_ROOT / "traces" / "counterfactual"
MERGED_ROOT = PROJECT_ROOT / "traces" / "merged"


def align_subqs_and_gold(sub_qs, gold_texts, hop_answers):
    k = min(len(sub_qs), len(gold_texts), len(hop_answers))
    if k < 1:
        return [], [], [], 0
    return sub_qs[:k], gold_texts[:k], hop_answers[:k], k


def write_row(out_f, row: dict) -> None:
    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_eligible(
    decomp: dict[str, list[str]],
    raw_id_map: dict[str, str],
    raw_index: dict,
    limit: int = 0,
) -> int:
    n = 0
    for dec_id in decomp:
        raw_id = raw_id_map.get(dec_id, dec_id)
        if raw_id not in raw_index:
            continue
        n += 1
        if limit and n >= limit:
            return n
    return n


def iter_decompose_jobs(
    decomp: dict[str, list[str]],
    raw_id_map: dict[str, str],
    raw_index: dict,
):
    for dec_id, sub_qs in decomp.items():
        raw_id = raw_id_map.get(dec_id, dec_id)
        record = raw_index.get(raw_id)
        if record is None:
            continue
        yield dec_id, raw_id, record, sub_qs


def process_example_batch(
    batch: list[dict],
    ranker: CosinePoolRanker,
    *,
    args,
    pos_by_k: dict,
    neg_pool: dict,
) -> int:
    """Run batched cosine retrieval and append pos/neg rows. Returns n processed."""
    retr: list[ExampleRetrieval] = []
    for item in batch:
        ex = ExampleRetrieval(
            rid=item["rid"],
            k=item["k"],
            paragraphs=item["paragraphs"],
            gold_idxs=item["gold_idxs"],
            queries=build_hop_queries(item["sub_qs"], item["hop_answers"]),
        )
        retr.append(ex)

    hop_wrongs_batch = ranker.process_batch(
        retr, topk=args.cos_topk, max_wrong=args.max_wrong_per_hop
    )

    for item, hop_wrongs in zip(batch, hop_wrongs_batch):
        rid = item["rid"]
        k = item["k"]
        pos_by_k[k].append(item["pos_row"])
        for h, wrong_cands in enumerate(hop_wrongs, start=1):
            for vi, (wrong_text, wrong_pidx, score) in enumerate(wrong_cands):
                ev_texts = list(item["gold_texts"])
                ev_texts[h - 1] = wrong_text
                neg_pool[(k, h)].append({
                    "id": f"{rid}__wrong_h{h}__cos{vi}",
                    "dataset": args.dataset,
                    "split": args.split,
                    "trace_type": "error",
                    "wrong_hops": [h],
                    "first_wrong_hop": h,
                    "K": k,
                    "reasoning_trace": assemble_trace(
                        item["q"], item["sub_qs"], ev_texts, item["ans"]
                    ),
                    "source_id": item["raw_id"],
                    "decompose_id": rid,
                    "variant": f"wrong_h{h}_cos{vi}",
                    "wrong_mode": "cosine_topk_non_gold",
                    "wrong_hop": h,
                    "wrong_paragraph_idx": wrong_pidx,
                    "cosine_score": round(score, 4),
                })
    return len(batch)


def main() -> None:
    ap = argparse.ArgumentParser(description="Gold + balanced counterfactual trace builder.")
    ap.add_argument("--dataset", choices=("musique",), default="musique")
    ap.add_argument("--split", choices=("train", "dev"), default="train")
    ap.add_argument("--decompose-file", type=Path, default=None)
    ap.add_argument("--enhance-decompose-file", type=Path, default=None)
    ap.add_argument(
        "--no-enhance",
        action="store_true",
        help="Do not merge train_nl_enhance.jsonl even if present",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--cos-topk", type=int, default=10)
    ap.add_argument("--max-wrong-per-hop", type=int, default=3)
    ap.add_argument("--cos-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--balance-ratio", type=float, default=1.0,
                    help="Target neg count per (K,h) = pos_count[K] * ratio")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-examples", type=int, default=0)
    ap.add_argument("--device", default=None, help="cuda / cuda:0 / cpu (default: auto)")
    ap.add_argument("--encode-batch-size", type=int, default=64,
                    help="SentenceTransformer encode batch size")
    ap.add_argument("--example-batch-size", type=int, default=32,
                    help="Number of examples per GPU encode batch")
    args = ap.parse_args()

    decompose_path = args.decompose_file or get_decompose_path(
        args.dataset, args.split, mode="gt"
    )
    if not decompose_path.is_file():
        sys.exit(f"Missing decompose file: {decompose_path}")

    enhance_path = None
    if not args.no_enhance and args.split == "train":
        enhance_path = args.enhance_decompose_file or get_decompose_enhance_path(
            args.dataset, args.split
        )

    decomp, raw_id_map = load_decompose_bundle(
        decompose_path,
        enhance_path=enhance_path,
        include_enhance=not args.no_enhance,
    )
    raw_index = load_raw_index(args.dataset, args.split)
    n_base = sum(1 for d in decomp if raw_id_map.get(d, d) == d)
    n_enh = len(decomp) - n_base
    print(f"Decompose entries: {len(decomp)} (base ~{n_base}, enhance ~{n_enh})")
    out_merged = args.out or (MERGED_ROOT / args.dataset / f"{args.split}.jsonl")
    out_gold = GOLD_ROOT / args.dataset / f"{args.split}.jsonl"
    out_cf = CF_ROOT / args.dataset / f"{args.split}.jsonl"
    stats_path = out_merged.with_suffix(".stats.json")
    for p in (out_merged, out_gold, out_cf):
        p.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    device = resolve_device(args.device)
    ranker = CosinePoolRanker(
        args.cos_model,
        device=device,
        encode_batch_size=args.encode_batch_size,
    )
    print(f"Cosine ranker: {args.cos_model} on {device}, "
          f"encode_batch={args.encode_batch_size}, example_batch={args.example_batch_size}")

    pos_by_k: dict[int, list[dict]] = defaultdict(list)
    neg_pool: dict[tuple[int, int], list[dict]] = defaultdict(list)

    total = args.limit_examples or count_eligible(
        decomp, raw_id_map, raw_index, limit=args.limit_examples
    )
    n_ex = 0
    pending: list[dict] = []
    pbar = tqdm(total=total, desc=f"{args.dataset}/{args.split}", unit="ex")
    for dec_id, raw_id, record, sub_qs_in in iter_decompose_jobs(
        decomp, raw_id_map, raw_index
    ):
        sub_qs = sub_qs_in
        gold_texts, gold_idxs_list = gold_evidence_musique(record)
        hop_answers = gold_hop_answers_musique(record)
        sub_qs, gold_texts, hop_answers, k = align_subqs_and_gold(
            sub_qs, gold_texts, hop_answers
        )
        if k < 1:
            continue

        gold_idxs = set(gold_idxs_list[:k])
        paragraphs = list(record.get("paragraphs") or [])
        q = str(record.get("question", "")).strip()
        ans = str(record.get("answer", "")).strip()
        is_enhance = dec_id != raw_id
        variant = "enhance" if is_enhance else "gold"

        pending.append({
            "rid": dec_id,
            "raw_id": raw_id,
            "k": k,
            "sub_qs": sub_qs,
            "gold_texts": gold_texts,
            "hop_answers": hop_answers,
            "gold_idxs": gold_idxs,
            "paragraphs": paragraphs,
            "q": q,
            "ans": ans,
            "pos_row": {
                "id": dec_id,
                "dataset": args.dataset,
                "split": args.split,
                "trace_type": "correct",
                "wrong_hops": [],
                "first_wrong_hop": -1,
                "K": k,
                "reasoning_trace": assemble_trace(q, sub_qs, gold_texts, ans),
                "source_id": raw_id,
                "decompose_id": dec_id,
                "variant": variant,
            },
        })

        if len(pending) >= args.example_batch_size:
            n = process_example_batch(
                pending, ranker, args=args, pos_by_k=pos_by_k, neg_pool=neg_pool
            )
            n_ex += n
            pbar.update(n)
            pbar.set_postfix(
                pos=n_ex, neg=sum(len(v) for v in neg_pool.values()), refresh=False
            )
            pending.clear()

        if args.limit_examples and n_ex + len(pending) >= args.limit_examples:
            # flush partial batch up to limit
            remain = args.limit_examples - n_ex
            if remain > 0 and pending:
                chunk = pending[:remain]
                n = process_example_batch(
                    chunk, ranker, args=args, pos_by_k=pos_by_k, neg_pool=neg_pool
                )
                n_ex += n
                pbar.update(n)
            pending.clear()
            break

    if pending:
        n = process_example_batch(
            pending, ranker, args=args, pos_by_k=pos_by_k, neg_pool=neg_pool
        )
        n_ex += n
        pbar.update(n)
    pbar.close()

    # Balance negatives per (K, h) to pos_count[K] * balance_ratio
    selected_neg: list[dict] = []
    balance_report: dict[str, dict] = {}
    for k, pos_rows in sorted(pos_by_k.items()):
        target = int(round(len(pos_rows) * args.balance_ratio))
        for h in range(1, k + 1):
            pool = neg_pool.get((k, h), [])
            key = f"K{k}_h{h}"
            if not pool:
                balance_report[key] = {"pos": len(pos_rows), "target_neg": target, "selected": 0}
                continue
            if len(pool) >= target:
                chosen = rng.sample(pool, target) if len(pool) > target else pool
            else:
                # Upsample rare buckets (e.g. K=4): cycle with shuffle
                chosen = list(pool)
                while len(chosen) < target:
                    extra = list(pool)
                    rng.shuffle(extra)
                    chosen.extend(extra)
                chosen = chosen[:target]
            selected_neg.extend(chosen)
            balance_report[key] = {
                "pos": len(pos_rows),
                "target_neg": target,
                "pool": len(pool),
                "selected": len(chosen),
            }

    n_pos = sum(len(v) for v in pos_by_k.values())
    all_pos: list[dict] = []
    for k in sorted(pos_by_k):
        all_pos.extend(pos_by_k[k])

    all_neg_unbalanced: list[dict] = []
    for key in sorted(neg_pool):
        all_neg_unbalanced.extend(neg_pool[key])

    with open(out_gold, "w", encoding="utf-8") as f:
        for row in tqdm(all_pos, desc="write gold", unit="row"):
            write_row(f, row)

    with open(out_cf, "w", encoding="utf-8") as f:
        for row in tqdm(all_neg_unbalanced, desc="write counterfactual", unit="row"):
            write_row(f, row)

    with open(out_merged, "w", encoding="utf-8") as out_f:
        merged_rows = all_pos + selected_neg
        for row in tqdm(merged_rows, desc="write merged", unit="row"):
            write_row(out_f, row)

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "n_examples": n_ex,
        "n_pos": n_pos,
        "n_neg_unbalanced": len(all_neg_unbalanced),
        "n_neg_balanced": len(selected_neg),
        "pos_by_K": {str(k): len(v) for k, v in sorted(pos_by_k.items())},
        "balance_ratio": args.balance_ratio,
        "cos_topk": args.cos_topk,
        "max_wrong_per_hop": args.max_wrong_per_hop,
        "enhance_decompose": str(enhance_path) if enhance_path else None,
        "n_decompose_base": n_base,
        "n_decompose_enhance": n_enh,
        "device": device,
        "encode_batch_size": args.encode_batch_size,
        "example_batch_size": args.example_batch_size,
        "outputs": {
            "gold": str(out_gold),
            "counterfactual": str(out_cf),
            "merged": str(out_merged),
        },
        "buckets": balance_report,
    }
    stats_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"gold:           {n_pos} → {out_gold}")
    print(f"counterfactual: {len(all_neg_unbalanced)} (unbalanced) → {out_cf}")
    print(f"merged:         {n_pos} pos + {len(selected_neg)} neg (balanced) → {out_merged}")
    print(f"stats → {stats_path}")
    for k in sorted(pos_by_k):
        print(f"  K={k}: {len(pos_by_k[k])} pos")


if __name__ == "__main__":
    main()
