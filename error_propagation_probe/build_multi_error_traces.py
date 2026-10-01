#!/usr/bin/env python3
"""
Build a self-contained multi-error-hop trace set to test whether the gate's
abnormal score at the FINAL transition rises with the NUMBER of wrong hops
in a trace (not just whether the single existing wrong hop is detectable).

Existing counterfactual traces (traces/merged/) only ever corrupt exactly one
hop (confirmed empirically: every wrong_hops in hidden_states/{train,dev}
manifests has length 1), and extraction stops right after that hop
(n_take = wh + 1 in hidden_states/extract_hidden_states.py), so there is no
data anywhere in the repo with >1 wrong hop, or with hidden states past a
wrong hop. This script builds new traces with n_wrong = 0..K wrong hops per
example (0 = the existing gold case) so extract_full_hidden_states.py (this
folder) can capture hidden states all the way to h_K for every one of them.

Reuses (imports, does not copy) traces/dataset_loaders.py,
traces/trace_evidence.py, traces/trace_format.py, and
traces/construct_balanced_traces.py::align_subqs_and_gold — production trace
code is untouched.

Usage:
  python build_multi_error_traces.py --split dev --limit-examples 200

--augment-existing lets you enrich the K=3/K=4 buckets of an ALREADY-BUILT
--out file with the missing wrong-hop combinations WITHOUT touching a single
existing row (no rebuild, no re-extraction of anything already extracted):
it reads --out, and for source_ids whose K is in --exhaustive-for-k, figures
out (from each example's existing wrong_hops) which combinations are still
missing, recomputes cosine-wrong-candidates for just that (typically much
smaller) K=3/K=4 subset, and APPENDS only the new rows. K=2 rows, and any
K=3/K=4 rows already present, are never rewritten, so their ids and content
stay stable and their already-extracted hidden states remain valid; only the
newly appended row ids need `extract_full_hidden_states.py --resume`.

  python build_multi_error_traces.py --split train --augment-existing --exhaustive-for-k 3 4
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
TRACES_DIR = PROJECT_ROOT / "traces"
sys.path.insert(0, str(TRACES_DIR))

from construct_balanced_traces import align_subqs_and_gold  # noqa: E402
from dataset_loaders import get_decompose_path, load_decompose_bundle, load_raw_index  # noqa: E402
from trace_evidence import (  # noqa: E402
    CosinePoolRanker,
    ExampleRetrieval,
    build_hop_queries,
    gold_evidence_musique,
    gold_hop_answers_musique,
    resolve_device,
)
from trace_format import assemble_trace  # noqa: E402


def build_item(dec_id: str, sub_qs, raw_id_map: dict, raw_index: dict) -> dict | None:
    """Same per-example filtering/shaping used to populate `pending` in the full-build path,
    factored out so --augment-existing can rebuild just the handful of items it needs."""
    raw_id = raw_id_map.get(dec_id, dec_id)
    record = raw_index.get(raw_id)
    if record is None:
        return None
    gold_texts, gold_idxs_list = gold_evidence_musique(record)
    hop_answers = gold_hop_answers_musique(record)
    sub_qs_a, gold_texts, hop_answers, k = align_subqs_and_gold(sub_qs, gold_texts, hop_answers)
    if k < 2:
        return None
    return {
        "rid": dec_id,
        "raw_id": raw_id,
        "k": k,
        "sub_qs": sub_qs_a,
        "gold_texts": gold_texts,
        "hop_answers": hop_answers,
        "gold_idxs": set(gold_idxs_list[:k]),
        "paragraphs": list(record.get("paragraphs") or []),
        "q": str(record.get("question", "")).strip(),
        "ans": str(record.get("answer", "")).strip(),
    }


def write_variant_rows(out_f, item: dict, *, dataset: str, split: str, chosen, n_wrong: int,
                        hop_wrongs, id_str: str) -> None:
    ev_texts = list(item["gold_texts"])
    for h in chosen:
        wrong_text, _wrong_pidx, _score = hop_wrongs[h - 1][0]
        ev_texts[h - 1] = wrong_text
    out_f.write(json.dumps({
        "id": id_str, "dataset": dataset, "split": split,
        "trace_type": "error", "wrong_hops": list(chosen), "n_wrong": n_wrong, "K": item["k"],
        "reasoning_trace": assemble_trace(item["q"], item["sub_qs"], ev_texts, item["ans"]),
        "source_id": item["raw_id"], "decompose_id": item["rid"],
    }, ensure_ascii=False) + "\n")


def augment_existing(args, decomp: dict, raw_id_map: dict, raw_index: dict,
                      exhaustive_for_k: set, out_path: Path) -> None:
    """Enrich only the K in exhaustive_for_k buckets of an already-built --out file with the
    missing wrong-hop combinations, appending new rows and never rewriting existing ones."""
    if not out_path.is_file():
        sys.exit(f"--augment-existing needs an existing --out file, none found at {out_path}")

    existing_rows: list[dict] = []
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                existing_rows.append(json.loads(line))

    # decompose_id is the key needed to re-derive an item; keep the FIRST K seen per raw_id
    # (they're all the same K within one raw_id) and the set of wrong_hops already on disk.
    dec_id_by_raw: dict[str, str] = {}
    k_by_raw: dict[str, int] = {}
    existing_combos: dict[str, dict[int, set]] = {}
    for row in existing_rows:
        raw_id = str(row.get("source_id") or "")
        dec_id_by_raw.setdefault(raw_id, str(row.get("decompose_id") or ""))
        k_by_raw[raw_id] = int(row["K"])
        if row.get("trace_type") == "error":
            n_wrong = int(row.get("n_wrong", len(row.get("wrong_hops") or [])))
            existing_combos.setdefault(raw_id, {}).setdefault(n_wrong, set()).add(
                frozenset(row.get("wrong_hops") or [])
            )

    target_raw_ids = [rid for rid, k in k_by_raw.items() if k in exhaustive_for_k]
    print(f"Existing file has {len(k_by_raw)} examples; {len(target_raw_ids)} already have "
          f"K in {sorted(exhaustive_for_k)} and are candidates for enrichment.")

    items: list[dict] = []
    for raw_id in target_raw_ids:
        dec_id = dec_id_by_raw[raw_id]
        sub_qs = decomp.get(dec_id)
        if sub_qs is None:
            continue
        item = build_item(dec_id, sub_qs, raw_id_map, raw_index)
        if item is not None:
            items.append(item)

    print(f"Recomputing cosine-wrong-candidates for {len(items)} examples (K subset only, no Llama) ...")
    device = resolve_device(args.device)
    ranker = CosinePoolRanker(args.cos_model, device=device, encode_batch_size=args.encode_batch_size)

    n_new = 0
    with out_path.open("a", encoding="utf-8") as out_f, tqdm(total=len(items), unit="ex") as pbar:
        for start in range(0, len(items), args.example_batch_size):
            batch = items[start : start + args.example_batch_size]
            retr = [
                ExampleRetrieval(
                    rid=item["rid"], k=item["k"], paragraphs=item["paragraphs"],
                    gold_idxs=item["gold_idxs"],
                    queries=build_hop_queries(item["sub_qs"], item["hop_answers"]),
                )
                for item in batch
            ]
            hop_wrongs_batch = ranker.process_batch(retr, topk=args.cos_topk, max_wrong=1)

            for item, hop_wrongs in zip(batch, hop_wrongs_batch):
                raw_id, rid, k = item["raw_id"], item["rid"], item["k"]
                available_hops = [h for h in range(1, k + 1) if hop_wrongs[h - 1]]
                have = existing_combos.get(raw_id, {})
                for n_wrong in range(1, k + 1):
                    if n_wrong > len(available_hops):
                        continue
                    all_combos = [frozenset(c) for c in itertools.combinations(available_hops, n_wrong)]
                    already = have.get(n_wrong, set())
                    missing = [c for c in all_combos if c not in already]
                    for aug_idx, combo in enumerate(missing):
                        chosen = sorted(combo)
                        id_str = f"{rid}__nwrong{n_wrong}_aug{aug_idx}"
                        write_variant_rows(out_f, item, dataset=args.dataset, split=args.split,
                                           chosen=chosen, n_wrong=n_wrong, hop_wrongs=hop_wrongs,
                                           id_str=id_str)
                        n_new += 1
            pbar.update(len(batch))

    print(f"Appended {n_new} new rows -> {out_path}")
    print("Existing rows (any K) were not touched — only run extract_full_hidden_states.py "
          "--resume on this file to pick up the new row ids.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=("musique",), default="musique")
    ap.add_argument("--split", choices=("train", "dev"), default="dev")
    ap.add_argument("--decompose-file", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--cos-topk", type=int, default=10)
    ap.add_argument("--cos-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-examples", type=int, default=200,
                     help="0 = all eligible examples (can be slow); default keeps this a quick probe. "
                          "With --resume, this is how many NEW examples to add this run, not a total.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--encode-batch-size", type=int, default=64)
    ap.add_argument("--example-batch-size", type=int, default=32)
    ap.add_argument("--resume", action="store_true",
                     help="Keep whatever's already in --out (by source_id) and only add examples "
                          "not already covered, appending instead of overwriting.")
    ap.add_argument("--exhaustive-for-k", type=int, nargs="+", default=[3, 4],
                     help="For examples whose K is in this list, enumerate ALL C(K, n_wrong) "
                          "wrong-hop combinations per n_wrong (instead of one random sample) to "
                          "boost sample count for underrepresented K buckets (MuSiQue's own K "
                          "distribution is dominated by K=2). K values not listed here (e.g. K=2) "
                          "keep the original single-random-sample behavior with unchanged ids, so "
                          "already-extracted hidden states for those rows stay reusable. Pass an "
                          "empty list to disable and reproduce the old behavior for every K.")
    ap.add_argument("--augment-existing", action="store_true",
                     help="Don't rebuild: read --out (must already exist), and for its examples "
                          "whose K is in --exhaustive-for-k, append only the wrong-hop combinations "
                          "that aren't already present. No existing row (any K) is rewritten, so "
                          "already-extracted hidden states stay valid — only run "
                          "extract_full_hidden_states.py --resume afterwards.")
    args = ap.parse_args()

    decompose_path = args.decompose_file or get_decompose_path(args.dataset, args.split, mode="gt")
    if not decompose_path.is_file():
        sys.exit(f"Missing decompose file: {decompose_path}")

    decomp, raw_id_map = load_decompose_bundle(decompose_path, enhance_path=None, include_enhance=False)
    raw_index = load_raw_index(args.dataset, args.split)

    out_path = args.out or (HERE / "data" / args.dataset / f"{args.split}_multi_error.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.augment_existing:
        augment_existing(args, decomp, raw_id_map, raw_index, set(args.exhaustive_for_k), out_path)
        return

    done_raw_ids: set[str] = set()
    if args.resume and out_path.is_file():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_raw_ids.add(json.loads(line)["source_id"])
        print(f"Resuming: {len(done_raw_ids)} examples already in {out_path}")

    exhaustive_for_k = set(args.exhaustive_for_k)
    device = resolve_device(args.device)
    ranker = CosinePoolRanker(args.cos_model, device=device, encode_batch_size=args.encode_batch_size)
    print(f"Cosine ranker: {args.cos_model} on {device}")

    # Gather eligible examples first (same filtering as construct_balanced_traces.py).
    pending: list[dict] = []
    n_seen = 0
    for dec_id, sub_qs in decomp.items():
        raw_id = raw_id_map.get(dec_id, dec_id)
        if raw_id in done_raw_ids:
            continue
        record = raw_index.get(raw_id)
        if record is None:
            continue
        gold_texts, gold_idxs_list = gold_evidence_musique(record)
        hop_answers = gold_hop_answers_musique(record)
        sub_qs_a, gold_texts, hop_answers, k = align_subqs_and_gold(sub_qs, gold_texts, hop_answers)
        if k < 2:
            continue  # need >=2 hops for a "post-error transition" to exist at all
        pending.append({
            "rid": dec_id,
            "raw_id": raw_id,
            "k": k,
            "sub_qs": sub_qs_a,
            "gold_texts": gold_texts,
            "hop_answers": hop_answers,
            "gold_idxs": set(gold_idxs_list[:k]),
            "paragraphs": list(record.get("paragraphs") or []),
            "q": str(record.get("question", "")).strip(),
            "ans": str(record.get("answer", "")).strip(),
        })
        n_seen += 1
        if args.limit_examples and n_seen >= args.limit_examples:
            break

    print(f"Eligible new examples: {len(pending)}"
          + (f" ({len(done_raw_ids)} already done, skipped)" if done_raw_ids else ""))

    file_mode = "a" if args.resume and out_path.is_file() else "w"
    n_rows = 0
    n_skipped_variants = 0
    with out_path.open(file_mode, encoding="utf-8") as out_f, tqdm(total=len(pending), unit="ex") as pbar:
        for start in range(0, len(pending), args.example_batch_size):
            batch = pending[start : start + args.example_batch_size]
            retr = [
                ExampleRetrieval(
                    rid=item["rid"], k=item["k"], paragraphs=item["paragraphs"],
                    gold_idxs=item["gold_idxs"],
                    queries=build_hop_queries(item["sub_qs"], item["hop_answers"]),
                )
                for item in batch
            ]
            hop_wrongs_batch = ranker.process_batch(retr, topk=args.cos_topk, max_wrong=1)

            for item, hop_wrongs in zip(batch, hop_wrongs_batch):
                rid, k = item["rid"], item["k"]
                q, sub_qs, ans = item["q"], item["sub_qs"], item["ans"]

                # n_wrong = 0: pure gold, matches traces/gold/ semantics.
                out_f.write(json.dumps({
                    "id": f"{rid}__nwrong0",
                    "dataset": args.dataset, "split": args.split,
                    "trace_type": "correct", "wrong_hops": [], "n_wrong": 0, "K": k,
                    "reasoning_trace": assemble_trace(q, sub_qs, item["gold_texts"], ans),
                    "source_id": item["raw_id"], "decompose_id": rid,
                }, ensure_ascii=False) + "\n")
                n_rows += 1

                # Which hops actually have a cosine wrong candidate available.
                available_hops = [h for h in range(1, k + 1) if hop_wrongs[h - 1]]
                # Per-example seeded RNG (not one shared stream across all examples) so that
                # enabling/disabling exhaustive enumeration for one K never perturbs the specific
                # random draws made for a DIFFERENT example at a different K (needed to keep K=2
                # rows byte-identical across runs for --resume / already-extracted hidden states).
                ex_rng = random.Random(f"{args.seed}:{item['raw_id']}")
                exhaustive = k in exhaustive_for_k
                for n_wrong in range(1, k + 1):
                    if n_wrong > len(available_hops):
                        n_skipped_variants += 1
                        continue
                    if exhaustive:
                        combos = list(itertools.combinations(available_hops, n_wrong))
                    else:
                        combos = [tuple(sorted(ex_rng.sample(available_hops, n_wrong)))]
                    for variant_idx, chosen in enumerate(combos):
                        ev_texts = list(item["gold_texts"])
                        for h in chosen:
                            wrong_text, _wrong_pidx, _score = hop_wrongs[h - 1][0]
                            ev_texts[h - 1] = wrong_text
                        suffix = f"_c{variant_idx}" if exhaustive else ""
                        out_f.write(json.dumps({
                            "id": f"{rid}__nwrong{n_wrong}{suffix}",
                            "dataset": args.dataset, "split": args.split,
                            "trace_type": "error", "wrong_hops": list(chosen), "n_wrong": n_wrong, "K": k,
                            "reasoning_trace": assemble_trace(q, sub_qs, ev_texts, ans),
                            "source_id": item["raw_id"], "decompose_id": rid,
                        }, ensure_ascii=False) + "\n")
                        n_rows += 1
            pbar.update(len(batch))

    print(f"Wrote {n_rows} rows ({len(pending)} examples) -> {out_path}")
    print(f"Skipped variants (not enough distinct wrong-candidate hops): {n_skipped_variants}")


if __name__ == "__main__":
    main()
