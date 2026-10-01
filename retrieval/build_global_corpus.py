#!/usr/bin/env python3
"""
Build a shared "global" retrieval corpus for one dataset+split, by pooling every example's own
paragraphs (gold + distractors) and deduplicating -- this is the prerequisite for running
run_retrieval_exp_wavefront_gate_v3_rawprefix.py with --retrieval-scope global instead of the
default per-example "distractor setting" pool.

Dedup key is (title, paragraph_text) together, NOT title alone -- the same real-world Wikipedia
article can appear in different examples' context with slightly different extracted text (some
datasets select different sentence ranges per example), so deduping on title alone risks
silently merging genuinely different text under one entry. Costs a slightly larger corpus in
exchange for never conflating two different snippets.

Output (two files, human-inspectable, not a pickle):
  corpus_meta.jsonl  -- one line per doc: {"idx": int, "title": str, "paragraph_text": str}
                        idx is this corpus's global id, aligned by row order with corpus_emb.npy
  corpus_emb.npy     -- float32 [N, D] embedding matrix, row i = doc at corpus_meta.jsonl line i
  corpus_summary.json -- n_examples, n_paragraphs_seen, n_unique (dedup stats), cos_model, dataset/split

Usage:
  python build_global_corpus.py --dataset musique --split dev
  python build_global_corpus.py --dataset 2wiki --split dev --out-dir global_corpus/2wiki_dev
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
TRACES_DIR = PROJECT_ROOT / "traces"
for p in (HERE, TRACES_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_retrieval_exp import (  # noqa: E402
    MUSIQUE_DIR, TWOWIKI_DEV_FILE, HOTPOT_DEV_FILE, load_dataset_records,
)
from trace_evidence import get_ranker  # noqa: E402


def load_global_corpus(corpus_dir: Path) -> tuple[list[dict], np.ndarray, dict[tuple[str, str], int]]:
    """Load a corpus this script already built. Returns (paragraphs, doc_emb, title_text_to_idx)
    -- paragraphs[i]/doc_emb[i] are row-aligned (paragraphs[i]["idx"] == i by construction),
    title_text_to_idx maps (title, paragraph_text) -> global idx, for remapping an example's own
    LOCAL gold_idx (from its own paragraphs list) into this corpus's global id space."""
    corpus_dir = Path(corpus_dir)
    meta_path = corpus_dir / "corpus_meta.jsonl"
    emb_path = corpus_dir / "corpus_emb.npy"
    paragraphs: list[dict] = []
    with meta_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                paragraphs.append(json.loads(line))
    doc_emb = np.load(emb_path)
    title_text_to_idx = {(p["title"], p["paragraph_text"]): p["idx"] for p in paragraphs}
    return paragraphs, doc_emb, title_text_to_idx


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["musique", "2wiki", "hotpot"], required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--musique-dir", type=Path, default=MUSIQUE_DIR)
    ap.add_argument("--twowiki-file", type=Path, default=TWOWIKI_DEV_FILE)
    ap.add_argument("--hotpot-file", type=Path, default=HOTPOT_DEV_FILE)
    ap.add_argument("--cos-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--encode-batch-size", type=int, default=64)
    ap.add_argument("--out-dir", type=Path, default=None,
                     help="defaults to global_corpus/{dataset}_{split}/ next to this script")
    ap.add_argument("--limit", type=int, default=0,
                     help="only pool the first N examples (0 = all) -- for a quick smoke test "
                          "of dedup/embedding before committing to the full split")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = HERE / "global_corpus" / f"{args.dataset}_{args.split}"
    return args


def main() -> None:
    t0 = time.perf_counter()
    args = parse_args()
    records = load_dataset_records(args)
    ids = sorted(records.keys())
    if args.limit:
        ids = ids[: args.limit]
    print(f"{len(ids)} {args.dataset} {args.split} examples")

    # dedup by (title, paragraph_text) -- see module docstring for why not title alone.
    seen: dict[tuple[str, str], int] = {}
    meta: list[dict] = []
    n_paragraphs_seen = 0
    for eid in ids:
        row = records[eid]
        for p in row.get("paragraphs") or []:
            n_paragraphs_seen += 1
            title = (p.get("title") or "").strip()
            text = (p.get("paragraph_text") or "").strip()
            if not text:
                continue
            key = (title, text)
            if key in seen:
                continue
            idx = len(meta)
            seen[key] = idx
            meta.append({"idx": idx, "title": title, "paragraph_text": text})

    print(f"{n_paragraphs_seen} paragraphs seen across all examples -> {len(meta)} unique "
          f"(title, text) pairs ({len(meta) / n_paragraphs_seen:.1%} kept)")

    print(f"Encoding {len(meta)} unique paragraphs with {args.cos_model} ...")
    ranker = get_ranker(args.cos_model)
    doc_texts = [f"{m['title']}\n{m['paragraph_text']}" for m in meta]
    doc_emb = ranker.encode_texts(doc_texts)
    print(f"embedding matrix: {doc_emb.shape}, dtype={doc_emb.dtype}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.out_dir / "corpus_meta.jsonl"
    emb_path = args.out_dir / "corpus_emb.npy"
    summary_path = args.out_dir / "corpus_summary.json"

    with meta_path.open("w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    np.save(emb_path, doc_emb.astype(np.float32))

    summary = {
        "dataset": args.dataset, "split": args.split, "cos_model": args.cos_model,
        "n_examples": len(ids), "n_paragraphs_seen": n_paragraphs_seen, "n_unique": len(meta),
        "embedding_dim": int(doc_emb.shape[1]) if len(meta) else None,
        "wall_clock_sec": round(time.perf_counter() - t0, 1),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nwrote -> {meta_path}")
    print(f"wrote -> {emb_path}")
    print(f"wrote -> {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
