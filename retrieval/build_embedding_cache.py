#!/usr/bin/env python3
"""
Precompute + cache passage embeddings for a global retrieval pool (one full
dataset split, all examples' paragraphs pooled together). Run this once per
(dataset, split, retriever, model) combination ahead of time; experiment runs
using run_retrieval_exp_wavefront_v2.py with a global-pool method
(baseline_global / colbert_global) just load the cache built here — see
embedding_cache.py for the on-disk format and GlobalEmbeddingCache.load().

Usage:
  python build_embedding_cache.py --dataset musique --split dev --retriever cosine
  python build_embedding_cache.py --dataset musique --split dev --retriever colbert
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from embedding_cache import DEFAULT_CACHE_ROOT, build_colbert_cache, build_cosine_cache
from run_retrieval_exp import HOTPOT_DEV_FILE, MUSIQUE_DIR, TWOWIKI_DEV_FILE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["musique", "2wiki", "hotpot"], required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--musique-dir", type=Path, default=MUSIQUE_DIR)
    ap.add_argument("--twowiki-file", type=Path, default=TWOWIKI_DEV_FILE)
    ap.add_argument("--hotpot-file", type=Path, default=HOTPOT_DEV_FILE)
    ap.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    ap.add_argument("--retriever", choices=["cosine", "colbert"], default="cosine")
    ap.add_argument("--model", default=None,
                    help="Default: BAAI/bge-base-en-v1.5 (cosine) or colbert-ir/colbertv2.0 (colbert).")
    ap.add_argument("--encode-batch-size", type=int, default=None,
                    help="Default: 256 (cosine) or 32 (colbert).")
    args = ap.parse_args()

    common = dict(
        dataset=args.dataset, split=args.split, cache_root=args.cache_root,
        musique_dir=args.musique_dir, twowiki_file=args.twowiki_file, hotpot_file=args.hotpot_file,
    )
    if args.retriever == "cosine":
        build_cosine_cache(
            model_name=args.model or "BAAI/bge-base-en-v1.5",
            encode_batch_size=args.encode_batch_size or 256,
            **common,
        )
    else:
        build_colbert_cache(
            model_name=args.model or "colbert-ir/colbertv2.0",
            encode_batch_size=args.encode_batch_size or 32,
            **common,
        )


if __name__ == "__main__":
    main()
