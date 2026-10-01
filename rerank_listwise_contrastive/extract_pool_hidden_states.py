#!/usr/bin/env python3
"""
Scheme B — step 2: extract Delta_j for every candidate in every pool built
by build_topk_pools.py. Needs a real GPU + Llama checkpoint (this is the
one step in Scheme B that can't be tested without one).

Reuses (imports, does not copy) run_retrieval_exp_wavefront.py's
batch_last_hidden (the same batched, cached, layer-generic hidden-state
extractor production's score_gate_requests uses internally) and
load_gate_model — no gate model training/scoring code involved, just the
raw "text -> last-token hidden at layer L" primitive.

For each pool: extract hidden(prefix_before) once, and hidden(prefix_before
+ " Step j+1: <expanded_q> Evidence: \"<candidate text>\"") for every
candidate; Delta_i = hidden(candidate_i) - hidden(prefix_before). All pools'
texts are batched together in one big call for efficiency, not pool by pool.

Output: one .npz with parallel arrays (deltas, emb_scores, is_gold, pool_id,
K, j) — one row per candidate across all pools, keyed by pool_id so
train_listwise_ranker.py can group rows back into their pool.

Usage (needs a real GPU + Llama checkpoint):
  python extract_pool_hidden_states.py --pools data/musique_train_pools.jsonl --layer 31
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"
TRACES_DIR = PROJECT_ROOT / "traces"
sys.path.insert(0, str(RETRIEVAL_DIR))
sys.path.insert(0, str(TRACES_DIR))

from run_retrieval_exp_wavefront import batch_last_hidden, load_gate_model  # noqa: E402
from trace_format import escape_double_quotes  # noqa: E402


def load_pools(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pools", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--gate-device", default="cuda:0")
    ap.add_argument("--hidden-batch-size", type=int, default=128)
    args = ap.parse_args()

    out_path = args.out or args.pools.with_suffix(".hidden.npz")

    pools = load_pools(args.pools)
    print(f"Loaded {len(pools)} pools from {args.pools}")

    model_args = argparse.Namespace(
        model=args.model, dtype=args.dtype, attn_implementation=args.attn_implementation,
        gate_device=args.gate_device,
    )
    print("Loading gate-style Llama model ...", flush=True)
    model, tokenizer = load_gate_model(model_args)

    texts: list[str] = []
    owners: list[tuple[int, int]] = []  # (pool_idx, slot); slot=-1 -> prefix_before
    for pool_idx, pool in enumerate(pools):
        texts.append(pool["prefix_before"])
        owners.append((pool_idx, -1))
        for slot, cand in enumerate(pool["candidates"]):
            text = (
                f'{pool["prefix_before"]} Step {pool["j"] + 1}: {pool["expanded_q"]}'
                f' Evidence: "{escape_double_quotes(cand["paragraph_text"].strip())}"'
            )
            texts.append(text)
            owners.append((pool_idx, slot))

    print(f"Extracting hidden states for {len(texts)} texts (layer {args.layer}) ...", flush=True)
    cache: dict[tuple[int, str], np.ndarray] = {}
    hiddens_by_layer = batch_last_hidden(
        texts=texts, model=model, tokenizer=tokenizer, layers=[args.layer],
        batch_size=args.hidden_batch_size, cache=cache, desc="extract",
    )
    hiddens = hiddens_by_layer[args.layer]

    prefix_hidden: dict[int, np.ndarray] = {}
    cand_hidden: dict[int, dict[int, np.ndarray]] = {}
    for (pool_idx, slot), h in zip(owners, hiddens):
        if slot == -1:
            prefix_hidden[pool_idx] = h
        else:
            cand_hidden.setdefault(pool_idx, {})[slot] = h

    deltas, emb_scores, is_gold, pool_ids, Ks, js = [], [], [], [], [], []
    for pool_idx, pool in enumerate(pools):
        h_prev = prefix_hidden[pool_idx]
        for slot, cand in enumerate(pool["candidates"]):
            h_cand = cand_hidden[pool_idx][slot]
            deltas.append((h_cand - h_prev).astype(np.float16))
            emb_scores.append(float(cand["emb_score"]))
            is_gold.append(bool(cand["is_gold"]))
            pool_ids.append(pool["pool_id"])
            Ks.append(int(pool["K"]))
            js.append(int(pool["j"]))

    np.savez_compressed(
        out_path,
        deltas=np.stack(deltas).astype(np.float32),
        emb_scores=np.asarray(emb_scores, dtype=np.float32),
        is_gold=np.asarray(is_gold, dtype=bool),
        pool_id=np.asarray(pool_ids, dtype=object),
        K=np.asarray(Ks, dtype=np.int64),
        j=np.asarray(js, dtype=np.int64),
    )
    print(f"Wrote {len(deltas)} candidate rows ({len(pools)} pools) -> {out_path}")


if __name__ == "__main__":
    main()
