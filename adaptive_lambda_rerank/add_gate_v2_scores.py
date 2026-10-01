#!/usr/bin/env python3
"""
Add gate v2 scores (Delta_j = h_j - h_{j-1}, production's gate/artifacts_pooled_v2) to an
existing musique_{split}_lambda_data.npz (built by build_training_data.py), which already
has emb_score + h_j-probe abnormal_score for the same real candidate pools — needed to test
the "confidence-weighted" combination idea (see conversation) offline, without a full
end-to-end wavefront rerun each time a new combination idea comes up.

Rebuilds the SAME pool list build_training_data.py used (same --pools/--extra-pools — for
train that means the original pools + the K=3/4 paraphrase-augmented ones), then for each
pool re-extracts hidden states ONLY at gate v2's own needed layer for that pool's j (15 or
23 — NOT layer 31, that's the probe's and is already in the npz), scores each candidate's
Delta through the appropriate per-j gate v2 model (reusing get_gate_artifact — the same
lookup production's own score_gate_requests uses), and appends `cand_gate_v2_score` (same
length/order as the existing cand_emb_score/cand_abnormal_score) to the npz.

Verifies row-for-row `pool_id` alignment between the freshly-loaded pools and the existing
npz before writing anything, so a --pools/--extra-pools mismatch can't silently scramble
the array order.

Chunked and checkpointed the same way build_training_data.py is (each chunk's gate_v2_score
slice written to its own file under <npz>.gate_v2_chunks/, --resume skips finished chunks) —
this is a real Llama pass over ~1.2M candidate texts for train, worth being resumable.

Usage:
  python add_gate_v2_scores.py --split train --extra-pools data/musique_train_pools_enhance.jsonl
  python add_gate_v2_scores.py --split dev
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
GATE_DIR = PROJECT_ROOT / "gate"
LISTWISE_DIR = PROJECT_ROOT / "rerank_listwise_contrastive"
sys.path.insert(0, str(RETRIEVAL_DIR))
sys.path.insert(0, str(GATE_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "traces"))
sys.path.insert(0, str(HERE))

from run_retrieval_exp import MUSIQUE_DIR, get_gate_artifact  # noqa: E402
from run_retrieval_exp_wavefront import batch_last_hidden, load_gate_model  # noqa: E402
from lr_artifacts import load_artifacts  # noqa: E402
from trace_format import escape_double_quotes  # noqa: E402

from build_training_data import load_pools  # noqa: E402
from train_lambda_model import build_pool_ranges  # noqa: E402


def process_pool_chunk(
    pools_chunk: list[dict], *, gate_v2_artifacts: dict, model, tokenizer, hidden_batch_size: int, desc: str,
) -> np.ndarray:
    """Returns a flat float32 array, one gate_v2_score per candidate, in the SAME
    pool-then-candidate order build_training_data.py already used for cand_emb_score."""
    # Every pool in one chunk might need a different layer (15 vs 23 depending on j), so
    # batch_last_hidden is asked for the union of layers this chunk actually needs.
    layer_by_pool: list[int] = []
    for pool in pools_chunk:
        art = get_gate_artifact(gate_v2_artifacts, "pooled", 0, int(pool["j"]))
        layer_by_pool.append(int(art["layer"]) if art is not None else 31)
    needed_layers = sorted(set(layer_by_pool))

    texts: list[str] = []
    owners: list[tuple[int, int]] = []  # (local pool idx, slot); slot=-1 -> prefix_before
    for local_idx, pool in enumerate(pools_chunk):
        texts.append(pool["prefix_before"])
        owners.append((local_idx, -1))
        for slot, cand in enumerate(pool["candidates"]):
            text = (
                f'{pool["prefix_before"]} Step {pool["j"] + 1}: {pool["expanded_q"]}'
                f' Evidence: "{escape_double_quotes(cand["paragraph_text"].strip())}"'
            )
            texts.append(text)
            owners.append((local_idx, slot))

    cache: dict[tuple[int, str], np.ndarray] = {}
    hiddens_by_layer = batch_last_hidden(
        texts=texts, model=model, tokenizer=tokenizer, layers=needed_layers,
        batch_size=hidden_batch_size, cache=cache, desc=desc,
    )

    prefix_hidden: dict[int, dict[int, np.ndarray]] = {}
    cand_hidden: dict[int, dict[int, dict[int, np.ndarray]]] = {}
    for (local_idx, slot), idx in ((o, i) for i, o in enumerate(owners)):
        for layer in needed_layers:
            h = hiddens_by_layer[layer][idx]
            if slot == -1:
                prefix_hidden.setdefault(local_idx, {})[layer] = h
            else:
                cand_hidden.setdefault(local_idx, {}).setdefault(slot, {})[layer] = h

    scores: list[float] = []
    for local_idx, pool in enumerate(pools_chunk):
        layer = layer_by_pool[local_idx]
        art = get_gate_artifact(gate_v2_artifacts, "pooled", 0, int(pool["j"]))
        n_cand = len(pool["candidates"])
        if art is None:
            scores.extend([0.0] * n_cand)
            continue
        h_prev = prefix_hidden[local_idx][layer]
        deltas = np.stack([
            cand_hidden[local_idx][slot][layer] - h_prev for slot in range(n_cand)
        ]).astype(np.float64)
        z = art["pca"].transform(deltas)
        abnormal = art["lr"].predict_proba(z)[:, 1]
        scores.extend(abnormal.tolist())

    return np.asarray(scores, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["musique"], default="musique")
    ap.add_argument("--split", choices=["train", "dev"], default="train")
    ap.add_argument("--npz", type=Path, default=None,
                     help=f"Default: {HERE}/data/musique_<split>_lambda_data.npz")
    ap.add_argument("--pools", type=Path, default=None,
                     help=f"Default: {LISTWISE_DIR}/data/musique_<split>_pools.jsonl — MUST "
                          "match what build_training_data.py used to build --npz, or the "
                          "pool_id alignment check below will fail on purpose.")
    ap.add_argument("--extra-pools", type=Path, nargs="+", default=None)
    ap.add_argument("--gate-v2-artifacts-dir", type=Path, default=GATE_DIR / "artifacts_pooled_v2")
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--gate-device", default="cuda:1")
    ap.add_argument("--hidden-batch-size", type=int, default=32)
    ap.add_argument("--limit-pools", type=int, default=0)
    args = ap.parse_args()

    npz_path = args.npz or (HERE / "data" / f"{args.dataset}_{args.split}_lambda_data.npz")
    if not npz_path.is_file():
        sys.exit(f"Missing {npz_path} — run build_training_data.py --split {args.split} first.")
    pools_path = args.pools or (LISTWISE_DIR / "data" / f"{args.dataset}_{args.split}_pools.jsonl")
    if not pools_path.is_file():
        sys.exit(f"Missing pools file: {pools_path}")

    existing = dict(np.load(npz_path, allow_pickle=True))
    n_pools_existing = len(existing["K"])
    print(f"Existing npz: {npz_path} ({n_pools_existing} pools)")

    pools = load_pools(pools_path)
    for extra_path in args.extra_pools or []:
        if not extra_path.is_file():
            sys.exit(f"Missing --extra-pools file: {extra_path}")
        pools.extend(load_pools(extra_path))
    if args.limit_pools:
        pools = pools[: args.limit_pools]
    print(f"Loaded {len(pools)} pools from --pools/--extra-pools")

    if len(pools) != n_pools_existing:
        sys.exit(
            f"Pool count mismatch: {len(pools)} freshly loaded vs {n_pools_existing} in the "
            f"npz — --pools/--extra-pools doesn't match what build_training_data.py used. "
            f"Refusing to guess an alignment."
        )
    mismatches = [
        i for i, (p, existing_id) in enumerate(zip(pools, existing["pool_id"])) if p["pool_id"] != existing_id
    ]
    if mismatches:
        sys.exit(
            f"pool_id mismatch at {len(mismatches)} position(s) (first: index {mismatches[0]}, "
            f"{pools[mismatches[0]]['pool_id']!r} != {existing['pool_id'][mismatches[0]]!r}) — "
            f"the pools are in a different order than the npz. Refusing to guess an alignment."
        )
    print("pool_id alignment check passed — freshly loaded pools match the npz row-for-row.")

    ranges = build_pool_ranges(existing["cand_pool_idx"], n_pools_existing)
    for pool_idx, (pool, (s, e)) in enumerate(zip(pools, ranges)):
        if len(pool["candidates"]) != (e - s):
            sys.exit(
                f"Candidate count mismatch at pool {pool_idx} ({pool['pool_id']}): "
                f"{len(pool['candidates'])} in --pools vs {e - s} in the npz."
            )

    chunk_dir = npz_path.parent / f"{npz_path.stem}.gate_v2_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_bounds = [
        (start, min(start + args.chunk_size, len(pools))) for start in range(0, len(pools), args.chunk_size)
    ]
    already_done = {
        (s, e) for s, e in chunk_bounds if (chunk_dir / f"chunk_{s:07d}_{e:07d}.npy").is_file()
    }
    if args.resume and already_done:
        print(f"--resume: {len(already_done)}/{len(chunk_bounds)} chunks already on disk, will skip those")
    pending_bounds = [b for b in chunk_bounds if not (args.resume and b in already_done)]

    if pending_bounds:
        gate_v2_artifacts = load_artifacts(args.gate_v2_artifacts_dir, "pooled")
        print(f"Gate v2 artifacts: {args.gate_v2_artifacts_dir} (j -> layer: "
              f"{ {j: art['layer'] for j, art in gate_v2_artifacts.items()} })")

        print("Loading gate-style Llama model ...", flush=True)
        model_args = argparse.Namespace(
            model=args.model, dtype=args.dtype, attn_implementation=args.attn_implementation,
            gate_device=args.gate_device,
        )
        model, tokenizer = load_gate_model(model_args)

        for ci, (start, end) in enumerate(pending_bounds):
            chunk_path = chunk_dir / f"chunk_{start:07d}_{end:07d}.npy"
            print(f"\n=== chunk {ci + 1}/{len(pending_bounds)}: pools [{start}:{end}) ===", flush=True)
            chunk_scores = process_pool_chunk(
                pools[start:end], gate_v2_artifacts=gate_v2_artifacts, model=model, tokenizer=tokenizer,
                hidden_batch_size=args.hidden_batch_size, desc=f"chunk{ci + 1}/{len(pending_bounds)}",
            )
            tmp_path = chunk_dir / f"chunk_{start:07d}_{end:07d}.tmp.npy"
            np.save(tmp_path, chunk_scores)
            tmp_path.rename(chunk_path)
            print(f"[chunk {ci + 1}/{len(pending_bounds)}] saved -> {chunk_path}", flush=True)

    all_scores = [np.load(chunk_dir / f"chunk_{s:07d}_{e:07d}.npy") for s, e in chunk_bounds]
    cand_gate_v2_score = np.concatenate(all_scores)
    if len(cand_gate_v2_score) != len(existing["cand_emb_score"]):
        sys.exit(
            f"Length mismatch after merge: {len(cand_gate_v2_score)} gate_v2 scores vs "
            f"{len(existing['cand_emb_score'])} existing candidate rows — something is wrong, not writing."
        )

    existing["cand_gate_v2_score"] = cand_gate_v2_score.astype(np.float32)
    np.savez_compressed(npz_path, **existing)
    print(f"\nWrote cand_gate_v2_score ({len(cand_gate_v2_score)} rows) into {npz_path}")


if __name__ == "__main__":
    main()
