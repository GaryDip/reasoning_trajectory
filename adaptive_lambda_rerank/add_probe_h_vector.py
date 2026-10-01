#!/usr/bin/env python3
"""
Add each candidate's own PCA-projected h_after vector (the h_j-probe's PCA, layer 31 —
same PCA/layer build_training_data.py already used to compute cand_abnormal_score, just
the intermediate vector was discarded after scoring instead of being kept) to an existing
musique_{split}_lambda_data.npz.

Needed for the "context-conditioned gate" idea (see conversation): instead of a single
fixed weight mixing gate_v2's Delta-score and the probe's h-score, learn a small gate
alpha_j = sigmoid(w^T h_after_j + b) PER CANDIDATE (each candidate's own h_after differs,
since each one appends different evidence text to the same prefix), then combine
final_score = alpha_j * gate_v2_score + (1 - alpha_j) * probe_score. This is structurally
different from the earlier failed adaptive-lambda gate: that one's input was fixed
pre-retrieval context shared by every candidate in a hop's pool (so training had no
candidate-level signal to learn from); this one's input is each candidate's own post-hoc
state, so within one pool different candidates CAN get different alpha.

Rebuilds the SAME pool list build_training_data.py used (must match --pools/--extra-pools),
verifies pool_id alignment against the existing npz (same safety check as
add_gate_v2_scores.py), then for each pool re-extracts hidden states at the probe's own
layer (31) for every candidate, PCA-projects with the probe's own fitted PCA (64-dim,
error_propagation_probe/probe_artifacts/pca.joblib), and appends `cand_h_pca` — shape
(n_candidates, 64) — to the npz. Does NOT recompute cand_abnormal_score (already there).

Chunked and checkpointed the same way add_gate_v2_scores.py is (each chunk's h_pca slice
written to its own file under <npz>.probe_h_chunks/, --resume skips finished chunks).

Usage:
  python add_probe_h_vector.py --split train --extra-pools data/musique_train_pools_enhance.jsonl
  python add_probe_h_vector.py --split dev
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"
PROBE_DIR = PROJECT_ROOT / "error_propagation_probe"
LISTWISE_DIR = PROJECT_ROOT / "rerank_listwise_contrastive"
sys.path.insert(0, str(RETRIEVAL_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "traces"))
sys.path.insert(0, str(HERE))

from run_retrieval_exp_wavefront import batch_last_hidden, load_gate_model  # noqa: E402
from trace_format import escape_double_quotes  # noqa: E402

from build_training_data import load_pools  # noqa: E402
from train_lambda_model import build_pool_ranges  # noqa: E402


def process_pool_chunk(
    pools_chunk: list[dict], *, pca, layer: int, model, tokenizer, hidden_batch_size: int, desc: str,
) -> np.ndarray:
    """Returns a float32 array of shape (n_candidates_in_chunk, pca.n_components_), one row
    per candidate, in the SAME pool-then-candidate order build_training_data.py already
    used for cand_emb_score/cand_abnormal_score."""
    texts: list[str] = []
    owners: list[tuple[int, int]] = []  # (local pool idx, slot) — no prefix_before needed here
    for local_idx, pool in enumerate(pools_chunk):
        for slot, cand in enumerate(pool["candidates"]):
            text = (
                f'{pool["prefix_before"]} Step {pool["j"] + 1}: {pool["expanded_q"]}'
                f' Evidence: "{escape_double_quotes(cand["paragraph_text"].strip())}"'
            )
            texts.append(text)
            owners.append((local_idx, slot))

    cache: dict[tuple[int, str], np.ndarray] = {}
    hiddens_by_layer = batch_last_hidden(
        texts=texts, model=model, tokenizer=tokenizer, layers=[layer],
        batch_size=hidden_batch_size, cache=cache, desc=desc,
    )
    hiddens = hiddens_by_layer[layer]

    cand_hidden: dict[int, dict[int, np.ndarray]] = {}
    for (local_idx, slot), h in zip(owners, hiddens):
        cand_hidden.setdefault(local_idx, {})[slot] = h

    rows: list[np.ndarray] = []
    for local_idx, pool in enumerate(pools_chunk):
        n_cand = len(pool["candidates"])
        h_stack = np.stack([cand_hidden[local_idx][slot] for slot in range(n_cand)]).astype(np.float64)
        rows.append(pca.transform(h_stack))

    return np.concatenate(rows, axis=0).astype(np.float32)


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
    ap.add_argument("--probe-artifacts-dir", type=Path, default=PROBE_DIR / "probe_artifacts")
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--gate-device", default="cuda:0")
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

    pca = joblib.load(args.probe_artifacts_dir / "pca.joblib")
    print(f"Probe PCA: {args.probe_artifacts_dir / 'pca.joblib'} (n_components={pca.n_components_}) "
          f"layer={args.layer}")

    chunk_dir = npz_path.parent / f"{npz_path.stem}.probe_h_chunks"
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
        print("Loading gate-style Llama model ...", flush=True)
        model_args = argparse.Namespace(
            model=args.model, dtype=args.dtype, attn_implementation=args.attn_implementation,
            gate_device=args.gate_device,
        )
        model, tokenizer = load_gate_model(model_args)

        for ci, (start, end) in enumerate(pending_bounds):
            chunk_path = chunk_dir / f"chunk_{start:07d}_{end:07d}.npy"
            print(f"\n=== chunk {ci + 1}/{len(pending_bounds)}: pools [{start}:{end}) ===", flush=True)
            chunk_vecs = process_pool_chunk(
                pools[start:end], pca=pca, layer=args.layer, model=model, tokenizer=tokenizer,
                hidden_batch_size=args.hidden_batch_size, desc=f"chunk{ci + 1}/{len(pending_bounds)}",
            )
            tmp_path = chunk_dir / f"chunk_{start:07d}_{end:07d}.tmp.npy"
            np.save(tmp_path, chunk_vecs)
            tmp_path.rename(chunk_path)
            print(f"[chunk {ci + 1}/{len(pending_bounds)}] saved -> {chunk_path} shape={chunk_vecs.shape}",
                  flush=True)

    all_vecs = [np.load(chunk_dir / f"chunk_{s:07d}_{e:07d}.npy") for s, e in chunk_bounds]
    cand_h_pca = np.concatenate(all_vecs, axis=0)
    if len(cand_h_pca) != len(existing["cand_emb_score"]):
        sys.exit(
            f"Length mismatch after merge: {len(cand_h_pca)} h-vectors vs "
            f"{len(existing['cand_emb_score'])} existing candidate rows — something is wrong, not writing."
        )

    existing["cand_h_pca"] = cand_h_pca.astype(np.float32)
    np.savez_compressed(npz_path, **existing)
    print(f"\nWrote cand_h_pca (shape {cand_h_pca.shape}) into {npz_path}")


if __name__ == "__main__":
    main()
