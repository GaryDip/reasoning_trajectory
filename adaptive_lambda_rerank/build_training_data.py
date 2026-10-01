#!/usr/bin/env python3
"""
Step 1: build training data for a per-hop, context-conditioned lambda predictor.

Idea (see conversation): don't learn a per-candidate scorer at all — keep production's
own rerank formula `emb_score - lambda * abnormal_score` exactly as-is (both emb_score
from BGE and abnormal_score from the error_propagation_probe stay frozen), and instead
learn a TINY model that predicts a single lambda PER HOP (shared across every candidate
in that hop's pool), conditioned on hop-level context:
  - q_main_emb:       BGE(main question), 768-d
  - h_prev:            Llama last-token hidden state of prefix_before (up through hop j-1),
                       4096-d — same extraction as the gate/probe, no new mechanism
  - expanded_q_emb:    BGE(this hop's expanded sub-question), 768-d

Reuses (imports, does not copy):
  - rerank_listwise_contrastive/data/musique_{split}_pools.jsonl — real top-k BGE pools
    already built by build_topk_pools.py (gold-marked, one row per hop). NOT rebuilt here.
  - retrieval/run_retrieval_exp_wavefront.py::batch_last_hidden / load_gate_model — same
    batched Llama hidden-state extractor the gate/probe/beam scripts all use.
  - error_propagation_probe/probe_artifacts/{pca,lr}.joblib — the ALREADY TRAINED probe;
    used here only to score each candidate (frozen, not retrained).
  - retrieval/run_retrieval_exp.py::load_dataset_records — to look up each pool's raw main
    question text from its source_id (pools only store expanded_q, not q_main).

The expensive step (Llama forward over every candidate) is chunked and checkpointed: each
chunk of --chunk-size pools is scored and written to its OWN .npz under
<out>.chunks/chunk_<start>_<end>.npz. Pass --resume to skip chunks whose file already
exists (e.g. after an OOM or a pre-emption on a shared GPU) instead of restarting from
scratch. Once every chunk is present, they're concatenated into the single final .npz
train_lambda_model.py expects — the merge step is idempotent and cheap, so it's safe to
just rerun this command until it reports "done" even if earlier invocations got killed
partway through.

Output: one .npz with pool-level context (q_main_emb, h_prev, expanded_q_emb, K, j,
gold_pos, pool_id) and candidate-level flat arrays (cand_pool_idx, emb_score,
abnormal_score, is_gold) — the ragged candidate lists are recovered at train time by
grouping on cand_pool_idx.

Usage:
  python build_training_data.py --split train
  python build_training_data.py --split train --resume   # after an interrupted run
  python build_training_data.py --split dev
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
LISTWISE_DIR = PROJECT_ROOT / "rerank_listwise_contrastive"
sys.path.insert(0, str(RETRIEVAL_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "traces"))

from run_retrieval_exp import MUSIQUE_DIR, load_dataset_records  # noqa: E402
from run_retrieval_exp_wavefront import batch_last_hidden, load_gate_model  # noqa: E402
from trace_format import escape_double_quotes  # noqa: E402


def load_pools(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_probe(probe_dir: Path):
    import joblib
    pca = joblib.load(probe_dir / "pca.joblib")
    lr = joblib.load(probe_dir / "lr.joblib")
    return pca, lr


def process_pool_chunk(
    pools_chunk: list[dict], expanded_q_chunk: np.ndarray, *, q_main_vec_by_sid: dict[str, np.ndarray],
    model, tokenizer, pca, lr, layer: int, hidden_batch_size: int, desc: str,
) -> dict[str, np.ndarray]:
    """Everything build_training_data.py used to do for ALL pools in one shot, scoped to
    just this chunk — one Llama batch_last_hidden call + one probe-scoring pass."""
    texts: list[str] = []
    owners: list[tuple[int, int]] = []  # (local pool idx within chunk, slot); slot=-1 -> prefix_before
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

    cache: dict[tuple[int, str], np.ndarray] = {}  # chunk-local cache, fine to discard after use
    hiddens_by_layer = batch_last_hidden(
        texts=texts, model=model, tokenizer=tokenizer, layers=[layer],
        batch_size=hidden_batch_size, cache=cache, desc=desc,
    )
    hiddens = hiddens_by_layer[layer]

    prefix_hidden: dict[int, np.ndarray] = {}
    cand_hidden: dict[int, dict[int, np.ndarray]] = {}
    for (local_idx, slot), h in zip(owners, hiddens):
        if slot == -1:
            prefix_hidden[local_idx] = h
        else:
            cand_hidden.setdefault(local_idx, {})[slot] = h

    n = len(pools_chunk)
    q_main_emb = np.zeros((n, 768), dtype=np.float32)
    h_prev = np.zeros((n, hiddens[0].shape[0]), dtype=np.float16)
    Ks = np.zeros(n, dtype=np.int64)
    js = np.zeros(n, dtype=np.int64)
    gold_pos = np.full(n, -1, dtype=np.int64)
    pool_ids = np.empty(n, dtype=object)
    source_ids = np.empty(n, dtype=object)

    cand_pool_idx: list[int] = []
    cand_emb_score: list[float] = []
    cand_abnormal_score: list[float] = []
    cand_is_gold: list[bool] = []

    for local_idx, pool in enumerate(pools_chunk):
        q_main_emb[local_idx] = q_main_vec_by_sid[pool["source_id"]]
        h_prev[local_idx] = prefix_hidden[local_idx].astype(np.float16)
        Ks[local_idx] = int(pool["K"])
        js[local_idx] = int(pool["j"])
        gold_pos[local_idx] = pool["gold_pos"] if pool["gold_pos"] is not None else -1
        pool_ids[local_idx] = pool["pool_id"]
        source_ids[local_idx] = pool["source_id"]

        cand_h = np.stack([cand_hidden[local_idx][slot] for slot in range(len(pool["candidates"]))])
        z = pca.transform(cand_h.astype(np.float64))
        abnormal = lr.predict_proba(z)[:, 1]

        for slot, cand in enumerate(pool["candidates"]):
            cand_pool_idx.append(local_idx)
            cand_emb_score.append(float(cand["emb_score"]))
            cand_abnormal_score.append(float(abnormal[slot]))
            cand_is_gold.append(bool(cand["is_gold"]))

    return dict(
        q_main_emb=q_main_emb,
        h_prev=h_prev,
        expanded_q_emb=np.asarray(expanded_q_chunk, dtype=np.float32),
        K=Ks, j=js, gold_pos=gold_pos, pool_id=pool_ids, source_id=source_ids,
        cand_pool_idx=np.asarray(cand_pool_idx, dtype=np.int64),
        cand_emb_score=np.asarray(cand_emb_score, dtype=np.float32),
        cand_abnormal_score=np.asarray(cand_abnormal_score, dtype=np.float32),
        cand_is_gold=np.asarray(cand_is_gold, dtype=bool),
    )


def merge_chunks(chunk_dir: Path, chunk_bounds: list[tuple[int, int]], out_path: Path) -> None:
    keys_pool = ["q_main_emb", "h_prev", "expanded_q_emb", "K", "j", "gold_pos", "pool_id", "source_id"]
    keys_cand = ["cand_emb_score", "cand_abnormal_score", "cand_is_gold"]
    merged: dict[str, list[np.ndarray]] = {k: [] for k in keys_pool + keys_cand + ["cand_pool_idx"]}

    pool_offset = 0
    for start, end in chunk_bounds:
        chunk_path = chunk_dir / f"chunk_{start:07d}_{end:07d}.npz"
        d = np.load(chunk_path, allow_pickle=True)
        for k in keys_pool:
            merged[k].append(d[k])
        for k in keys_cand:
            merged[k].append(d[k])
        merged["cand_pool_idx"].append(d["cand_pool_idx"] + pool_offset)
        pool_offset += end - start

    final = {k: np.concatenate(v) for k, v in merged.items()}
    np.savez_compressed(out_path, **final)
    n_with_gold = int((final["gold_pos"] >= 0).sum())
    print(f"Merged {len(chunk_bounds)} chunks -> {len(final['K'])} pools "
          f"({n_with_gold} with gold present) -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["musique"], default="musique")
    ap.add_argument("--split", choices=["train", "dev"], default="train")
    ap.add_argument("--pools", type=Path, default=None,
                     help=f"Default: {LISTWISE_DIR}/data/musique_<split>_pools.jsonl")
    ap.add_argument("--extra-pools", type=Path, nargs="+", default=None,
                     help="Additional pool jsonl file(s) to concatenate in (e.g. "
                          "build_enhance_pools.py's K=3/4 paraphrase-augmented pools). "
                          "Never required — omit to reproduce the original single-file behavior.")
    ap.add_argument("--musique-dir", type=Path, default=MUSIQUE_DIR)
    ap.add_argument("--probe-dir", type=Path, default=PROJECT_ROOT / "error_propagation_probe" / "probe_artifacts")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--chunk-size", type=int, default=2000,
                     help="Pools per checkpoint chunk. Smaller = more frequent checkpoints "
                          "but a bit more overhead; larger = fewer, bigger Llama batches.")
    ap.add_argument("--resume", action="store_true",
                     help="Skip chunks whose .npz already exists under <out>.chunks/ instead "
                          "of recomputing them — safe to pass on every invocation, a no-op "
                          "the first time and on a fully-finished run.")
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--gate-device", default="cuda:0")
    ap.add_argument("--hidden-batch-size", type=int, default=16)
    ap.add_argument("--cos-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--limit-pools", type=int, default=0, help="0 = all pools in the file.")
    args = ap.parse_args()

    pools_path = args.pools or (LISTWISE_DIR / "data" / f"{args.dataset}_{args.split}_pools.jsonl")
    if not pools_path.is_file():
        sys.exit(f"Missing pools file: {pools_path} (run rerank_listwise_contrastive/build_topk_pools.py first)")
    out_path = args.out or (HERE / "data" / f"{args.dataset}_{args.split}_lambda_data.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_dir = out_path.parent / f"{out_path.stem}.chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    pools = load_pools(pools_path)
    print(f"Loaded {len(pools)} pools from {pools_path}")
    for extra_path in args.extra_pools or []:
        if not extra_path.is_file():
            sys.exit(f"Missing --extra-pools file: {extra_path}")
        extra = load_pools(extra_path)
        pools.extend(extra)
        print(f"  + {len(extra)} extra pools from {extra_path} (total now {len(pools)})")
    if args.limit_pools:
        pools = pools[: args.limit_pools]

    chunk_bounds = [
        (start, min(start + args.chunk_size, len(pools)))
        for start in range(0, len(pools), args.chunk_size)
    ]
    already_done = {
        (s, e) for s, e in chunk_bounds if (chunk_dir / f"chunk_{s:07d}_{e:07d}.npz").is_file()
    }
    if args.resume and already_done:
        print(f"--resume: {len(already_done)}/{len(chunk_bounds)} chunks already on disk, will skip those")
    pending_bounds = [b for b in chunk_bounds if not (args.resume and b in already_done)]

    if not pending_bounds:
        print("All chunks already present, skipping straight to merge.")
    else:
        ds_args = argparse.Namespace(dataset=args.dataset, musique_dir=args.musique_dir, split=args.split)
        records = load_dataset_records(ds_args)

        # BGE encodes are cheap — always (re)done in full for simplicity, not chunked/resumed.
        from sentence_transformers import SentenceTransformer
        print(f"Loading BGE ({args.cos_model}) ...", flush=True)
        st = SentenceTransformer(args.cos_model)

        unique_source_ids = sorted({p["source_id"] for p in pools})
        q_main_text_by_sid = {
            sid: str((records.get(sid) or {}).get("question", "")).strip() for sid in unique_source_ids
        }
        print(f"Encoding {len(unique_source_ids)} unique main questions ...", flush=True)
        q_main_vecs = st.encode(
            [q_main_text_by_sid[sid] for sid in unique_source_ids],
            normalize_embeddings=True, show_progress_bar=True, batch_size=256,
        )
        q_main_vec_by_sid = dict(zip(unique_source_ids, q_main_vecs))

        print(f"Encoding {len(pools)} expanded sub-questions ...", flush=True)
        expanded_q_vecs = st.encode(
            [p["expanded_q"] for p in pools], normalize_embeddings=True, show_progress_bar=True, batch_size=256,
        )

        print("Loading gate-style Llama model ...", flush=True)
        model_args = argparse.Namespace(
            model=args.model, dtype=args.dtype, attn_implementation=args.attn_implementation,
            gate_device=args.gate_device,
        )
        model, tokenizer = load_gate_model(model_args)

        pca, lr = load_probe(args.probe_dir)
        print(f"Probe loaded from {args.probe_dir}")

        for ci, (start, end) in enumerate(pending_bounds):
            chunk_path = chunk_dir / f"chunk_{start:07d}_{end:07d}.npz"
            print(f"\n=== chunk {ci + 1}/{len(pending_bounds)}: pools [{start}:{end}) ===", flush=True)
            chunk_data = process_pool_chunk(
                pools[start:end], expanded_q_vecs[start:end], q_main_vec_by_sid=q_main_vec_by_sid,
                model=model, tokenizer=tokenizer, pca=pca, lr=lr, layer=args.layer,
                hidden_batch_size=args.hidden_batch_size, desc=f"chunk{ci + 1}/{len(pending_bounds)}",
            )
            # np.savez_compressed appends ".npz" itself if the name doesn't already end in
            # ".npz" — the tmp name must end in ".npz" too, or the file numpy actually writes
            # won't match tmp_path and the rename below fails.
            tmp_path = chunk_dir / f"chunk_{start:07d}_{end:07d}.tmp.npz"
            np.savez_compressed(tmp_path, **chunk_data)
            tmp_path.rename(chunk_path)  # atomic-ish: never leaves a half-written chunk_*.npz behind
            print(f"[chunk {ci + 1}/{len(pending_bounds)}] saved -> {chunk_path}", flush=True)

    merge_chunks(chunk_dir, chunk_bounds, out_path)


if __name__ == "__main__":
    main()
