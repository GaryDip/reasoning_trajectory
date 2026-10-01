#!/usr/bin/env python3
"""
Pack traces_v2 (+ hop_answer labels) + hidden_states_v2 into ONE consolidated cache so training
never has to touch the 83k individual .npz files again (that's what made the first real-data
smoke test crawl -- a flat directory of 83k small files is slow to hit repeatedly, once per
trace per epoch). All hops from all traces get concatenated into one big float32 array; a
companion index gives each trace's (start, end) slice into it plus its case_id/K/labels.

Output: one .npz with:
  x_all            [total_hops, 8192]  float32   -- concat(h_after, delta) per hop, in trace order
  evidence_label    [total_hops]        float32   -- 1.0 if is_correct else 0.0 (binary)
  hop_answer_label  [total_hops]        float32   -- hop_answer_f1 (continuous [0,1], soft BCE
                                                      target, not binarized EM -- partial credit
                                                      on a near-miss short answer is real signal)
  case_id           [n_traces]          object (str)
  K                 [n_traces]          int32
  final_f1          [n_traces]          float32
  offset_start      [n_traces]          int64     -- x_all[offset_start[i]:offset_end[i]] is trace i's hops
  offset_end        [n_traces]          int64
  trace_id          [n_traces]          object (str)

Usage:
  python build_feature_cache.py \
    --traces-path output_full/traces_v2/musique/train_with_hop_labels.jsonl \
    --hidden-states-dir output_full/hidden_states_v2/musique/train \
    --out-path gate_mtl_shared_backbone/cache/musique_train.npz
(paths are relative to new_data_recipe/ for the first two, matching where build_dataset.py wrote them)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces-path", type=Path, required=True)
    ap.add_argument("--hidden-states-dir", type=Path, required=True)
    ap.add_argument("--out-path", type=Path, required=True)
    ap.add_argument("--progress-every", type=int, default=5000)
    args = ap.parse_args()

    t0 = time.perf_counter()
    x_chunks: list[np.ndarray] = []
    evidence_label_chunks: list[np.ndarray] = []
    hop_answer_label_chunks: list[np.ndarray] = []
    case_ids: list[str] = []
    trace_ids: list[str] = []
    Ks: list[int] = []
    final_f1s: list[float] = []
    offset_starts: list[int] = []
    offset_ends: list[int] = []

    cursor = 0
    n_lines = 0
    n_skipped_missing_npz = 0
    n_skipped_missing_labels = 0

    with args.traces_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            t = json.loads(line)
            npz_path = args.hidden_states_dir / f"{t['trace_id']}.npz"
            if not npz_path.is_file():
                n_skipped_missing_npz += 1
                continue
            if any(h.get("hop_answer_f1") is None for h in t["hops"]) or t.get("final_answer_f1") is None:
                n_skipped_missing_labels += 1
                continue

            with np.load(npz_path) as d:
                hidden = d["hidden"].astype(np.float32)  # [K+1, hidden_dim]
            K = len(t["hops"])
            h_after = hidden[1:K + 1]
            h_prev = hidden[0:K]
            delta = h_after - h_prev
            x = np.concatenate([h_after, delta], axis=-1)  # [K, 2*hidden_dim]

            x_chunks.append(x)
            evidence_label_chunks.append(np.array([1.0 if h["is_correct"] else 0.0 for h in t["hops"]], dtype=np.float32))
            hop_answer_label_chunks.append(np.array([float(h["hop_answer_f1"]) for h in t["hops"]], dtype=np.float32))
            case_ids.append(t["case_id"])
            trace_ids.append(t["trace_id"])
            Ks.append(K)
            final_f1s.append(float(t["final_answer_f1"]))
            offset_starts.append(cursor)
            cursor += K
            offset_ends.append(cursor)

            if n_lines % args.progress_every == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {n_lines} lines processed ({len(case_ids)} kept), {elapsed:.1f}s elapsed", flush=True)

    print(f"\nconcatenating {len(x_chunks)} traces / {cursor} hops ...", flush=True)
    x_all = np.concatenate(x_chunks, axis=0)
    evidence_label = np.concatenate(evidence_label_chunks, axis=0)
    hop_answer_label = np.concatenate(hop_answer_label_chunks, axis=0)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_path,
        x_all=x_all,
        evidence_label=evidence_label,
        hop_answer_label=hop_answer_label,
        case_id=np.array(case_ids, dtype=object),
        trace_id=np.array(trace_ids, dtype=object),
        K=np.array(Ks, dtype=np.int32),
        final_f1=np.array(final_f1s, dtype=np.float32),
        offset_start=np.array(offset_starts, dtype=np.int64),
        offset_end=np.array(offset_ends, dtype=np.int64),
    )

    elapsed = time.perf_counter() - t0
    print(f"\n{n_lines} lines read, {n_skipped_missing_npz} missing npz, {n_skipped_missing_labels} missing labels")
    print(f"kept {len(case_ids)} traces / {cursor} hops")
    print(f"x_all shape: {x_all.shape}, {x_all.nbytes / 1e9:.2f} GB")
    print(f"wrote -> {args.out_path}  ({elapsed:.1f}s total)")


if __name__ == "__main__":
    main()
