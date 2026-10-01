#!/usr/bin/env python3
"""Load the consolidated feature cache (build_feature_cache.py's output) and batch straight out
of the in-memory packed arrays -- no per-trace file I/O, no torch Dataset/DataLoader overhead,
just numpy slicing + padding. See update_doc discussion 2026-09-04."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


class PackedCache:
    def __init__(self, path: Path | str) -> None:
        d = np.load(path, allow_pickle=True)
        self.x_all = d["x_all"]                    # [total_hops, 8192] float32
        self.evidence_label = d["evidence_label"]    # [total_hops]
        self.hop_answer_label = d["hop_answer_label"]  # [total_hops]
        self.case_id = d["case_id"]                  # [n_traces] object
        self.trace_id = d["trace_id"]                 # [n_traces] object
        self.K = d["K"]                               # [n_traces] int32
        self.final_f1 = d["final_f1"]                 # [n_traces] float32
        self.offset_start = d["offset_start"]          # [n_traces] int64
        self.offset_end = d["offset_end"]               # [n_traces] int64
        self.n_traces = len(self.case_id)
        self.is_natural_seed: np.ndarray | None = None  # loaded lazily via load_natural_seed_flags

    def load_natural_seed_flags(self, traces_path: Path | str) -> None:
        """recipe/is_natural_seed isn't in the packed cache (added after it was built) --
        pulled from traces_v2.jsonl once here instead of rebuilding the whole (npz-backed)
        cache. Pure JSON scan, no hidden-state I/O, seconds not minutes."""
        flag_by_trace_id: dict[str, bool] = {}
        with open(traces_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = json.loads(line)
                flag_by_trace_id[t["trace_id"]] = bool(t.get("is_natural_seed", False))
        self.is_natural_seed = np.array(
            [flag_by_trace_id.get(tid, False) for tid in self.trace_id], dtype=bool
        )

    def indices_for_case_ids(self, case_ids: set[str]) -> np.ndarray:
        mask = np.array([cid in case_ids for cid in self.case_id])
        return np.nonzero(mask)[0]

    def indices_for_case_ids_and_k(self, case_ids: set[str], k: int, *, natural_only: bool = False) -> np.ndarray:
        mask = np.array([(cid in case_ids) and (kk == k) for cid, kk in zip(self.case_id, self.K)])
        if natural_only:
            if self.is_natural_seed is None:
                raise RuntimeError("call load_natural_seed_flags() first")
            mask &= self.is_natural_seed
        return np.nonzero(mask)[0]


def batch_from_indices(cache: PackedCache, trace_indices: np.ndarray) -> dict:
    """Build one padded batch (mask included) from a set of trace indices, by slicing straight
    out of cache.x_all -- no copies until the final pad/stack."""
    Ks = cache.K[trace_indices]
    K_max = int(Ks.max())
    B = len(trace_indices)
    D = cache.x_all.shape[1]

    x = np.zeros((B, K_max, D), dtype=np.float32)
    mask = np.zeros((B, K_max), dtype=np.float32)
    evidence_label = np.zeros((B, K_max), dtype=np.float32)
    hop_answer_label = np.zeros((B, K_max), dtype=np.float32)
    final_f1_label = np.zeros((B, K_max), dtype=np.float32)

    for i, ti in enumerate(trace_indices):
        s, e = int(cache.offset_start[ti]), int(cache.offset_end[ti])
        K = e - s
        x[i, :K] = cache.x_all[s:e]
        mask[i, :K] = 1.0
        evidence_label[i, :K] = cache.evidence_label[s:e]
        hop_answer_label[i, :K] = cache.hop_answer_label[s:e]
        final_f1_label[i, :K] = cache.final_f1[ti]

    return {
        "x": torch.from_numpy(x),
        "mask": torch.from_numpy(mask),
        "evidence_label": torch.from_numpy(evidence_label),
        "hop_answer_label": torch.from_numpy(hop_answer_label),
        "final_f1_label": torch.from_numpy(final_f1_label),
        "K": Ks,
        "trace_idx": trace_indices,
    }


def iter_epoch_batches(
    cache: PackedCache, trace_indices: np.ndarray, batch_size: int, *, shuffle: bool = True, seed: int | None = None,
) -> Iterator[dict]:
    idx = trace_indices.copy()
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for s in range(0, len(idx), batch_size):
        yield batch_from_indices(cache, idx[s:s + batch_size])
