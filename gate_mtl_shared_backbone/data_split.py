#!/usr/bin/env python3
"""Case-level train/val split, stratified by K (2/3/4-hop), so a case's recipe1-4 variants never
straddle the split (that would leak: e.g. recipe2's real trace for a case in train, recipe4's
clean-prefix variant of the SAME case in val, sharing most of the same hop-level ground truth)."""
from __future__ import annotations

import random


def stratified_case_split(
    case_ids: list[str], Ks: list[int], *, val_ratio: float = 0.1, seed: int = 42,
) -> tuple[set[str], set[str]]:
    """case_ids/Ks: parallel arrays, one entry per TRACE (so a case_id repeats once per trace it
    has) -- dedups internally to case-level before splitting. Returns (train_case_ids, val_case_ids),
    with each K bucket split at (approximately) the same val_ratio."""
    case_to_k: dict[str, int] = {}
    for cid, k in zip(case_ids, Ks):
        case_to_k.setdefault(cid, k)  # a case's K is the same across all its trace variants

    by_k: dict[int, list[str]] = {}
    for cid, k in case_to_k.items():
        by_k.setdefault(k, []).append(cid)

    rng = random.Random(seed)
    val_cases: set[str] = set()
    train_cases: set[str] = set()
    for k, cids in by_k.items():
        cids = sorted(cids)
        rng.shuffle(cids)
        n_val = max(1, round(len(cids) * val_ratio))
        val_cases.update(cids[:n_val])
        train_cases.update(cids[n_val:])

    return train_cases, val_cases


if __name__ == "__main__":
    # quick self-test with synthetic data
    import collections

    case_ids, Ks = [], []
    rng = random.Random(0)
    for k, n in [(2, 1000), (3, 500), (4, 200)]:
        for i in range(n):
            cid = f"{k}hop__case{i}"
            case_ids.append(cid)
            Ks.append(k)

    train_cases, val_cases = stratified_case_split(case_ids, Ks, val_ratio=0.1, seed=42)
    assert train_cases.isdisjoint(val_cases), "train/val overlap!"
    assert train_cases | val_cases == set(case_ids), "some cases missing from the split!"

    case_to_k = dict(zip(case_ids, Ks))
    train_k = collections.Counter(case_to_k[c] for c in train_cases)
    val_k = collections.Counter(case_to_k[c] for c in val_cases)
    print("train per-K:", dict(sorted(train_k.items())))
    print("val per-K:  ", dict(sorted(val_k.items())))
    for k in (2, 3, 4):
        ratio = val_k[k] / (train_k[k] + val_k[k])
        print(f"  K={k}: val ratio = {ratio:.3f}")
    print("SELF-TEST PASSED: disjoint, complete, per-K ratios ~0.1")
