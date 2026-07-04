"""Shared hop / transition label helpers for LR gate training."""

from __future__ import annotations

from typing import Any, Optional


def parse_wrong_hops(
    mrow: Optional[dict[str, Any]] = None,
    z: Any = None,
) -> list[int]:
    """Selection-error hops (1-indexed), e.g. [1, 2] or [2, 3, 4]."""
    wrong_hops: list[int] = []
    if mrow:
        raw = mrow.get("wrong_hops")
        if isinstance(raw, list):
            wrong_hops = [int(h) for h in raw]
        elif mrow.get("first_wrong_hop") is not None:
            v = int(mrow["first_wrong_hop"])
            if v >= 1:
                wrong_hops = [v]
        else:
            legacy = mrow.get("wrong_evidence_at_hop")
            if legacy is not None:
                v = int(legacy)
                if v >= 1:
                    wrong_hops = [v]
    if not wrong_hops and z is not None:
        import numpy as np

        if "wrong_hops" in z:
            wrong_hops = [int(x) for x in np.asarray(z["wrong_hops"]).reshape(-1)]
        elif "first_wrong_hop" in z:
            v = int(np.asarray(z["first_wrong_hop"]).reshape(-1)[0])
            if v >= 1:
                wrong_hops = [v]
        elif "wrong_evidence_at_hop" in z:
            v = int(np.asarray(z["wrong_evidence_at_hop"]).reshape(-1)[0])
            if v >= 1:
                wrong_hops = [v]
    return sorted(set(wrong_hops))


def should_intervene_at_j(
    wrong_hops: list[int],
    j: int,
    *,
    trace_type: str,
) -> bool:
    """
    Positive label at transition j: hop (j+1) had a selection error.

    hop 1 wrong -> j=0 (Q->E1); hop 3 wrong -> j=2 (E2->E3).
    """
    if trace_type != "error" or not wrong_hops:
        return False
    return (j + 1) in wrong_hops


def err_hop_bucket_for_row(
    wrong_hops: list[int],
    j: int,
    *,
    trace_type: str,
) -> str:
    """Metric bucket for one (trace, transition j) row."""
    if trace_type == "correct":
        return "correct"
    hop = j + 1
    if hop in wrong_hops:
        return f"err_hop_{hop}"
    return "error_other"
