"""
Candidate pool abstraction: WHERE retrieval candidates come from, decoupled
from HOW they're scored (retrievers.py) and WHETHER they get reranked
(rerankers.py). Used by run_retrieval_exp_wavefront_v2.py — the original
run_retrieval_exp.py / run_retrieval_exp_wavefront.py are untouched and keep
their single, hardcoded "search this example's own paragraphs" behavior.

Today's only pool source is "this example's own local distractor/in-pool
paragraphs" (LocalPoolSource — exactly ex.paragraphs, unchanged). The new
piece is GlobalPoolSource: every example searches the same pre-embedded,
whole-dataset corpus (see embedding_cache.py for how that corpus is built and
cached to disk ahead of time).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class CandidatePool:
    """
    paragraphs: same shape as ExampleState.paragraphs today (each a dict with
    at least "idx", "title", "paragraph_text"), except every dict here is
    guaranteed to carry a "_source_id" key identifying which example it
    actually belongs to. For a local pool that's always the current example;
    for a global pool it can be any example in the dataset — this is what
    find_gold_rank_scoped() uses to avoid false gold-rank hits when two
    different examples' paragraphs happen to share the same local `idx`.

    precomputed: opaque, retriever-specific payload. None means "encode
    paragraphs from scratch for this query" (today's only behavior, used by
    LocalPoolSource). A retriever that doesn't understand the payload it's
    handed should fall back to on-the-fly encoding rather than guessing.
    """
    paragraphs: list[dict[str, Any]]
    precomputed: Any | None = None


class PoolSource(ABC):
    @abstractmethod
    def get_pool(self, ex: Any) -> CandidatePool:
        """`ex` is a run_retrieval_exp_wavefront(_v2).ExampleState-like object
        with at least `.eid` and `.paragraphs`."""


def _tag_source_id(paragraphs: list[dict[str, Any]], source_id: str) -> list[dict[str, Any]]:
    """Attach "_source_id" without mutating the caller's original dicts —
    ex.paragraphs is read fresh every hop; mutating it in place would leak
    a stale tag if the same ExampleState were ever reused across a different
    pool source."""
    tagged = []
    for p in paragraphs:
        if p.get("_source_id") == source_id:
            tagged.append(p)
        else:
            q = dict(p)
            q["_source_id"] = source_id
            tagged.append(q)
    return tagged


class LocalPoolSource(PoolSource):
    """Today's only behavior: each example searches its own paragraphs.
    Byte-for-byte equivalent to `ex.paragraphs` as consumed by the original
    scripts, just tagged with _source_id for find_gold_rank_scoped()."""

    def get_pool(self, ex: Any) -> CandidatePool:
        return CandidatePool(paragraphs=_tag_source_id(ex.paragraphs, ex.eid), precomputed=None)


class GlobalPoolSource(PoolSource):
    """Every example searches the same corpus-wide pool. Built once from a
    loaded GlobalEmbeddingCache (embedding_cache.py) whose passages already
    carry the correct "_source_id" per passage (set at cache-build time, not
    here — there is no single "current example" for a global pool)."""

    def __init__(self, paragraphs: list[dict[str, Any]], precomputed: Any):
        for p in paragraphs:
            if "_source_id" not in p:
                raise ValueError(
                    "GlobalPoolSource requires every paragraph to carry a '_source_id' "
                    "(set when the embedding cache was built) — got one without it."
                )
        self._pool = CandidatePool(paragraphs=paragraphs, precomputed=precomputed)

    def get_pool(self, ex: Any) -> CandidatePool:
        return self._pool


def find_gold_rank_scoped(
    ranked_paras: list[tuple[dict[str, Any], float]],
    gold_idx: int,
    *,
    example_id: str,
) -> int | None:
    """
    Same job as run_retrieval_exp.find_gold_rank, but also requires the
    candidate to actually belong to `example_id`.

    run_retrieval_exp.find_gold_rank matches purely on `para["idx"] ==
    gold_idx`, which is safe today because every candidate is always drawn
    from the current example's own paragraph list, where idx is unique. Once
    candidates can come from a GlobalPoolSource (paragraphs pooled across the
    whole dataset), idx collides constantly — every example's own paragraphs
    are numbered from a small local range (0..~19), so some *other* example's
    idx=3 will frequently rank highly for THIS example's query too. Without
    the source-id check, that would silently count as a false gold-rank hit.

    For local pools, every candidate's "_source_id" is already the current
    example's own id (see LocalPoolSource), so this is exactly equivalent to
    the original find_gold_rank there — `para.get("_source_id", example_id)`
    falls back to treating an untagged para as belonging to the current
    example, so any caller that (by mistake) hands in untagged paragraphs
    still gets the old behavior rather than silently matching nothing.
    """
    for rank, (para, _) in enumerate(ranked_paras, start=1):
        if (
            int(para.get("idx", -1)) == gold_idx
            and para.get("_source_id", example_id) == example_id
        ):
            return rank
    return None
