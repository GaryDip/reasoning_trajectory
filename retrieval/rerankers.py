"""
Reranker abstraction: does a first-stage retrieval result get re-scored (and
possibly expanded) before becoming the final top-3, independent of where
candidates came from (candidate_pool.py) or how they were first scored
(retrievers.py).

Two-phase interface mirrors what run_retrieval_exp_wavefront.py's
score_gate_requests already does in one batched call across many contexts at
once (for vLLM/gate-model batching efficiency): first score a small initial
slice, decide per-context whether more candidates need scoring, then score
the expanded slice for only the contexts that needed it. rerank_driver.py
drives this; a new reranker is just a new subclass here, the driver never
changes.

Every class's math is the original run_retrieval_exp_wavefront.py hop loop's
855-924 lines, unchanged — this file only reorganizes it into reusable
objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from run_retrieval_exp import rerank_by_final_score


class Reranker(ABC):
    @abstractmethod
    def initial_requests(self, ctx: Any) -> list[tuple[dict, float]] | None:
        """Which of ctx.candidates_all to gate-score first. Return None if no
        scoring is needed at all — the reranker must have already set
        ctx.top3 directly in that case (see NoRerank)."""

    @abstractmethod
    def apply_initial(self, ctx: Any, scored: list[tuple[dict, float, float]]) -> bool:
        """scored is [(para, emb_score, abnormal_score), ...], parallel to
        initial_requests(ctx). Sets ctx.top3 if already resolved. Returns
        True iff an expand pass is needed."""

    def expand_requests(self, ctx: Any) -> list[tuple[dict, float]] | None:
        """Only called when apply_initial returned True."""
        return None

    def apply_expand(self, ctx: Any, scored: list[tuple[dict, float, float]]) -> None:
        """Sets ctx.top3 from the expanded scored candidates."""


class NoRerank(Reranker):
    """baseline / colbert today: take the first-stage top-3 as-is, no gate
    scoring at all."""

    def initial_requests(self, ctx: Any) -> None:
        ctx.top3 = ctx.candidates_all[:3]
        return None

    def apply_initial(self, ctx: Any, scored) -> bool:
        return False


class LRRerank(Reranker):
    """lr_rerank today: gate-score the top-`topk` candidates, then rerank by
    emb_score - lambda * abnormal_score, keep the new top-3."""

    def __init__(self, *, topk: int, lambda_lr: float):
        self.topk = topk
        self.lambda_lr = lambda_lr

    def initial_requests(self, ctx: Any) -> list[tuple[dict, float]]:
        return ctx.candidates_all[: self.topk]

    def apply_initial(self, ctx: Any, scored: list[tuple[dict, float, float]]) -> bool:
        ctx.top3 = rerank_by_final_score(scored, self.lambda_lr)[:3]
        return False


class GatedRuleRerank(Reranker):
    """gated_rule_a / gated_rule_b today: gate-score only the top-3 first; if
    the rule fires (rule "a": top-1 abnormal score over threshold; rule "b":
    at least 2 of the top-3 over threshold), expand to `expand_topk` and
    rerank by emb_score - lambda * abnormal_score. If it doesn't fire, keep
    the original (embedding-only-ranked) top-3.

    `get_artifact(ctx)` looks up this (K, hop) transition's gate artifact so
    its own calibrated threshold can be used (falling back to
    `default_threshold` for artifacts saved without one) — passed in rather
    than looked up here so this class doesn't need to know about the
    artifacts dict's shape or the pooled/non-pooled mode switch.
    """

    def __init__(
        self,
        rule: str,
        *,
        expand_topk: int,
        lambda_lr: float,
        default_threshold: float,
        get_artifact: Callable[[Any], dict | None],
    ):
        if rule not in ("a", "b"):
            raise ValueError(f"rule must be 'a' or 'b', got {rule!r}")
        self.rule = rule
        self.expand_topk = expand_topk
        self.lambda_lr = lambda_lr
        self.default_threshold = default_threshold
        self.get_artifact = get_artifact

    def initial_requests(self, ctx: Any) -> list[tuple[dict, float]]:
        return ctx.candidates_all[:3]

    def apply_initial(self, ctx: Any, scored: list[tuple[dict, float, float]]) -> bool:
        ab_top3 = [ab for _, _, ab in scored]
        art = self.get_artifact(ctx) or {}
        threshold = art.get("threshold", self.default_threshold)
        if self.rule == "a":
            fired = bool(ab_top3 and ab_top3[0] > threshold)
        else:
            fired = sum(ab > threshold for ab in ab_top3) >= 2
        ctx.gate_fired = fired
        ctx.top3_initial_ab_scores = [round(ab, 4) for ab in ab_top3]
        if fired:
            return True
        ctx.top3 = [(p, s) for p, s, _ in scored]
        return False

    def expand_requests(self, ctx: Any) -> list[tuple[dict, float]]:
        return ctx.candidates_all[: self.expand_topk]

    def apply_expand(self, ctx: Any, scored: list[tuple[dict, float, float]]) -> None:
        ctx.top3 = rerank_by_final_score(scored, self.lambda_lr)[:3]
