"""
Method registry: the one place that says "method name X = this pool source +
this retriever + this reranker". Adding a new combination (a new retriever, a
new pool source, or a new reranker) never requires touching the hop loop in
run_retrieval_exp_wavefront_v2.py — it's a new entry here.

`oracle` is intentionally NOT a MethodSpec: it doesn't select its own
candidates, it just adds oracle-coverage bookkeeping on top of whichever
methods are already selected (unchanged from the original scripts).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from candidate_pool import GlobalPoolSource, LocalPoolSource, PoolSource
from embedding_cache import GlobalEmbeddingCache
from rerankers import GatedRuleRerank, LRRerank, NoRerank, Reranker
from retrievers import ColbertRetriever, CosineRetriever, Retriever
from run_retrieval_exp import get_gate_artifact

# Which cache kind each global method needs — the single source of truth for
# both build_method_registry (below) and run_retrieval_exp_wavefront_v2.py's
# _load_global_caches (which only needs to know "does this run's --methods
# list need a cosine cache / a colbert cache loaded", not the individual
# method names).
COSINE_GLOBAL_METHODS = frozenset({"baseline_global", "baseline_global_lr_rerank"})
COLBERT_GLOBAL_METHODS = frozenset({"colbert_global", "colbert_global_lr_rerank"})
GLOBAL_METHODS = COSINE_GLOBAL_METHODS | COLBERT_GLOBAL_METHODS
ALL_METHODS = frozenset(
    {"baseline", "colbert", "lr_rerank", "gated_rule_a", "gated_rule_b"} | GLOBAL_METHODS
)


@dataclass
class MethodSpec:
    pool_source: PoolSource
    retriever: Retriever
    reranker: Reranker


def build_method_registry(
    *,
    methods: list[str],
    artifacts: dict,
    gate_artifact_mode: str,
    topk: int,
    expand_topk: int,
    lambda_lr: float,
    abnormal_threshold: float,
    cos_model: str,
    colbert_model: str,
    global_cache_cosine: GlobalEmbeddingCache | None = None,
    global_cache_colbert: GlobalEmbeddingCache | None = None,
) -> dict[str, MethodSpec]:
    unknown = [m for m in methods if m != "oracle" and m not in ALL_METHODS]
    if unknown:
        raise ValueError(f"unknown method(s): {unknown} (known: {sorted(ALL_METHODS)})")

    def _artifact_for(ctx: Any) -> dict | None:
        return get_gate_artifact(artifacts, gate_artifact_mode, ctx.ex.K, ctx.hop_j - 1)

    local = LocalPoolSource()
    registry: dict[str, MethodSpec] = {}

    if "baseline" in methods:
        registry["baseline"] = MethodSpec(local, CosineRetriever(cos_model), NoRerank())

    if "colbert" in methods:
        registry["colbert"] = MethodSpec(local, ColbertRetriever(colbert_model), NoRerank())

    if "lr_rerank" in methods:
        registry["lr_rerank"] = MethodSpec(
            local, CosineRetriever(cos_model), LRRerank(topk=topk, lambda_lr=lambda_lr),
        )

    for rule in ("a", "b"):
        name = f"gated_rule_{rule}"
        if name in methods:
            registry[name] = MethodSpec(
                local,
                CosineRetriever(cos_model),
                GatedRuleRerank(
                    rule,
                    expand_topk=expand_topk,
                    lambda_lr=lambda_lr,
                    default_threshold=abnormal_threshold,
                    get_artifact=_artifact_for,
                ),
            )

    def _require_cache(cache: GlobalEmbeddingCache | None, *, method: str, kind: str) -> GlobalPoolSource:
        if cache is None:
            raise ValueError(
                f"method '{method}' requires a loaded {kind} GlobalEmbeddingCache "
                f"(build one with build_embedding_cache.py --retriever {kind} first)."
            )
        return GlobalPoolSource(cache.paragraphs, cache.precomputed)

    if "baseline_global" in methods:
        pool = _require_cache(global_cache_cosine, method="baseline_global", kind="cosine")
        registry["baseline_global"] = MethodSpec(pool, CosineRetriever(cos_model), NoRerank())

    if "colbert_global" in methods:
        pool = _require_cache(global_cache_colbert, method="colbert_global", kind="colbert")
        registry["colbert_global"] = MethodSpec(pool, ColbertRetriever(colbert_model), NoRerank())

    # Global-pool retrieval + the same emb_score - lambda*gate_score reranking
    # as local-pool lr_rerank — e.g. ColBERT recall over the whole corpus,
    # then the trained LR gate reranks the top-k the same way it always has.
    if "baseline_global_lr_rerank" in methods:
        pool = _require_cache(global_cache_cosine, method="baseline_global_lr_rerank", kind="cosine")
        registry["baseline_global_lr_rerank"] = MethodSpec(
            pool, CosineRetriever(cos_model), LRRerank(topk=topk, lambda_lr=lambda_lr),
        )

    if "colbert_global_lr_rerank" in methods:
        pool = _require_cache(global_cache_colbert, method="colbert_global_lr_rerank", kind="colbert")
        registry["colbert_global_lr_rerank"] = MethodSpec(
            pool, ColbertRetriever(colbert_model), LRRerank(topk=topk, lambda_lr=lambda_lr),
        )

    return registry
