"""
Batched driver that ties Reranker instances (rerankers.py) to
run_retrieval_exp_wavefront.score_gate_requests() — the batched hidden-state
gate scorer that was made multi-layer/per-artifact-aware in a prior session.
score_gate_requests() itself is imported unchanged; this module only decides
WHICH contexts feed it and HOW their results get applied, by delegating to
each context's own method's Reranker.

Adding a new reranker never requires touching this function — write a new
Reranker subclass (rerankers.py) and register it in method_registry.py.
"""

from __future__ import annotations

from typing import Any

from run_retrieval_exp_wavefront import GateRequest, score_gate_requests


def run_rerank_batch(
    contexts: list[Any],
    method_specs: dict[str, Any],
    *,
    artifacts: dict,
    gate_artifact_mode: str,
    model,
    tokenizer,
    default_layer: int,
    hidden_batch_size: int,
    hidden_cache: dict,
    desc: str | None = None,
) -> None:
    """Mutates each ctx in `contexts` in place, setting ctx.top3 (and, for
    gated rerankers, ctx.gate_fired / ctx.top3_initial_ab_scores)."""
    gate_req_id = 0
    gate_reqs: list[GateRequest] = []

    for ctx in contexts:
        reranker = method_specs[ctx.method].reranker
        reqs = reranker.initial_requests(ctx)
        if reqs is None:
            # Reranker resolved ctx.top3 directly without needing any gate
            # scoring at all (e.g. NoRerank).
            continue
        gate_reqs.append(GateRequest(gate_req_id, ctx, reqs))
        gate_req_id += 1

    if not gate_reqs:
        return

    scored = score_gate_requests(
        gate_reqs,
        artifacts=artifacts,
        gate_artifact_mode=gate_artifact_mode,
        model=model,
        tokenizer=tokenizer,
        default_layer=default_layer,
        hidden_batch_size=hidden_batch_size,
        hidden_cache=hidden_cache,
        desc=desc,
    )

    expand_reqs: list[GateRequest] = []
    for req in gate_reqs:
        ctx = req.ctx
        reranker = method_specs[ctx.method].reranker
        needs_expand = reranker.apply_initial(ctx, scored[req.req_id])
        if not needs_expand:
            continue
        exp = reranker.expand_requests(ctx)
        if not exp:
            continue
        expand_reqs.append(GateRequest(gate_req_id, ctx, exp))
        gate_req_id += 1

    if not expand_reqs:
        return

    expanded_scored = score_gate_requests(
        expand_reqs,
        artifacts=artifacts,
        gate_artifact_mode=gate_artifact_mode,
        model=model,
        tokenizer=tokenizer,
        default_layer=default_layer,
        hidden_batch_size=hidden_batch_size,
        hidden_cache=hidden_cache,
        desc=f"{desc}-expand" if desc else None,
    )
    for req in expand_reqs:
        ctx = req.ctx
        reranker = method_specs[ctx.method].reranker
        reranker.apply_expand(ctx, expanded_scored[req.req_id])
