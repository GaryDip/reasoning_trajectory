#!/usr/bin/env python3
"""Order-invariant gold-passage matching (the "v2" protocol, 2026-10-11).

Why this exists
---------------
HotpotQA ships exactly two supporting paragraphs and the annotation fixes one arbitrary order
for them; the two branches of a 2Wiki `comparison` / `bridge_comparison` question are likewise
interchangeable. Comparing the selected passage at hop j against `gold_idxs[j]` therefore counts a
legitimate reordering as a retrieval error, which costs HotpotQA about 23 points of chain@1.

The first order-invariant implementation (`p_j in set(gold_idxs)`) over-corrected: picking the
SAME gold passage at two different hops passed, which is one of the most common real failures on
comparison questions. This module is the fixed protocol: a hop may only claim a gold passage that
no earlier hop already claimed.

Scope
-----
`order_invariant_applies` decides per question. MuSiQue decompositions, and 2Wiki
`compositional` / `inference`, are genuinely ordered and stay positional, so for them both
protocols return identical numbers (which doubles as a self-check).

2Wiki bridge_comparison
-----------------------
The dedup'd `supporting_facts` title order is **interleaved**, not branch-by-branch: for all
2,751 dev questions it equals the order the `evidences` triples first mention each subject, i.e.
`(branch1 hop1, branch2 hop1, branch1 hop2, branch2 hop2)` -- e.g. for "Which film has the
director who was born later, El Extraño Viaje or Love In Pawn?" the gold order is
(El extraño viaje, Love in Pawn, Fernando Fernán Gómez, Charles Saunders). The chain structure
g1->g3 and g2->g4 is confirmed on 2,601/2,751 by matching each triple's object against the next
triple's subject (the other 150 differ only in disambiguation suffixes, e.g. "Charles Saunders"
vs "Charles Saunders (director)" -- a string artefact, not a different structure).

So the branch grouping is NOT (g1,g2)+(g3,g4), and `ORDER_INVARIANT_TASK.md`'s "two branches
swapped as a whole" rule does not describe this data. Per that document's own fallback we use the
plain "permutation without reuse" rule, and additionally report a `precedence` diagnostic that
keeps the only ordering constraint the data really has: each film must precede its own director
(g1 before g3, g2 before g4). It is strictly between the positional and the permutation rules.
"""
from __future__ import annotations

ORDER_INVARIANT_2WIKI_TYPES = ("comparison", "bridge_comparison")


def order_invariant_applies(dataset: str, qtype: str | None) -> bool:
    if dataset == "hotpot":
        return True
    if dataset == "2wiki":
        return (qtype or "") in ORDER_INVARIANT_2WIKI_TYPES
    return False


def _gold_set(gold: list) -> set:
    """Gold ids, dropping the None entries a global-corpus run leaves for unmapped passages."""
    return {g for g in gold if g is not None}


def hop_hits(para_ids: list, gold: list, oi: bool) -> list[bool]:
    """Per-hop hit over the aligned prefix j < min(len(para_ids), len(gold)).

    positional       para_ids[j] == gold[j]
    order-invariant  para_ids[j] in set(gold) and para_ids[j] not in para_ids[:j]
    """
    m = min(len(para_ids), len(gold))
    if not oi:
        return [para_ids[j] == gold[j] and gold[j] is not None for j in range(m)]
    G = _gold_set(gold)
    out = []
    for j in range(m):
        out.append(para_ids[j] in G and para_ids[j] not in para_ids[:j])
    return out


def hop_hits_at_k(para_ids: list, gold: list, ranked: list[list] | None, oi: bool,
                  k: int = 3) -> list[bool] | None:
    """Per-hop hit@k. Returns None when the run's case log has no candidate lists.

    positional       gold[j] is among the top k of hop j
    order-invariant  some top-k candidate of hop j is gold and was not already selected earlier
    """
    if not ranked:
        return None
    m = min(len(para_ids), len(gold), len(ranked))
    G = _gold_set(gold)
    out = []
    for j in range(m):
        top = list(ranked[j])[:k]
        if oi:
            earlier = set(para_ids[:j])
            out.append(any(a in G and a not in earlier for a in top))
        else:
            out.append(gold[j] is not None and gold[j] in top)
    return out


def chain_at1(para_ids: list, gold: list, oi: bool, *, dedup: bool = True) -> tuple[bool, bool]:
    """(loose, strict). loose ignores a wrong hop count; strict requires Kp == Kg.

    `dedup=False` reproduces the first, buggy order-invariant implementation (a gold passage
    could be claimed twice), kept so the size of that bias can be reported.
    """
    m = min(len(para_ids), len(gold))
    if m == 0:
        return False, False
    if not oi:
        loose = all(para_ids[j] == gold[j] and gold[j] is not None for j in range(m))
        return loose, loose and len(para_ids) == len(gold)
    G = _gold_set(gold)
    prefix = para_ids[:m]
    loose = all(p in G for p in prefix) and (not dedup or len(set(prefix)) == m)
    strict = (len(para_ids) == len(gold) and len(G) == len(gold)
              and set(para_ids) == G and (not dedup or len(set(para_ids)) == len(para_ids)))
    return loose, strict


def selected_a_gold_twice(para_ids: list, gold: list) -> bool:
    """Did two different hops select the same gold passage? (what the old rule let through)"""
    G = _gold_set(gold)
    seen = set()
    for p in para_ids:
        if p in G:
            if p in seen:
                return True
            seen.add(p)
    return False


def bridge_comparison_precedence_ok(para_ids: list, gold: list) -> bool | None:
    """The 2Wiki bridge_comparison diagnostic: a permutation that keeps g1<g3 and g2<g4.

    Returns None when the shape is not the annotated 4-hop one, so callers can skip it.
    """
    if len(gold) != 4 or len(para_ids) != 4 or len(_gold_set(gold)) != 4:
        return None
    if set(para_ids) != set(gold) or len(set(para_ids)) != 4:
        return False
    pos = {p: i for i, p in enumerate(para_ids)}
    return pos[gold[0]] < pos[gold[2]] and pos[gold[1]] < pos[gold[3]]
