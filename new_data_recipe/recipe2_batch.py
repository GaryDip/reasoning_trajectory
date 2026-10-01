#!/usr/bin/env python3
"""
Step 6 (see update_doc/0907/0907update.md section 9): batched version of recipe2_core's hop
loop -- same real retrieval + real generation + real propagation + post-hoc labeling + pair
extraction, but processing a LIST of cases together, batching the embedding-encode and LLM
generate_batch calls across all active cases at each hop (mirrors
online_listwise_gate/train_online_wavefront.py's hop-batched structure).

Also implements recipe 3 (forced wrong-hop injection, step 7) and recipe 4 (forced clean
prefix + real continuation, step 9) as optional parameters on the SAME batched loop, since all
three recipes are the same mechanics with different rules for "what gets committed at hop j".
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
TRACES_DIR = PROJECT_ROOT / "traces"
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"
BASE_INFER = PROJECT_ROOT.parent / "multihop_trajectory" / "llama_infer_reasoning"
for p in (TRACES_DIR, RETRIEVAL_DIR, BASE_INFER):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from trace_evidence import expand_hop_template, paragraph_doc_text, get_ranker  # noqa: E402
from trace_format import escape_double_quotes  # noqa: E402
from run_musique_pipeline import prompt_short_answer, judge_answer_official  # noqa: E402
from run_retrieval_exp_wavefront import (  # noqa: E402
    build_final_reader_cot_prompt_comparison_hint,
    parse_final_answer_from_cot,
)

# BGE's official s2p (short query -> long passage) retrieval instruction: only the QUERY side
# gets this prefix, never the passage/document side -- see BAAI/bge-*-en-v1.5 model cards.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _query_encode_texts(ranker, texts: list[str], cos_model: str) -> np.ndarray:
    """Encode QUERY-side texts, prepending the BGE retrieval instruction when the ranker is a
    BGE model (matches what BGE's own model card recommends for asymmetric retrieval; passages
    encoded via paragraph_doc_text/encode_texts directly are deliberately left unprefixed)."""
    if "bge" in cos_model.lower():
        texts = [BGE_QUERY_INSTRUCTION + t for t in texts]
    return ranker.encode_texts(texts)


def run_recipe2_batch(
    cases,
    *,
    generator,
    cos_model="BAAI/bge-base-en-v1.5",
    retrieve_k=10,
    max_new_tokens=32,
    ranker=None,
    forced_hops=None,
    trace_id_suffix="recipe2_real",
    recipe_name="2_real_baseline",
    clean_prefix_gold_hop_answers=None,
    clean_prefix_upto=None,
    target_patterns=None,
    is_natural_seed=False,
):
    """
    cases: list of {case_id, question, sub_qs, gold_idxs, gold_texts, paragraphs}, each already
      truncated to that case's own K.
    forced_hops: {case_list_index: hop_number (1-indexed)} -- at that hop, force-pick the
      highest-cosine NON-gold candidate instead of the natural top-1 (recipe 3, single-point).
      If no qualifying non-gold candidate exists, falls back to natural top-1.
    clean_prefix_upto / clean_prefix_gold_hop_answers: {case_list_index: hop_number} -- hops
      1..that number use GOLD evidence + gold-hop-answer-based query expansion (recipe 4's
      clean prefix); hops after it proceed via real retrieval as usual.
    target_patterns: {case_list_index: [bool, ...] (length K)} -- EVERY hop is forced according
      to the pattern (True -> force gold, False -> force a non-gold candidate), covering all
      2^K-1 non-all-correct combinations (the all-correct one is recipe 1's job). Takes priority
      over forced_hops/clean_prefix_upto for a given case. Query expansion still always uses
      whatever was REALLY generated at each prior hop (prior_answers), exactly like the natural
      path -- only which candidate gets committed is prescribed, not the propagated text.
    is_natural_seed: tag every trace produced by this call as the "real, nothing forced" seed
      trace or not (recipe 2's own call passes True; every other recipe passes False, the
      default) -- lets downstream consumers tell "what actually happened" apart from
      "deliberately constructed" traces without having to re-derive it from forced_hops/
      clean_prefix_upto/target_patterns bookkeeping.
    Returns one trace dict per case, same schema run_recipe2_trace produces (plus
    "forced_hop"/"clean_prefix_upto"/"target_pattern"/"is_natural_seed" fields as applicable).
    """
    forced_hops = forced_hops or {}
    clean_prefix_upto = clean_prefix_upto or {}
    clean_prefix_gold_hop_answers = clean_prefix_gold_hop_answers or {}
    target_patterns = target_patterns or {}
    ranker = ranker or get_ranker(cos_model)
    n = len(cases)
    K_list = [len(c["sub_qs"]) for c in cases]
    max_hops = max(K_list) if K_list else 0

    flat_docs = []
    spans = []
    for c in cases:
        s = len(flat_docs)
        flat_docs.extend(paragraph_doc_text(p) for p in c["paragraphs"])
        spans.append((s, len(flat_docs)))
    flat_doc_emb = ranker.encode_texts(flat_docs) if flat_docs else np.zeros((0, 768))
    doc_embs = [flat_doc_emb[s:e] for s, e in spans]

    prior_answers = [[] for _ in cases]
    hops_out = [[] for _ in cases]
    pairs_out = [[] for _ in cases]
    prefix_texts = [f"Question: {c['question'].strip()}" for c in cases]

    for j in range(max_hops):
        active_idx = [i for i in range(n) if j < K_list[i]]
        if not active_idx:
            break

        in_clean_prefix = {i: (j < clean_prefix_upto.get(i, 0)) for i in active_idx}
        expanded_qs = {}
        for i in active_idx:
            raw_sq = cases[i]["sub_qs"][j]
            if in_clean_prefix[i]:
                gold_answers = clean_prefix_gold_hop_answers.get(i, [])
                expanded_qs[i] = expand_hop_template(raw_sq, gold_answers[:j])
            else:
                expanded_qs[i] = expand_hop_template(raw_sq, prior_answers[i])

        q_texts = [expanded_qs[i] for i in active_idx]
        q_emb = _query_encode_texts(ranker, q_texts, cos_model) if q_texts else np.zeros((0, 768))

        committed = {}
        top3s = {}
        forced_applied = {}
        for local_i, i in enumerate(active_idx):
            paragraphs = cases[i]["paragraphs"]
            doc_emb = doc_embs[i]
            gold_idx = cases[i]["gold_idxs"][j] if j < len(cases[i]["gold_idxs"]) else None
            if doc_emb.shape[0] == 0:
                committed[i] = None
                top3s[i] = []
                forced_applied[i] = False
                continue
            sim = doc_emb @ q_emb[local_i]
            order = np.argsort(-sim)[:retrieve_k]
            ranked = [(paragraphs[int(o)], float(sim[int(o)])) for o in order]
            top3s[i] = ranked[:3]

            if i in target_patterns:
                want_correct = target_patterns[i][j]
                if want_correct:
                    gold_para = next((p for p in paragraphs if int(p.get("idx", -1)) == gold_idx), None)
                    committed[i] = (gold_para, 1.0) if gold_para is not None else (ranked[0] if ranked else None)
                    forced_applied[i] = gold_para is not None
                else:
                    wrong = next((pr for pr in ranked if int(pr[0].get("idx", -1)) != gold_idx), None)
                    committed[i] = wrong if wrong is not None else (ranked[0] if ranked else None)
                    forced_applied[i] = wrong is not None
            elif in_clean_prefix[i]:
                gold_para = next((p for p in paragraphs if int(p.get("idx", -1)) == gold_idx), None)
                committed[i] = (gold_para, 1.0) if gold_para is not None else (ranked[0] if ranked else None)
                forced_applied[i] = gold_para is not None
            elif i in forced_hops and forced_hops[i] == j + 1:
                wrong = next((pr for pr in ranked if int(pr[0].get("idx", -1)) != gold_idx), None)
                if wrong is not None:
                    committed[i] = wrong
                    forced_applied[i] = True
                else:
                    committed[i] = ranked[0] if ranked else None
                    forced_applied[i] = False
            else:
                committed[i] = ranked[0] if ranked else None
                forced_applied[i] = False

        prompts = []
        prompt_owner = []
        for i in active_idx:
            if committed[i] is None:
                continue
            para, _score = committed[i]
            prompts.append(prompt_short_answer(expanded_qs[i], para))
            prompt_owner.append(i)
        raw_answers = generator.generate_batch(prompts, max_new_tokens, desc=f"hop{j + 1}") if prompts else []
        answer_by_i = {i: a.strip() for i, a in zip(prompt_owner, raw_answers)}

        for i in active_idx:
            raw_sq = cases[i]["sub_qs"][j]
            expanded_q = expanded_qs[i]
            gold_idx = cases[i]["gold_idxs"][j] if j < len(cases[i]["gold_idxs"]) else None
            if committed[i] is None:
                hops_out[i].append({
                    "hop": j + 1, "sub_question_raw": raw_sq, "sub_question_expanded": expanded_q,
                    "cosine_top3": [], "committed_idx": None, "gold_idx": gold_idx,
                    "is_correct": False, "short_answer_generated": None,
                })
                prior_answers[i].append("NA")
                continue

            para, _score = committed[i]
            committed_idx = int(para.get("idx", -1))
            is_correct = committed_idx == gold_idx
            short_answer = answer_by_i.get(i, "NA")

            hop_row = {
                "hop": j + 1,
                "sub_question_raw": raw_sq,
                "sub_question_expanded": expanded_q,
                "cosine_top3": [
                    {"idx": int(p.get("idx", -1)), "title": p.get("title"), "score": round(float(s), 4)}
                    for p, s in top3s[i]
                ],
                "committed_idx": committed_idx,
                "committed_evidence_text": (para.get("paragraph_text") or "").strip(),
                "gold_idx": gold_idx,
                "is_correct": is_correct,
                "short_answer_generated": short_answer,
            }
            if forced_applied.get(i) and (i in target_patterns or i in forced_hops or in_clean_prefix[i]):
                hop_row["forced"] = True

            case_id = cases[i]["case_id"]
            trace_id = f"{case_id}__{trace_id_suffix}"
            if not is_correct and j < len(cases[i]["gold_texts"]):
                pair_id = f"{trace_id}__hop{j + 1}"
                hop_row["pair_id"] = pair_id
                pairs_out[i].append({
                    "pair_id": pair_id, "case_id": case_id, "trace_id": trace_id, "hop": j + 1,
                    "context_prefix": prefix_texts[i], "sub_question": expanded_q,
                    "evidence_correct": {"idx": gold_idx, "text": cases[i]["gold_texts"][j]},
                    "evidence_wrong": {"idx": committed_idx, "text": hop_row["committed_evidence_text"]},
                })

            hops_out[i].append(hop_row)
            prior_answers[i].append(short_answer)
            ev_escaped = escape_double_quotes(hop_row["committed_evidence_text"])
            prefix_texts[i] += f' Step {j + 1}: {expanded_q} Evidence: "{ev_escaped}"'

    traces = []
    for i, c in enumerate(cases):
        n_correct = sum(1 for h in hops_out[i] if h["is_correct"])
        trace = {
            "trace_id": f"{c['case_id']}__{trace_id_suffix}",
            "case_id": c["case_id"],
            "recipe": recipe_name,
            "K": K_list[i],
            "question": c["question"],
            "hops": hops_out[i],
            "pairs": pairs_out[i],
            "all_correct": n_correct == K_list[i],
            "n_correct": n_correct,
            "is_natural_seed": bool(is_natural_seed),
        }
        if i in forced_hops:
            trace["forced_hop"] = forced_hops[i]
        if i in clean_prefix_upto:
            trace["clean_prefix_upto"] = clean_prefix_upto[i]
        if i in target_patterns:
            trace["target_pattern"] = [bool(x) for x in target_patterns[i]]
        traces.append(trace)
    return traces


def add_final_answers_batch(traces, gold_records, generator, *, max_new_tokens=128):
    """Batched version of recipe2_core.add_final_answer."""
    prompts = []
    for trace in traces:
        hop_steps = [(h["sub_question_expanded"], h.get("committed_evidence_text") or "") for h in trace["hops"]]
        prior = [h.get("short_answer_generated") or "NA" for h in trace["hops"]]
        prompts.append(build_final_reader_cot_prompt_comparison_hint(trace["question"], hop_steps, prior))
    raw_answers = generator.generate_batch(prompts, max_new_tokens, desc="final_answer") if prompts else []
    for trace, gold_record, raw in zip(traces, gold_records, raw_answers):
        predicted = parse_final_answer_from_cot(raw, fallback=raw.strip())
        em, f1 = judge_answer_official(predicted, gold_record)
        trace["final_answer_generated"] = predicted
        trace["final_answer_em"] = bool(em)
        trace["final_answer_f1"] = f1
    return traces


def build_recipe1_gold_trace(case, gold_hop_answers):
    """Step 8 (see 0907update.md section 9): pure gold trace, deterministic, no LLM calls --
    every hop uses gold evidence, sub-questions expanded with expand_hop_template using GOLD
    hop answers (this is what fixes motivation 1's query-form mismatch: same expansion
    convention retrieval already uses, just fed gold answers instead of a real generation)."""
    from trace_evidence import build_hop_queries

    K = len(case["sub_qs"])
    expanded_qs = build_hop_queries(case["sub_qs"], gold_hop_answers)
    hops = []
    for j in range(K):
        gold_idx = case["gold_idxs"][j]
        hops.append({
            "hop": j + 1,
            "sub_question_raw": case["sub_qs"][j],
            "sub_question_expanded": expanded_qs[j],
            "cosine_top3": [],
            "committed_idx": gold_idx,
            "committed_evidence_text": case["gold_texts"][j],
            "gold_idx": gold_idx,
            "is_correct": True,
            "short_answer_generated": gold_hop_answers[j] if j < len(gold_hop_answers) else "",
        })
    return {
        "trace_id": f"{case['case_id']}__recipe1_gold",
        "case_id": case["case_id"],
        "recipe": "1_gold_injected",
        "K": K,
        "question": case["question"],
        "hops": hops,
        "pairs": [],
        "all_correct": True,
        "n_correct": K,
        "is_natural_seed": False,
    }


def should_skip_recipe1(recipe2_trace):
    """Dedup rule (section 4): recipe 2's own real trace already fully matched gold -> recipe 1
    would be redundant (recipe 2's version is more realistic: real generated short answers, not
    canonical gold text), skip it."""
    return bool(recipe2_trace.get("all_correct"))
