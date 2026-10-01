#!/usr/bin/env python3
"""
Core single-example recipe-2 trace runner (see update_doc/0907/0907update.md section 3/9):
real hop-by-hop retrieval (cosine top-1, no gate) + real short-answer generation + real
propagation, with post-hoc labeling (committed vs gold) and pair extraction wherever a hop
goes wrong. Not batched yet (step 3/9 scope) -- one example at a time, one generate_batch
call per hop. Batching across examples comes in step 6.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
TRACES_DIR = PROJECT_ROOT / "traces"
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"
BASE_INFER = PROJECT_ROOT.parent / "multihop_trajectory" / "llama_infer_reasoning"
for p in (TRACES_DIR, RETRIEVAL_DIR, BASE_INFER):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from trace_evidence import expand_hop_template  # noqa: E402
from trace_format import escape_double_quotes  # noqa: E402
from run_retrieval_exp import embed_retrieval  # noqa: E402

# BGE's official s2p (short query -> long passage) retrieval instruction: only the QUERY side
# gets this prefix, never the passage/document side (embed_retrieval's own doc encoding is left
# untouched) -- see BAAI/bge-*-en-v1.5 model cards.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
from run_musique_pipeline import prompt_short_answer, judge_answer_official  # noqa: E402
from run_retrieval_exp_wavefront import (  # noqa: E402
    build_final_reader_cot_prompt_comparison_hint,
    parse_final_answer_from_cot,
    batch_last_hidden,
)


def run_recipe2_trace(
    *,
    case_id: str,
    question: str,
    sub_qs: list[str],
    gold_idxs: list[int],
    gold_texts: list[str],
    paragraphs: list[dict[str, Any]],
    generator,
    cos_model: str = "BAAI/bge-base-en-v1.5",
    retrieve_k: int = 10,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    """Run ONE example's full real (baseline, no-gate) hop-by-hop trace. Returns a dict matching
    the traces_v2 schema (minus hidden_state_ref/final_answer_*, added in later steps)."""
    K = len(sub_qs)
    prior_answers: list[str] = []
    hops: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    prefix_text = f"Question: {question.strip()}"

    for j in range(K):
        raw_sq = sub_qs[j]
        expanded_q = expand_hop_template(raw_sq, prior_answers)
        query_for_encode = BGE_QUERY_INSTRUCTION + expanded_q if "bge" in cos_model.lower() else expanded_q
        ranked = embed_retrieval(query_for_encode, paragraphs, retrieve_k, cos_model)
        top3 = [
            {"idx": int(p.get("idx", -1)), "title": p.get("title"), "score": round(float(s), 4)}
            for p, s in ranked[:3]
        ]
        if not ranked:
            hops.append({
                "hop": j + 1, "sub_question_raw": raw_sq, "sub_question_expanded": expanded_q,
                "cosine_top3": [], "committed_idx": None, "gold_idx": gold_idxs[j] if j < len(gold_idxs) else None,
                "is_correct": False, "short_answer_generated": None,
            })
            prior_answers.append("NA")
            continue

        committed_para, committed_score = ranked[0]
        committed_idx = int(committed_para.get("idx", -1))
        gold_idx = gold_idxs[j] if j < len(gold_idxs) else None
        is_correct = committed_idx == gold_idx

        prompt = prompt_short_answer(expanded_q, committed_para)
        raw_answer = generator.generate_batch([prompt], max_new_tokens, desc=f"hop{j + 1}")[0]
        short_answer = raw_answer.strip()

        hop_id = f"{case_id}__recipe2_real__hop{j + 1}"
        hop_row: dict[str, Any] = {
            "hop": j + 1,
            "sub_question_raw": raw_sq,
            "sub_question_expanded": expanded_q,
            "cosine_top3": top3,
            "committed_idx": committed_idx,
            "committed_evidence_text": (committed_para.get("paragraph_text") or "").strip(),
            "gold_idx": gold_idx,
            "is_correct": is_correct,
            "short_answer_generated": short_answer,
        }

        if not is_correct and j < len(gold_texts):
            pair_id = f"{case_id}__recipe2_real__hop{j + 1}"
            hop_row["pair_id"] = pair_id
            pairs.append({
                "pair_id": pair_id,
                "case_id": case_id,
                "trace_id": f"{case_id}__recipe2_real",
                "hop": j + 1,
                "context_prefix": prefix_text,
                "sub_question": expanded_q,
                "evidence_correct": {"idx": gold_idx, "text": gold_texts[j]},
                "evidence_wrong": {"idx": committed_idx, "text": hop_row["committed_evidence_text"]},
            })

        hops.append(hop_row)
        prior_answers.append(short_answer)
        ev_escaped = escape_double_quotes(hop_row["committed_evidence_text"])
        prefix_text += f' Step {j + 1}: {expanded_q} Evidence: "{ev_escaped}"'

    n_correct = sum(1 for h in hops if h["is_correct"])
    return {
        "trace_id": f"{case_id}__recipe2_real",
        "case_id": case_id,
        "recipe": "2_real_baseline",
        "K": K,
        "question": question,
        "hops": hops,
        "pairs": pairs,
        "all_correct": n_correct == K,
        "n_correct": n_correct,
        "is_natural_seed": True,
    }


def add_final_answer(trace: dict[str, Any], gold_record: dict[str, Any], generator,
                      *, max_new_tokens: int = 128) -> dict[str, Any]:
    """Step 4 (see 0907update.md section 9): generate the final answer from THIS trace's own
    committed evidence chain (not gold) using the same production final-reader prompt, then
    score EM/F1 against gold. Mutates and returns `trace` with final_answer_* fields added."""
    hop_steps = [(h["sub_question_expanded"], h.get("committed_evidence_text") or "") for h in trace["hops"]]
    prior = [h.get("short_answer_generated") or "NA" for h in trace["hops"]]
    prompt = build_final_reader_cot_prompt_comparison_hint(trace["question"], hop_steps, prior)
    raw = generator.generate_batch([prompt], max_new_tokens, desc="final_answer")[0]
    predicted = parse_final_answer_from_cot(raw, fallback=raw.strip())
    em, f1 = judge_answer_official(predicted, gold_record)
    trace["final_answer_generated"] = predicted
    trace["final_answer_em"] = bool(em)
    trace["final_answer_f1"] = f1
    return trace


def build_cumulative_prefixes(trace: dict[str, Any]) -> list[str]:
    """h_0..h_K cumulative prefix strings for THIS trace's own committed evidence chain --
    h_0 is "Question: ..." alone, h_j adds hop j's expanded sub-question + committed evidence.
    Reconstructed purely from the trace dict (no extra state needed), same text-assembly
    convention as traces/trace_format.py::assemble_trace / hidden_states/trace_parse.py's
    cumulative_prefix_strings."""
    prefixes = [f"Question: {trace['question'].strip()}"]
    acc = prefixes[0]
    for h in trace["hops"]:
        ev = escape_double_quotes(h.get("committed_evidence_text") or "")
        acc = f'{acc} Step {h["hop"]}: {h["sub_question_expanded"]} Evidence: "{ev}"'
        prefixes.append(acc)
    return prefixes


def add_hidden_states(
    trace: dict[str, Any],
    *,
    gate_model,
    gate_tokenizer,
    layer: int,
    npz_path: Path,
    hidden_batch_size: int = 8,
) -> dict[str, Any]:
    """Step 5 (see 0907update.md section 9): extract RAW (non-PCA) last-token hidden states for
    every prefix in this trace (h_0..h_K) +, for every pair, the h_after of both the correct and
    the wrong evidence at that hop (sharing h_prev = the trace's own h_{hop-1}, no extra
    extraction needed for that half). Saves one npz per trace, sets hidden_state_ref /
    hidden_correct_ref / hidden_wrong_ref pointers into it. PCA is NOT computed here -- stays a
    stage-4 (fit-time) concern, same convention hidden_states/extract_hidden_states.py already
    uses."""
    main_prefixes = build_cumulative_prefixes(trace)  # length K+1

    pair_texts: list[str] = []
    pair_owner: list[tuple[int, str]] = []  # (pair index, "correct"|"wrong")
    for pi, pair in enumerate(trace["pairs"]):
        h_prev_prefix = main_prefixes[pair["hop"] - 1]
        for which in ("correct", "wrong"):
            ev = escape_double_quotes(pair[f"evidence_{which}"]["text"])
            pair_texts.append(f'{h_prev_prefix} Step {pair["hop"]}: {pair["sub_question"]} Evidence: "{ev}"')
            pair_owner.append((pi, which))

    all_texts = main_prefixes + pair_texts
    cache: dict[tuple[int, str], Any] = {}
    hiddens_by_layer = batch_last_hidden(
        texts=all_texts, model=gate_model, tokenizer=gate_tokenizer, layers=[layer],
        batch_size=hidden_batch_size, cache=cache, desc="hidden_states",
    )
    hiddens = hiddens_by_layer[layer]  # list[np.ndarray], same order as all_texts

    import numpy as np

    n_main = len(main_prefixes)
    main_arr = np.stack(hiddens[:n_main], axis=0)  # [K+1, hidden_dim]
    pair_correct_arr = np.zeros((len(trace["pairs"]), main_arr.shape[1]), dtype=main_arr.dtype)
    pair_wrong_arr = np.zeros((len(trace["pairs"]), main_arr.shape[1]), dtype=main_arr.dtype)
    for (pi, which), h in zip(pair_owner, hiddens[n_main:]):
        (pair_correct_arr if which == "correct" else pair_wrong_arr)[pi] = h

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        hidden=main_arr,
        pair_hidden_correct=pair_correct_arr,
        pair_hidden_wrong=pair_wrong_arr,
        pair_ids=np.array([p["pair_id"] for p in trace["pairs"]], dtype=object),
        layer=np.array([layer], dtype=np.int32),
    )

    for h in trace["hops"]:
        h["hidden_state_ref"] = {"npz": npz_path.name, "prefix_idx": h["hop"]}
    for pi, pair in enumerate(trace["pairs"]):
        pair["hidden_correct_ref"] = {"npz": npz_path.name, "pair_idx": pi, "which": "correct"}
        pair["hidden_wrong_ref"] = {"npz": npz_path.name, "pair_idx": pi, "which": "wrong"}
    trace["hidden_states_npz"] = npz_path.name
    return trace
