#!/usr/bin/env python3
"""Measure, with the real tokenizer and the real prompt builders, every token count the FLOPs
estimate needs. CPU only, ~2 minutes. Writes cost/results/measured_tokens.json.

This replaces the assumed constants the first version of `estimate_flops.py` used. What is still
assumed afterwards is listed under "assumed" in the output, with the reason.

The gate figure is the important one: it reconstructs, for each hop, the prefix text and all ten
candidate texts exactly as run_retrieval_exp_wavefront_gate_v3_rawprefix.py builds them (the
candidate lists are in the run's case log), wraps them in the same chat template, and counts the
forwards the way the code actually does -- i.e. honouring `hidden_cache`, which makes every
prefix after hop 1 free because the next hop's prefix is byte-identical to the chosen candidate's
text at this hop. That gives 1 + sum|C_j| forwards, not K*(1+|C_j|).
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "retrieval"))
sys.path.insert(0, str(ROOT / "traces"))

MAIN_RUN = "20261008_071635_l15_musique_lambda0p60_dev"
KDIST = {2: 1252, 3: 760, 4: 405}
POOL = 10


def load_cases(run: str) -> list[dict]:
    f = glob.glob(str(ROOT / f"retrieval/results/{run}/retrieval_cases_*.jsonl"))[0]
    return [json.loads(l) for l in open(f) if l.strip()]


def chainrag_records(ds: str) -> list[dict]:
    """results.jsonl is a stream of pretty-printed objects, not one per line."""
    txt = (ROOT / f"chainrag/processed_data/{ds}/results.jsonl").read_text()
    dec, out, i = json.JSONDecoder(), [], 0
    while i < len(txt):
        while i < len(txt) and txt[i] in " \n\r\t":
            i += 1
        if i >= len(txt):
            break
        o, i = dec.raw_decode(txt, i)
        out.append(o)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-k", type=int, default=60, help="questions per K for the gate scan")
    ap.add_argument("--out", type=Path, default=ROOT / "cost/results/measured_tokens.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from run_retrieval_exp import build_trace_prefix, escape_double_quotes
    from run_retrieval_exp_wavefront import (
        FINAL_READER_SYSTEM_PROMPT, build_final_reader_cot_prompt_comparison_hint,
        prompt_short_answer_with_context, render_gate_chat_text,
    )

    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", use_fast=True)
    n = lambda x: len(tok(x, add_special_tokens=False)["input_ids"])          # noqa: E731
    n_gate = lambda x: n(render_gate_chat_text(tok, x))                        # noqa: E731

    rows = {r["id"]: r for r in map(json.loads,
                                    open(ROOT / "data/raw/musique/musique_ans_v1.0_dev.jsonl"))}
    subq = {str(json.loads(l)["id"]): json.loads(l)["sub_questions"]
            for l in open(ROOT / "data/decompose/musique/bart/dev_nl.jsonl")}
    cases = load_cases(MAIN_RUN)

    out: dict = {"source": {"run": MAIN_RUN, "tokenizer": "meta-llama/Llama-3.1-8B-Instruct"}}

    # ---- plain text lengths -------------------------------------------------------------
    paras = [p["paragraph_text"] for r in list(rows.values())[:400] for p in r["paragraphs"]]
    out["passage_tokens"] = round(st.mean(map(n, paras[:3000])), 1)
    out["question_tokens"] = round(st.mean(map(n, [r["question"] for r in
                                                   list(rows.values())[:800]])), 1)
    flat_sq = [s for v in list(subq.values())[:1200] for s in v]
    out["sub_question_tokens"] = round(st.mean(map(n, flat_sq[:3000])), 1)

    # ---- the two generation prompts, built by the real builders --------------------------
    hop_p, fin_p = [], []
    sys_len = n(FINAL_READER_SYSTEM_PROMPT)
    for c in cases[:300]:
        r, subs = rows.get(c["id"]), subq.get(c["id"]) or []
        if not r or not subs:
            continue
        paras_by_idx = {int(p["idx"]): p for p in r["paragraphs"]}
        ev = [paras_by_idx[p].get("paragraph_text", "") for p in c["para_ids"] if p in paras_by_idx]
        steps, prior = list(zip(subs, ev)), c.get("prior") or []
        for j, pid in enumerate(c["para_ids"]):
            if pid in paras_by_idx and j < len(subs):
                hop_p.append(n(prompt_short_answer_with_context(
                    r["question"], steps[:j], prior[:j], subs[j], paras_by_idx[pid])))
        fin_p.append(sys_len + n(build_final_reader_cot_prompt_comparison_hint(
            r["question"], steps, prior)))
    out["final_reader_system_tokens"] = sys_len
    out["hop_answer_prompt_tokens"] = round(st.mean(hop_p), 1)
    out["final_reader_prompt_tokens"] = round(st.mean(fin_p), 1)

    # ---- the gate: exact texts, exact forward count --------------------------------------
    by_k: dict[int, list[tuple[int, int]]] = {}
    for c in cases:
        r, subs, ranked = rows.get(c["id"]), subq.get(c["id"]) or [], c.get("hop_ranked_para_ids")
        if not r or not subs or not ranked:
            continue
        K = len(c["para_ids"])
        if len(by_k.get(K, [])) >= args.per_k:
            continue
        paras_by_idx = {int(p["idx"]): p for p in r["paragraphs"]}
        steps, tot, fwd = [], 0, 0
        for j in range(K):
            pre = build_trace_prefix(r["question"], steps)
            if j == 0:                       # only hop 1's prefix misses the cache
                tot += n_gate(pre); fwd += 1
            for pid in ranked[j][:POOL]:
                p = paras_by_idx.get(pid)
                if p is None:
                    continue
                ev_t = (p.get("paragraph_text") or "").strip()
                sq_j = subs[j] if j < len(subs) else ""
                tot += n_gate(f'{pre} Step {j + 1}: {sq_j} Evidence: '
                              f'"{escape_double_quotes(ev_t)}"')
                fwd += 1
            pid = c["para_ids"][j]
            if pid in paras_by_idx and j < len(subs):
                steps.append((subs[j], (paras_by_idx[pid].get("paragraph_text") or "").strip()))
        by_k.setdefault(K, []).append((tot, fwd))
    w = sum(KDIST.values())
    out["gate_by_K"] = {str(K): {"n_sampled": len(v),
                                 "tokens": round(st.mean(x[0] for x in v), 1),
                                 "forwards": round(st.mean(x[1] for x in v), 1)}
                        for K, v in sorted(by_k.items())}
    out["gate_tokens"] = round(sum(st.mean(x[0] for x in by_k[K]) * c_ / w
                                   for K, c_ in KDIST.items() if K in by_k), 1)
    out["gate_forwards"] = round(sum(st.mean(x[1] for x in by_k[K]) * c_ / w
                                     for K, c_ in KDIST.items() if K in by_k), 2)

    # ---- ChainRAG: its own per-question log already counts its LLM calls -----------------
    out["chainrag"] = {}
    for ds in ("musique", "2wikimqa", "hotpotqa"):
        recs = chainrag_records(ds)
        out["chainrag"][ds] = {
            "n": len(recs),
            "llm_calls_mean": round(st.mean(r["llm_calls"] for r in recs), 2),
            "llm_calls_median": st.median(r["llm_calls"] for r in recs),
            "llm_calls_range": [min(r["llm_calls"] for r in recs),
                                max(r["llm_calls"] for r in recs)],
            "sub_questions_mean": round(st.mean(len(r["sub_questions"]) for r in recs), 2),
            "context_sentences_mean": round(st.mean(r["total_context_sentences"] for r in recs), 1),
            "rewrite_rate": round(sum(1 for r in recs for s in r["retrieved_titles_by_subq"]
                                      if s["sub_question"] != s["modified_sub_question"])
                                  / sum(len(r["sub_questions"]) for r in recs), 4)}

    out["assumed"] = {
        "hop_answer_decode_tokens": [8, "案例日志只存解析后的短答案（均值 3.3 token），"
                                        "真实解码长度未落盘；上限 64。解码项占总量 <2%"],
        "final_reader_decode_tokens": [96, "最终 reader 是 CoT，日志只存解析出的答案"
                                           "（均值 3.6 token）；上限 128"],
        "chainrag_prompt_tokens": [700, "ChainRAG 不落盘 prompt。按每次调用携带约 20 句"
                                        "（每句约 30 token）+ 指令估算；它每题上下文共约 "
                                        "70 句（已实测）"],
        "chainrag_sentence_tokens": [30, "同上，未落盘"],
        "chainrag_rerank_pairs_per_hop": [50, "未落盘"],
        "grithopper_answer_decode_tokens": [96, "与我们的最终 reader 同口径"],
    }
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
