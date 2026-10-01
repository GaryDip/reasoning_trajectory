#!/usr/bin/env python3
"""
First end-to-end try of the MTL gate's evidence head (task1 only -- same "is this hop's
committed evidence correct" meaning as gate v3's LR score) as a retrieval rerank signal.
Dataset-generic (--dataset musique|2wiki|hotpot, like the rest of retrieval/) -- musique is the
only one the MTL model was actually trained on for now (traces_v2 construction is musique-only
so far), so other datasets are a pure generalization probe, not something scores are expected
to hold up on yet. Compares three methods, real cosine retrieval + real Llama generation for all:

  baseline    -- cosine top-1 every hop, no gate (own internal "gate off" ablation -- NOT the
                 same mechanism as gate_v3_main_method_report.md row A, which goes through an
                 extra LLM passage-select step this pipeline doesn't have; compare mtl_rerank
                 against row D instead, that script has no select step either)
  mtl_rerank  -- cosine top-`--topk` candidates -> MTL evidence head (task1) scores each
                 candidate (real h_prev/h_after/delta from the SAME Llama forward pass style as
                 gate v3's LR scoring) -> rerank via emb_score - lambda*abnormal (reuses
                 retrieval/run_retrieval_exp.py::rerank_by_final_score verbatim) -> commit top-1
  mtl_sum3    -- same candidate pool + hidden states as mtl_rerank, but the FINAL decision
                 ignores cosine entirely: commit whichever candidate maximizes
                 sigmoid(evidence_logit) + sigmoid(hop_answer_logit) + sigmoid(final_f1_logit),
                 the plain sum of all three heads (not just task1, and no lambda/cosine mixing)

Metrics now reuse retrieval/run_retrieval_exp.py's own MetricAccum/ChainAccum/AnswerAccum
classes, so field names (recall@1, recall@3, mrr, chain_recall@1, chain_recall@3, answer_em,
answer_f1) match gate_v3_main_method_report.md's table directly -- recall@1/@3 are full-pool
gold rank (found via find_gold_rank over the whole top-`--topk` candidate pool BEFORE any
rerank/commit decision), not just "did the committed candidate happen to be gold". The one
deliberate thing NOT replicated from run_retrieval_exp_wavefront_gate_v3_rawprefix.py is its
raw/expanded dual-prefix ("rawprefix") machinery -- that exists only because gate v3 was
trained on never-expanded "[Answer N]" placeholder text, so it needs a second un-expanded
accumulated prefix just for scoring. This MTL gate was trained on fully-expanded text
throughout (traces_v2), so there's no train/inference mismatch to work around here -- one
prefix, used for both retrieval-query expansion and gate scoring, is already correct.
Everything else (retrieval, short-answer prompt, final-reader prompt, beam_width=1 top-1
commit) is meant to match that script exactly.

Usage:
  CUDA_VISIBLE_DEVICES=0 python run_e2e.py --dataset musique --limit-per-k 20 --checkpoint checkpoints/mtl_gate.pt
  CUDA_VISIBLE_DEVICES=0 python run_e2e.py --dataset 2wiki --limit-per-k 20 --checkpoint checkpoints/mtl_gate.pt

  # lambda sweep on the TRAIN subset gate v3's own sweep used (never touches dev):
  CUDA_VISIBLE_DEVICES=0 python run_e2e.py --split train \
    --decompose-file ../decompose/bart/data/musique_train_lambda_subset_big_bart_nl.jsonl \
    --lambda-sweep 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
TRACES_DIR = PROJECT_ROOT / "traces"
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"
BASE_INFER = PROJECT_ROOT.parent / "multihop_trajectory" / "llama_infer_reasoning"
for p in (HERE, TRACES_DIR, RETRIEVAL_DIR, BASE_INFER):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from trace_evidence import expand_hop_template, get_ranker  # noqa: E402
from trace_format import escape_double_quotes  # noqa: E402
from run_musique_pipeline import judge_answer_official  # noqa: E402
from run_retrieval_exp import (  # noqa: E402
    K_BUCKETS, MUSIQUE_DIR, TWOWIKI_DEV_FILE, HOTPOT_DEV_FILE, AnswerAccum, ChainAccum,
    MetricAccum, find_gold_rank, load_dataset_records, load_decompose_index,
    normalize_short_answer, prepare_sub_questions, resolve_decompose_file, rerank_by_final_score,
    select_example_ids,
)
from run_retrieval_exp_wavefront import (  # noqa: E402
    BatchVllmGenerator, FINAL_READER_SYSTEM_PROMPT, load_gate_model, warmup_gate_model_memory,
    batch_last_hidden, build_final_reader_cot_prompt, prompt_short_answer_with_context,
    parse_final_answer_from_cot,
)
from model import MTLGateModel  # noqa: E402

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["musique", "2wiki", "hotpot"], default="musique")
    ap.add_argument("--musique-dir", type=Path, default=MUSIQUE_DIR)
    ap.add_argument("--twowiki-file", type=Path, default=TWOWIKI_DEV_FILE)
    ap.add_argument("--hotpot-file", type=Path, default=HOTPOT_DEV_FILE)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--decompose-mode", choices=["gt", "bart_decompose"], default="bart_decompose")
    ap.add_argument("--decompose-file", type=Path, default=None,
                     help="override the templated decompose path -- e.g. point at "
                          "decompose/bart/data/musique_train_lambda_subset_big_bart_nl.jsonl, "
                          "the same 3000-example BART-decompose TRAIN subset gate v3's own "
                          "lambda sweep used, to tune --lambda-mtl without touching dev at all")
    ap.add_argument("--answerable-only", action="store_true", default=True,
                     help="matches production's default -- drops MuSiQue dev cases with no valid gold answer")
    ap.add_argument("--limit-per-k", type=int, default=20)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=10, help="candidate pool size for mtl_rerank")
    ap.add_argument("--hidden-batch-size", type=int, default=8,
                     help="batch size for gate hidden-state forward passes (matches production default)")
    ap.add_argument("--lambda-mtl", type=float, default=0.5, help="matches gate v3's swept optimum on musique")
    ap.add_argument("--lambda-sweep", default=None,
                     help="comma-separated lambda values, e.g. '0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8' "
                          "-- when set, REPLACES --methods with one mtl_rerank_lam{v} variant per "
                          "value (baseline/mtl_sum3 dropped, --lambda-mtl ignored), all sharing one "
                          "model load so the sweep only pays vLLM/gate startup cost once")
    ap.add_argument("--methods", default="baseline,mtl_rerank,mtl_sum3",
                     help="comma-separated subset of baseline,mtl_rerank,mtl_sum3 to run "
                          "(ignored if --lambda-sweep is set)")
    ap.add_argument("--checkpoint", type=Path, default=HERE / "checkpoints/mtl_gate.pt")
    ap.add_argument("--cos-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-new-tokens", type=int, default=64, help="matches production's max_new_tokens_answer")
    ap.add_argument("--out-path", type=Path, default=None,
                     help="defaults to e2e_{dataset}_results.json")
    args = ap.parse_args()
    if args.out_path is None:
        args.out_path = HERE / f"e2e_{args.dataset}_results.json"
    return args


def main() -> None:
    t0 = time.perf_counter()
    args = parse_args()

    decompose_file = resolve_decompose_file(args.decompose_mode, args.split, args.decompose_file, args.dataset)
    # load_dataset_records dispatches on args.dataset and reads whichever of
    # musique_dir/twowiki_file/hotpot_file that dataset needs -- pass args itself (it now
    # carries all three, added above) rather than a hand-built Namespace missing the other two.
    records = load_dataset_records(args)
    decompose_idx = load_decompose_index(decompose_file)
    ids = sorted(set(records) & set(decompose_idx))
    if args.answerable_only:
        ids = [eid for eid in ids if records[eid].get("answerable", True)]
    ids = select_example_ids(ids, decompose_idx, limit=0, limit_per_k=args.limit_per_k, seed=args.sample_seed)
    print(f"{len(ids)} {args.dataset} {args.split} examples selected "
          f"(decompose={args.decompose_mode}, answerable_only={args.answerable_only})")

    cases = []
    for eid in ids:
        sub_qs = prepare_sub_questions(decompose_idx[eid], decompose_mode=args.decompose_mode)
        if not sub_qs:
            continue
        record = records[eid]
        gold_decomp = record.get("question_decomposition") or []
        gold_idxs = []
        for gj in range(len(sub_qs)):
            if gj < len(gold_decomp):
                try:
                    gold_idxs.append(int(gold_decomp[gj].get("paragraph_support_idx")))
                except (TypeError, ValueError):
                    gold_idxs.append(None)
            else:
                gold_idxs.append(None)
        cases.append({
            "case_id": eid, "question": record.get("question", ""), "sub_qs": sub_qs,
            "gold_idxs": gold_idxs, "paragraphs": record.get("paragraphs") or [], "record": record,
        })
    print(f"{len(cases)} cases with usable decompose")

    print("Loading gate model (HF, for hidden states) ...")
    gate_args = argparse.Namespace(model=args.model, dtype="bfloat16", gate_device="cuda:0", attn_implementation=None)
    gate_model, gate_tokenizer = load_gate_model(gate_args)
    warmup_gate_model_memory(gate_model, gate_tokenizer, batch_size=8)

    print(f"Loading MTL gate checkpoint {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    mtl_model = MTLGateModel(
        input_dim=ckpt["input_dim"],
        trunk_hidden=ckpt.get("trunk_hidden", 512),
        trunk_out=ckpt.get("trunk_out", 128),
        head_hidden=ckpt.get("head_hidden", 64),
    )
    mtl_model.load_state_dict(ckpt["model_state_dict"])
    mtl_model.eval()

    print("Loading vLLM generator ...")
    gen = BatchVllmGenerator(
        args.model, dtype="bfloat16", tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
    )
    ranker = get_ranker(args.cos_model)

    method_lambda: dict[str, float] = {}
    if args.lambda_sweep:
        lam_values = [float(v) for v in args.lambda_sweep.split(",")]
        methods = [f"mtl_rerank_lam{v}" for v in lam_values]
        method_lambda = dict(zip(methods, lam_values))
        print(f"lambda sweep: {lam_values}")
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        method_lambda = {m: args.lambda_mtl for m in methods if m == "mtl_rerank" or m.startswith("mtl_rerank_")}
    K_list = [len(c["sub_qs"]) for c in cases]
    max_hops = max(K_list) if K_list else 0
    n = len(cases)

    # per-method running state
    prior_answers = {m: [[] for _ in cases] for m in methods}
    hop_evidence_texts = {m: [[] for _ in cases] for m in methods}
    hop_subqs = {m: [[] for _ in cases] for m in methods}  # expanded sub-question actually used at each hop
    prefix_texts = {m: [f"Question: {c['question'].strip()}" for c in cases] for m in methods}
    gold_ranks = {m: [[] for _ in cases] for m in methods}  # per-case list of int|None (find_gold_rank), one per hop

    doc_embs = {}
    flat_docs, spans = [], []
    for c in cases:
        s = len(flat_docs)
        flat_docs.extend(f"{p.get('title') or ''}\n{(p.get('paragraph_text') or '').strip()}" for p in c["paragraphs"])
        spans.append((s, len(flat_docs)))
    flat_doc_emb = ranker.encode_texts(flat_docs) if flat_docs else np.zeros((0, 768))
    doc_embs = [flat_doc_emb[s:e] for s, e in spans]

    for j in range(max_hops):
        active = [i for i in range(n) if j < K_list[i]]
        if not active:
            break
        print(f"\n=== hop {j + 1}/{max_hops}: {len(active)} active cases ===", flush=True)

        for method in methods:
            expanded_qs = {i: expand_hop_template(cases[i]["sub_qs"][j], prior_answers[method][i]) for i in active}
            q_texts = [BGE_QUERY_INSTRUCTION + expanded_qs[i] for i in active]
            q_emb = ranker.encode_texts(q_texts)

            # Phase 1: pure cosine ranking per case (cheap, numpy only) -- no gate work yet.
            ranked_by_i: dict[int, list[tuple[dict, float]]] = {}
            for local_i, i in enumerate(active):
                paragraphs = cases[i]["paragraphs"]
                doc_emb = doc_embs[i]
                if doc_emb.shape[0] == 0:
                    ranked_by_i[i] = []
                    continue
                sim = doc_emb @ q_emb[local_i]
                # always fetch the full --topk pool for BOTH methods (matches production always
                # retrieving retrieve_k candidates regardless of method) -- baseline still only
                # ever COMMITS the top-1 by raw cosine, but recall@3 needs the full pool to check
                # "is gold anywhere in the top 3", not just whether the single committed pick was it.
                order = np.argsort(-sim)[:args.topk]
                ranked_by_i[i] = [(paragraphs[int(o)], float(sim[int(o)])) for o in order]

            committed = {}
            full_ranked_by_i: dict[int, list[tuple[dict, float]]] = {}
            if method == "baseline":
                for i in active:
                    ranked = ranked_by_i[i]
                    committed[i] = ranked[0] if ranked else None
                    full_ranked_by_i[i] = ranked
            else:  # mtl_rerank / mtl_sum3 -- score ALL candidates from ALL active cases in ONE
                   # batched Llama forward pass + ONE batched MTL forward pass, not one tiny call
                   # per case (that was the original bug: ~6000 individual batch_last_hidden
                   # calls across a full run instead of ~4, each paying full per-call overhead
                   # for a forward pass over just ~11 texts -- this is what made the run take 3+
                   # hours instead of the ~30min a properly hop-batched pass takes). Both mtl_*
                   # methods share this identical hidden-extraction + MTL-forward block -- they
                   # only differ in how the three heads' outputs get combined into one score
                   # below (lambda-mixed-with-cosine vs plain sum of all three heads).
                   all_texts: list[str] = []
                   owners: list[tuple[int, int | None]] = []  # (case i, cand_idx or None=h_prev)
                   for i in active:
                       if not ranked_by_i[i]:
                           continue
                       all_texts.append(prefix_texts[method][i])
                       owners.append((i, None))
                       for cand_idx, (para, _) in enumerate(ranked_by_i[i]):
                           ev = escape_double_quotes((para.get("paragraph_text") or "").strip())
                           all_texts.append(f'{prefix_texts[method][i]} Step {j + 1}: {expanded_qs[i]} Evidence: "{ev}"')
                           owners.append((i, cand_idx))

                   hiddens = batch_last_hidden(
                       texts=all_texts, model=gate_model, tokenizer=gate_tokenizer, layers=[args.layer],
                       batch_size=args.hidden_batch_size, cache={}, desc=f"hop{j + 1} gate",
                   )[args.layer] if all_texts else []

                   h_prev_by_i: dict[int, np.ndarray] = {}
                   h_cand_by_i: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
                   for (i, cand_idx), h in zip(owners, hiddens):
                       if cand_idx is None:
                           h_prev_by_i[i] = h
                       else:
                           h_cand_by_i[i][cand_idx] = h

                   # one batched MTL forward over every (case, candidate) pair this hop -- all
                   # three heads computed together, not just evidence_logit.
                   x_rows, x_owners = [], []
                   for i in active:
                       if i not in h_prev_by_i:
                           continue
                       h_prev = h_prev_by_i[i]
                       for cand_idx in range(len(ranked_by_i[i])):
                           h_j = h_cand_by_i[i][cand_idx]
                           delta = h_j - h_prev
                           x_rows.append(np.concatenate([h_j, delta]))
                           x_owners.append((i, cand_idx))
                   if x_rows:
                       x = torch.from_numpy(np.stack(x_rows)).float().unsqueeze(1)  # [N, 1, D]
                       with torch.no_grad():
                           out = mtl_model(x)
                       p_evidence = torch.sigmoid(out["evidence_logit"]).squeeze(-1).numpy()      # [N]
                       p_hop_answer = torch.sigmoid(out["hop_answer_logit"]).squeeze(-1).numpy()  # [N]
                       p_final_f1 = torch.sigmoid(out["final_f1_logit"]).squeeze(-1).numpy()      # [N]
                   else:
                       p_evidence = p_hop_answer = p_final_f1 = np.zeros(0)

                   if method in method_lambda:  # "mtl_rerank" or a "mtl_rerank_lam{v}" sweep variant
                       # cosine emb_score mixed with the evidence head only, via
                       # rerank_by_final_score (emb_score - lambda*abnormal) -- same formula
                       # gate v3 uses, just a different model producing "abnormal". Each method's
                       # own lambda comes from method_lambda (all equal to --lambda-mtl outside
                       # a sweep; one distinct value per method when --lambda-sweep is set).
                       lam = method_lambda[method]
                       scored_by_i: dict[int, list[tuple[dict, float, float]]] = defaultdict(list)
                       for (i, cand_idx), pc in zip(x_owners, p_evidence):
                           para, emb_score = ranked_by_i[i][cand_idx]
                           scored_by_i[i].append((para, emb_score, 1.0 - float(pc)))
                       for i in active:
                           if not ranked_by_i[i] or i not in scored_by_i:
                               committed[i] = None
                               full_ranked_by_i[i] = []
                               continue
                           reranked = rerank_by_final_score(scored_by_i[i], lam)
                           committed[i] = reranked[0]
                           full_ranked_by_i[i] = reranked
                   else:  # mtl_sum3 -- no cosine at all in the final decision, just the sum of
                          # all three heads' probabilities (cosine is still used to build the
                          # candidate POOL in phase 1 above, since some initial retrieval step is
                          # needed to narrow a paragraph pool down to something scoreable -- but
                          # which candidate WINS is decided purely by the gate now).
                       sum3_by_i: dict[int, list[tuple[dict, float]]] = defaultdict(list)
                       for (i, cand_idx), pe, pa, pf in zip(x_owners, p_evidence, p_hop_answer, p_final_f1):
                           para, _emb_score = ranked_by_i[i][cand_idx]
                           sum3_by_i[i].append((para, float(pe) + float(pa) + float(pf)))
                       for i in active:
                           if not ranked_by_i[i] or i not in sum3_by_i:
                               committed[i] = None
                               full_ranked_by_i[i] = []
                               continue
                           ranked_sum3 = sorted(sum3_by_i[i], key=lambda t: -t[1])
                           committed[i] = ranked_sum3[0]
                           full_ranked_by_i[i] = ranked_sum3

            # find_gold_rank over the FULL pool (ranked/reranked, before any top-1 commit
            # decision) -- matches production's full_pool_gold_rank recall@1/@3 definition
            # exactly (with beam_width=1, "top-1 in the full pool" and "the committed pick" are
            # the same thing, so this is consistent with `committed` above, just also captures
            # recall@3 which a single committed pick alone can't).
            for i in active:
                gold_idx = cases[i]["gold_idxs"][j] if j < len(cases[i]["gold_idxs"]) else None
                rank = find_gold_rank(full_ranked_by_i.get(i, []), gold_idx) if gold_idx is not None else None
                gold_ranks[method][i].append(rank)

            # prompt_short_answer_with_context matches production's default short-answer prompt
            # builder -- unlike the plain prompt_short_answer (passage + question only, no prior
            # reasoning), this feeds the model the full reasoning trace so far (hop_subqs/
            # hop_evidence_texts/prior_answers up to but not including the current hop), which
            # is what the gate_v3_main_method_report.md numbers were actually produced with.
            prompts, owners = [], []
            for i in active:
                if committed[i] is None:
                    continue
                hop_steps_before = list(zip(hop_subqs[method][i], hop_evidence_texts[method][i]))
                prompts.append(prompt_short_answer_with_context(
                    cases[i]["question"], hop_steps_before, prior_answers[method][i],
                    expanded_qs[i], committed[i][0],
                ))
                owners.append(i)
            raw_answers = gen.generate_batch(prompts, args.max_new_tokens, desc=f"hop{j + 1}_{method}") if prompts else []
            # normalize_short_answer (not just .strip()) matches production: trims to the first
            # line and maps empty/"no information in passage"-style non-answers to "NA" before
            # this gets propagated into the next hop's [Answer N] expansion.
            answer_by_i = {i: normalize_short_answer(a) for i, a in zip(owners, raw_answers)}

            for i in active:
                if committed[i] is None:
                    prior_answers[method][i].append("NA")
                    hop_evidence_texts[method][i].append("")
                    hop_subqs[method][i].append(expanded_qs[i])
                    continue
                para, _ = committed[i]
                short_answer = answer_by_i.get(i, "NA")
                prior_answers[method][i].append(short_answer)
                hop_evidence_texts[method][i].append((para.get("paragraph_text") or "").strip())
                hop_subqs[method][i].append(expanded_qs[i])
                ev = escape_double_quotes((para.get("paragraph_text") or "").strip())
                prefix_texts[method][i] += f' Step {j + 1}: {expanded_qs[i]} Evidence: "{ev}"'

    # final answers
    print("\n=== final answers ===", flush=True)
    final_results = {m: {} for m in methods}
    for method in methods:
        prompts = []
        for i, c in enumerate(cases):
            # hop_subqs/hop_evidence_texts were tracked incrementally during the hop loop above
            # (same lists prompt_short_answer_with_context read from, one entry per completed
            # hop) -- reuse them directly instead of recomputing via expand_hop_template.
            hop_steps = list(zip(hop_subqs[method][i], hop_evidence_texts[method][i]))
            prompts.append(build_final_reader_cot_prompt(
                c["question"], hop_steps, prior_answers[method][i],
            ))
        # generate_chat_batch + FINAL_READER_SYSTEM_PROMPT matches production (it does NOT use
        # plain generate_batch for the final reader -- the system prompt is a separate chat
        # message, not folded into the user prompt).
        raw = gen.generate_chat_batch(FINAL_READER_SYSTEM_PROMPT, prompts, 128, desc=f"final_{method}")
        for i, c in enumerate(cases):
            # fallback matches production: the last hop's own short answer, not the raw
            # (possibly off-format) final-reader text stripped.
            fallback = prior_answers[method][i][-1] if prior_answers[method][i] else ""
            predicted = parse_final_answer_from_cot(raw[i], fallback=fallback)
            em, f1 = judge_answer_official(predicted, c["record"])
            final_results[method][c["case_id"]] = {"predicted": predicted, "em": bool(em), "f1": f1}

    # aggregate metrics, overall + per K -- reusing run_retrieval_exp.py's own accumulator
    # classes so field names match gate_v3_main_method_report.md's table directly.
    def agg(method: str, ks: list[int] | None = None):
        idxs = [i for i in range(n) if ks is None or K_list[i] in ks]
        if not idxs:
            return None
        hop_acc, chain_acc, ans_acc = MetricAccum(), ChainAccum(), AnswerAccum()
        for i in idxs:
            ranks = gold_ranks[method][i]
            for r in ranks:
                hop_acc.update(r)
            chain_acc.update(ranks)
            r = final_results[method][cases[i]["case_id"]]
            ans_acc.update(int(r["em"]), r["f1"])
        return {"n_cases": len(idxs), **hop_acc.result(), **chain_acc.result(), **ans_acc.result()}

    report = {}
    for method in methods:
        report[method] = {"overall": agg(method)}
        for k in K_BUCKETS:
            report[method][f"K={k}"] = agg(method, [k])

    args.out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote -> {args.out_path}")
    print(f"\n{'method':<12} {'view':<8} {'n':>4} {'recall@1':>9} {'recall@3':>9} "
          f"{'chain@1':>8} {'chain@3':>8} {'EM':>7} {'F1':>7}")
    for method in methods:
        for view, m in report[method].items():
            if m is None:
                continue
            print(f"{method:<12} {view:<8} {m['n_cases']:>4} {m['recall@1']:>9.3f} {m['recall@3']:>9.3f} "
                  f"{m['chain_recall@1']:>8.3f} {m['chain_recall@3']:>8.3f} "
                  f"{m['answer_em']:>7.3f} {m['answer_f1']:>7.3f}")
    print(f"\ntotal wall clock: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
