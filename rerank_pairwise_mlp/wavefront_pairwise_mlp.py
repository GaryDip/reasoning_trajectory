#!/usr/bin/env python3
"""
Production-wavefront-equivalent evaluation for the pairwise-MLP reranker
(train_pairwise_mlp.py's artifacts) — same hop loop, same accumulator
classes, same output JSON/jsonl shape as
retrieval/run_retrieval_exp_wavefront.py, so results are directly comparable
to the existing gated_rule_a / lr_rerank / beam_search numbers in
update_doc/0713/0713update.md. NOT a lighter oracle-prior probe like this
folder's eval_retrieval.py — real vLLM passage selection, real hop-answer
generation, real final-answer generation, the whole chain.

The only thing that changes vs production wavefront is the reranking
formula: instead of score_gate_requests's PCA+LR + emb_score - lambda*abnormal,
each retrieved candidate is scored directly by the trained pairwise MLP on
[emb_score, PCA(Delta)] (train_pairwise_mlp.py's artifacts) and ranked by
that score alone.

Reuses (imports, does not copy) run_retrieval_exp.py's dataset loading,
accumulator classes, and run_retrieval_exp_wavefront.py's BatchVllmGenerator,
ExampleState/MethodState/init_example_state, prompt builders, batch_last_hidden,
load_gate_model — neither production script is modified. Lives here (a new
file in rerank_pairwise_mlp/), not in retrieval/.

Usage (needs a real GPU + Llama checkpoint + vLLM):
  python wavefront_pairwise_mlp.py --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"
TRACES_DIR = PROJECT_ROOT / "traces"
sys.path.insert(0, str(RETRIEVAL_DIR))
sys.path.insert(0, str(TRACES_DIR))

from run_retrieval_exp import (  # noqa: E402
    K_BUCKETS,
    MUSIQUE_DIR,
    AnswerAccum,
    CaseRecallAccum,
    ChainAccum,
    MetricAccum,
    SelectionAccum,
    SelectionChainAccum,
    _get_pipeline_helpers,
    build_trace_prefix,
    case_recall_flags,
    embed_retrieval,
    find_gold_rank,
    gold_hop_ranks,
    gold_hop_selections,
    load_dataset_records,
    load_decompose_index,
    prepare_sub_questions,
    resolve_decompose_file,
    resolve_output_dir,
    sanitize_run_tag,
    select_example_ids,
)
from run_retrieval_exp_wavefront import (  # noqa: E402
    FINAL_READER_SYSTEM_PROMPT,
    BatchVllmGenerator,
    batch_last_hidden,
    build_final_reader_cot_prompt,
    init_example_state,
    load_gate_model,
    parse_final_answer_from_cot,
    prompt_select_passage_with_context,
    prompt_short_answer_with_context,
)
from trace_format import escape_double_quotes  # noqa: E402

METHOD = "pairwise_mlp"


class MLPScorer:
    def __init__(self, artifacts_dir: Path):
        import joblib
        import torch
        from torch import nn

        meta = json.loads((artifacts_dir / "meta.json").read_text())
        self.pca = joblib.load(artifacts_dir / "pca.joblib")
        ckpt = torch.load(artifacts_dir / "model.pt", map_location="cpu", weights_only=False)

        class _Net(nn.Module):
            def __init__(self, in_dim, hidden_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )

            def forward(self, x):
                return self.net(x).squeeze(-1)

        self.model = _Net(ckpt["in_dim"], ckpt["hidden_dim"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.torch = torch
        self.layer = meta.get("layer", 31)
        self.hop_features = meta.get("hop_features", False)
        self.num_j_classes = meta.get("num_j_classes", 4)

    def score(self, emb_scores: np.ndarray, deltas: np.ndarray, *,
              K: int | None = None, j: int | None = None) -> np.ndarray:
        z = self.pca.transform(deltas.astype(np.float64))
        parts = [emb_scores.reshape(-1, 1), z]
        if self.hop_features:
            if K is None or j is None:
                raise ValueError("This artifact was trained with hop features; pass K and j.")
            n = len(emb_scores)
            onehot = np.zeros((n, self.num_j_classes), dtype=np.float32)
            onehot[:, min(max(j, 0), self.num_j_classes - 1)] = 1.0
            k_col = np.full((n, 1), float(K), dtype=np.float32)
            parts.extend([onehot, k_col])
        X = np.concatenate(parts, axis=1).astype(np.float32)
        with self.torch.no_grad():
            return self.model(self.torch.from_numpy(X)).numpy()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["musique"], default="musique")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--musique-dir", type=Path, default=MUSIQUE_DIR)
    ap.add_argument("--decompose-mode", choices=["gt", "bart_decompose"], default="bart_decompose")
    ap.add_argument("--decompose-file", type=Path, default=None)
    ap.add_argument("--artifacts-dir", type=Path, default=HERE / "artifacts")
    ap.add_argument("--results-root", type=Path, default=HERE / "results")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--out-cases", type=Path, default=None)
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--retrieve-k", type=int, default=10)
    ap.add_argument("--cos-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--answerable-only", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--limit-per-k", type=int, default=0)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--gate-device", default="cuda:0")
    ap.add_argument("--hidden-batch-size", type=int, default=16)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-new-tokens-select", type=int, default=64)
    ap.add_argument("--max-new-tokens-answer", type=int, default=64)
    ap.add_argument("--max-new-tokens-final", type=int, default=128)
    ap.add_argument("--final-reader", action="store_true", default=True)
    return ap.parse_args()


def _make_accum(cls):
    return {"all": cls(), **{K: cls() for K in K_BUCKETS}}


def main() -> None:
    args = parse_args()
    args.decompose_file = resolve_decompose_file(
        args.decompose_mode, args.split, args.decompose_file, args.dataset
    )

    ds_args = argparse.Namespace(dataset=args.dataset, musique_dir=args.musique_dir, split=args.split)
    records = load_dataset_records(ds_args)
    decompose_idx = load_decompose_index(args.decompose_file)
    ids = sorted(set(records) & set(decompose_idx))
    if args.answerable_only:
        ids = [eid for eid in ids if records[eid].get("answerable", True)]
    ids = select_example_ids(ids, decompose_idx, limit=args.limit, limit_per_k=args.limit_per_k,
                              seed=args.sample_seed)
    if args.num_shards > 1:
        ids = ids[args.shard_index::args.num_shards]

    tag = args.run_tag or f"wavefront_pairwise_mlp_{args.dataset}"
    args.out_dir = resolve_output_dir(
        out_dir=args.out_dir, results_root=args.results_root, run_tag=tag,
        decompose_mode=args.decompose_mode, methods=[METHOD],
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
    if args.out_cases is None:
        args.out_cases = args.out_dir / f"retrieval_cases_{args.split}{tag_suffix}.jsonl"

    (expand_hop_template, _a, parse_json_choice, _c, _d, _e, judge_answer_official) = _get_pipeline_helpers()

    print(f"Output directory: {args.out_dir}")
    print(f"Examples: {len(ids)}  decompose={args.decompose_file}  method={METHOD}")

    examples = []
    for eid in ids:
        row = records[eid]
        sub_qs = prepare_sub_questions(decompose_idx.get(eid) or [], decompose_mode=args.decompose_mode)
        if not sub_qs:
            continue
        examples.append(init_example_state(eid, row, sub_qs, [METHOD], []))

    scorer = MLPScorer(args.artifacts_dir)
    print(f"Loaded scorer from {args.artifacts_dir} (layer={scorer.layer})")

    print("Loading gate-style Llama model (Delta extraction) ...", flush=True)
    gate_model, gate_tokenizer = load_gate_model(args)
    print("Loading vLLM generator ...", flush=True)
    generator = BatchVllmGenerator(
        args.model, dtype=args.dtype, tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
    )

    accum = _make_accum(MetricAccum)
    chain_accum = _make_accum(ChainAccum)
    chain_accum_k_match = _make_accum(ChainAccum)
    case_recall_accum = _make_accum(CaseRecallAccum)
    sel_accum = _make_accum(SelectionAccum)
    sel_chain_accum = _make_accum(SelectionChainAccum)
    sel_chain_accum_k_match = _make_accum(SelectionChainAccum)
    ans_accum = _make_accum(AnswerAccum)

    hidden_cache: dict[tuple[int, str], np.ndarray] = {}
    max_hops = max((ex.K for ex in examples), default=0)
    run_t0 = time.perf_counter()

    for hop_j in range(1, max_hops + 1):
        hop_t0 = time.perf_counter()
        print(f"\n=== hop {hop_j}/{max_hops} ===", flush=True)

        active = [ex for ex in examples if hop_j <= ex.K]
        contexts: list[dict[str, Any]] = []
        for ex in active:
            ms = ex.methods[METHOD]
            raw_sq = ex.sub_questions[hop_j - 1]
            expanded_q = expand_hop_template(raw_sq, ms.prior)
            prefix_before = build_trace_prefix(ex.q_main, ms.hop_steps)
            candidates_all = embed_retrieval(expanded_q, ex.paragraphs, args.retrieve_k, args.cos_model)
            if not candidates_all:
                continue
            contexts.append({
                "ex": ex, "expanded_q": expanded_q, "prefix_before": prefix_before,
                "candidates_all": candidates_all,
            })
        print(f"[hop {hop_j}] {len(contexts)} contexts to score", flush=True)

        # Batched Delta extraction: one prefix_before + one text per candidate, per context.
        texts: list[str] = []
        owners: list[tuple[int, int]] = []  # (ctx_idx, slot); slot=-1 -> prefix_before
        for ci, ctx in enumerate(contexts):
            texts.append(ctx["prefix_before"])
            owners.append((ci, -1))
            for slot, (p, _s) in enumerate(ctx["candidates_all"]):
                text = (
                    f'{ctx["prefix_before"]} Step {hop_j}: {ctx["expanded_q"]}'
                    f' Evidence: "{escape_double_quotes((p.get("paragraph_text") or "").strip())}"'
                )
                texts.append(text)
                owners.append((ci, slot))

        hiddens_by_layer = batch_last_hidden(
            texts=texts, model=gate_model, tokenizer=gate_tokenizer, layers=[scorer.layer],
            batch_size=args.hidden_batch_size, cache=hidden_cache, desc=f"hop{hop_j} delta",
        )
        hiddens = hiddens_by_layer[scorer.layer]
        prefix_hidden: dict[int, np.ndarray] = {}
        cand_hidden: dict[int, dict[int, np.ndarray]] = {}
        for (ci, slot), h in zip(owners, hiddens):
            if slot == -1:
                prefix_hidden[ci] = h
            else:
                cand_hidden.setdefault(ci, {})[slot] = h

        for ci, ctx in enumerate(contexts):
            candidates_all = ctx["candidates_all"]
            deltas = np.stack([
                cand_hidden[ci][slot] - prefix_hidden[ci] for slot in range(len(candidates_all))
            ]).astype(np.float32)
            emb_scores = np.asarray([s for _p, s in candidates_all], dtype=np.float32)
            scores = scorer.score(emb_scores, deltas, K=ctx["ex"].K, j=hop_j - 1)
            ranked = sorted(zip([p for p, _s in candidates_all], scores), key=lambda t: -t[1])
            ctx["top3"] = ranked[:3]

        # Gold-rank bookkeeping.
        for ctx in contexts:
            ex = ctx["ex"]
            hop_row = ex.hop_results[hop_j - 1]
            gold_pi = hop_row.get("gold_para_idx")
            if hop_row.get("has_gold_hop") and gold_pi is not None:
                gold_rank = find_gold_rank(ctx["top3"], int(gold_pi))
                accum["all"].update(gold_rank)
                if ex.K in accum:
                    accum[ex.K].update(gold_rank)
            else:
                gold_rank = None
            ctx["gold_rank"] = gold_rank
            ex.methods[METHOD].ex_ranks.append(gold_rank)

        # Batched vLLM passage selection.
        selection_prompts = [
            prompt_select_passage_with_context(
                ctx["ex"].q_main, ctx["ex"].methods[METHOD].hop_steps,
                ctx["ex"].methods[METHOD].prior, ctx["expanded_q"],
                [p for p, _s in ctx["top3"]],
            )
            for ctx in contexts
        ]
        selection_raw = generator.generate_batch(
            selection_prompts, args.max_new_tokens_select, desc=f"select-hop{hop_j}",
        ) if selection_prompts else []

        answer_prompts: list[str] = []
        answer_meta: list[tuple[dict[str, Any], int, dict[str, Any], bool | None]] = []
        for ctx, raw in zip(contexts, selection_raw):
            top3_paras = [p for p, _s in ctx["top3"]]
            choice = parse_json_choice(raw)
            if choice is None:
                choice = 0
            choice = max(0, min(len(top3_paras) - 1, choice))
            chosen_para = top3_paras[choice]
            ex = ctx["ex"]
            hop_row = ex.hop_results[hop_j - 1]
            gold_pi = hop_row.get("gold_para_idx")
            if hop_row.get("has_gold_hop") and gold_pi is not None:
                sel_correct = int(chosen_para.get("idx", -1)) == int(gold_pi)
                sel_accum["all"].update(sel_correct)
                if ex.K in sel_accum:
                    sel_accum[ex.K].update(sel_correct)
            else:
                sel_correct = None
            ex.methods[METHOD].ex_sel_correct.append(sel_correct)
            answer_meta.append((ctx, choice, chosen_para, sel_correct))
            ms = ex.methods[METHOD]
            answer_prompts.append(
                prompt_short_answer_with_context(ex.q_main, ms.hop_steps, ms.prior, ctx["expanded_q"], chosen_para)
            )

        answer_raw = generator.generate_batch(
            answer_prompts, args.max_new_tokens_answer, desc=f"answer-hop{hop_j}",
        ) if answer_prompts else []

        from run_retrieval_exp import normalize_short_answer
        for (ctx, choice, chosen_para, sel_correct), raw in zip(answer_meta, answer_raw):
            ex = ctx["ex"]
            ms = ex.methods[METHOD]
            sub_ans = normalize_short_answer(raw)
            ev_text = (chosen_para.get("paragraph_text") or "").strip()
            ms.prior.append(sub_ans)
            ms.hop_steps.append((ctx["expanded_q"], ev_text))
            ex.hop_results[hop_j - 1][METHOD] = {
                "expanded_subq": ctx["expanded_q"],
                "top3_para_ids": [int(p.get("idx", -1)) for p, _s in ctx["top3"]],
                "top3_scores": [round(float(s), 4) for _p, s in ctx["top3"]],
                "gold_rank": ctx["gold_rank"],
                "gold_in_top3": ctx["gold_rank"] is not None and ctx["gold_rank"] <= 3,
                "selection": {"mode": "select", "choice": choice,
                              "chosen_para_id": int(chosen_para.get("idx", -1)), "correct": sel_correct},
                "sub_answer": sub_ans,
            }

        print(f"[hop {hop_j}] done, elapsed {(time.perf_counter() - hop_t0) / 60:.1f} min", flush=True)

    if args.final_reader:
        print("\n=== Final reader ===", flush=True)
        final_tasks = [ex for ex in examples if ex.methods[METHOD].hop_steps]
        final_prompts = [
            build_final_reader_cot_prompt(ex.q_main, ex.methods[METHOD].hop_steps, ex.methods[METHOD].prior)
            for ex in final_tasks
        ]
        final_raw = generator.generate_chat_batch(
            FINAL_READER_SYSTEM_PROMPT, final_prompts, args.max_new_tokens_final, desc="final-reader",
        )
        for ex, raw in zip(final_tasks, final_raw):
            ms = ex.methods[METHOD]
            ms.final_cot = raw
            fallback = ms.prior[-1] if ms.prior else ""
            ms.final_answer = parse_final_answer_from_cot(raw, fallback=fallback)

    # Final per-example metrics + case writing.
    args.out_cases.parent.mkdir(parents=True, exist_ok=True)
    cases_fp = args.out_cases.open("w", encoding="utf-8")
    n_examples = n_k_match = 0
    for ex in examples:
        n_examples += 1
        if ex.k_match:
            n_k_match += 1
        ms = ex.methods[METHOD]
        gold_ranks = gold_hop_ranks(ex.hop_results, METHOD, ex.K_gold)
        gold_sels = gold_hop_selections(ex.hop_results, METHOD, ex.K_gold)
        case_recall_accum["all"].update(gold_ranks)
        if ex.K_gold in case_recall_accum:
            case_recall_accum[ex.K_gold].update(gold_ranks)
        cr1, cr3 = case_recall_flags(gold_ranks)

        if ex.k_match:
            chain_accum_k_match["all"].update(gold_ranks)
            if ex.K in chain_accum_k_match:
                chain_accum_k_match[ex.K].update(gold_ranks)
            sel_chain_accum_k_match["all"].update(gold_sels)
            if ex.K in sel_chain_accum_k_match:
                sel_chain_accum_k_match[ex.K].update(gold_sels)
        if ms.ex_ranks:
            chain_accum["all"].update(ms.ex_ranks)
            if ex.K in chain_accum:
                chain_accum[ex.K].update(ms.ex_ranks)
        if ms.ex_sel_correct:
            sel_chain_accum["all"].update(ms.ex_sel_correct)
            if ex.K in sel_chain_accum:
                sel_chain_accum[ex.K].update(ms.ex_sel_correct)

        predicted = ms.final_answer if args.final_reader else (ms.prior[-1] if ms.prior else "")
        if predicted:
            try:
                em, f1 = judge_answer_official(predicted, ex.row)
            except Exception:
                em, f1 = None, None
        else:
            em, f1 = None, None
        ans_accum["all"].update(em, f1)
        if ex.K in ans_accum:
            ans_accum[ex.K].update(em, f1)

        cases_fp.write(json.dumps({
            "id": ex.eid, "question": ex.q_main, "K_pred": ex.K, "K_gold": ex.K_gold, "K_match": ex.k_match,
            "gold_hop_ranks": gold_ranks, "case_recall1": cr1, "case_recall3": cr3,
            "predicted_answer": predicted, "answer_em": em, "answer_f1": f1,
            "final_reader_cot": ms.final_cot if args.final_reader else None,
            "hop_results": ex.hop_results,
        }, ensure_ascii=False) + "\n")
    cases_fp.close()

    def by_k(accum_dict):
        return {str(K): accum_dict[K].result() for K in K_BUCKETS if K in accum_dict}

    results = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "method": METHOD,
        "n_examples": n_examples, "n_k_match": n_k_match,
        "k_match_rate": round(n_k_match / n_examples, 4) if n_examples else 0.0,
        "overall": accum["all"].result(), "by_K": by_k(accum),
        "chain_overall": chain_accum["all"].result(), "chain_by_K": by_k(chain_accum),
        "chain_overall_k_match": chain_accum_k_match["all"].result(),
        "chain_by_K_k_match": by_k(chain_accum_k_match),
        "case_recall_overall": case_recall_accum["all"].result(),
        "case_recall_by_K_gold": by_k(case_recall_accum),
        "selection_overall": sel_accum["all"].result(), "selection_by_K": by_k(sel_accum),
        "chain_selection_overall": sel_chain_accum["all"].result(),
        "chain_selection_by_K": by_k(sel_chain_accum),
        "chain_selection_overall_k_match": sel_chain_accum_k_match["all"].result(),
        "chain_selection_by_K_k_match": by_k(sel_chain_accum_k_match),
        "answer_overall": ans_accum["all"].result(), "answer_by_K": by_k(ans_accum),
        "timing": {"wall_clock_sec": round(time.perf_counter() - run_t0, 2)},
    }
    out_json = args.out_dir / f"retrieval_exp_{args.split}{tag_suffix}.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_json}")
    print(f"recall@1={results['overall']['recall@1']}  chain_recall@1={results['chain_overall']['chain_recall@1']}  "
          f"answer_em={results['answer_overall']['answer_em']}  answer_f1={results['answer_overall']['answer_f1']}")


if __name__ == "__main__":
    main()
