#!/usr/bin/env python3
"""
Scheme A — retrieval-level evaluation (recall@1/@3/MRR), NOT full EM/F1.
Deliberately cheaper than an end-to-end run: uses GOLD prior-hop answers to
fill each hop's [Answer N] placeholders (instead of generating them via
vLLM), isolating "does this reranker structure pick gold at rank 1 better"
from "does the reader chain answers correctly" — matches the same
isolate-one-variable principle used in reader_ceiling_probe. Only needs the
Llama gate-style model (for Delta extraction), not vLLM.

For each hop: top-20 cosine retrieval -> extract Delta for every candidate
(same "prefix + Step j + Evidence" template used to build hidden_states/ in
the first place, so features match training) -> score with the trained
pairwise MLP -> rank -> compare against gold_para_idx. Reports this new
reranker's recall@1/@3/MRR against a cosine-only baseline (and optionally
the existing production gate's lambda-weighted formula, if
--compare-artifacts-dir is given).

Usage (needs a real GPU + Llama checkpoint):
  python eval_retrieval.py --artifacts-dir artifacts --limit 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RETRIEVAL_DIR = PROJECT_ROOT / "retrieval"
TRACES_DIR = PROJECT_ROOT / "traces"
GATE_DIR = PROJECT_ROOT / "gate"
sys.path.insert(0, str(RETRIEVAL_DIR))
sys.path.insert(0, str(TRACES_DIR))
sys.path.insert(0, str(GATE_DIR))

from run_retrieval_exp import (  # noqa: E402
    MUSIQUE_DIR,
    _get_pipeline_helpers,
    build_trace_prefix,
    embed_retrieval,
    find_gold_rank,
    load_dataset_records,
    load_decompose_index,
    prepare_sub_questions,
    resolve_decompose_file,
    select_example_ids,
)
from run_retrieval_exp_wavefront import init_example_state, load_gate_model  # noqa: E402
from trace_evidence import gold_hop_answers_musique  # noqa: E402
from trace_format import escape_double_quotes  # noqa: E402

METHOD = "pairwise_mlp"


class MLPScorer:
    """Loads the artifacts train_pairwise_mlp.py produced; scores
    [emb_score, PCA(Delta), onehot(j), K] — the hop-depth features are only
    appended if meta.json says the artifact was trained with them (backward
    compatible with older artifacts trained before this was added)."""

    def __init__(self, artifacts_dir: Path):
        import joblib
        import torch
        from torch import nn

        meta = __import__("json").loads((artifacts_dir / "meta.json").read_text())
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
    ap.add_argument("--retrieve-k", type=int, default=20)
    ap.add_argument("--cos-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--answerable-only", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--gate-device", default="cuda:0")
    ap.add_argument("--hidden-batch-size", type=int, default=8)
    return ap.parse_args()


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
    ids = select_example_ids(ids, decompose_idx, limit=args.limit, limit_per_k=0, seed=args.sample_seed)

    (expand_hop_template, *_rest) = _get_pipeline_helpers()

    print(f"Examples: {len(ids)}  decompose={args.decompose_file}")

    examples = []
    gold_answers_by_id: dict[str, list[str]] = {}
    for eid in ids:
        row = records[eid]
        sub_qs = prepare_sub_questions(decompose_idx.get(eid) or [], decompose_mode=args.decompose_mode)
        if not sub_qs:
            continue
        examples.append(init_example_state(eid, row, sub_qs, [METHOD], []))
        gold_answers_by_id[eid] = gold_hop_answers_musique(row)

    scorer = MLPScorer(args.artifacts_dir)
    print(f"Loaded scorer from {args.artifacts_dir} (layer={scorer.layer})")

    print("Loading gate-style Llama model (for Delta extraction only, no vLLM needed) ...", flush=True)
    model_args = argparse.Namespace(
        model=args.model, dtype=args.dtype, attn_implementation=args.attn_implementation,
        gate_device=args.gate_device,
    )
    model, tokenizer = load_gate_model(model_args)
    dev = next(model.parameters()).device

    def batch_hidden(texts: list[str]) -> np.ndarray:
        import torch
        out = []
        bs = args.hidden_batch_size
        for start in range(0, len(texts), bs):
            chunk = texts[start:start + bs]
            rendered = [
                tokenizer.apply_chat_template([{"role": "user", "content": t}],
                                               add_generation_prompt=False, tokenize=False)
                for t in chunk
            ]
            batch = tokenizer(rendered, return_tensors="pt", padding=True).to(dev)
            with torch.inference_mode():
                res = model(**batch, output_hidden_states=True, use_cache=False)
            hs = res.hidden_states[scorer.layer + 1]
            attn = batch["attention_mask"]
            last_idx = attn.size(1) - 1 - torch.flip(attn, dims=[1]).argmax(dim=1)
            idx = torch.arange(hs.size(0), device=dev)
            out.extend(hs[idx, last_idx].float().cpu().numpy())
        return np.stack(out).astype(np.float32)

    max_hops = max((ex.K for ex in examples), default=0)
    recall_hits = {"model": {1: 0, 3: 0}, "cosine": {1: 0, 3: 0}}
    mrr_sum = {"model": 0.0, "cosine": 0.0}
    n_hops_scored = 0

    for hop_j in range(1, max_hops + 1):
        print(f"=== hop {hop_j}/{max_hops} ===", flush=True)
        for ex in examples:
            if hop_j > ex.K:
                continue
            hop_row = ex.hop_results[hop_j - 1]
            gold_pi = hop_row.get("gold_para_idx")
            if not (hop_row.get("has_gold_hop") and gold_pi is not None):
                continue

            gold_answers = gold_answers_by_id[ex.eid]
            prior = gold_answers[: hop_j - 1]
            raw_sq = ex.sub_questions[hop_j - 1]
            expanded_q = expand_hop_template(raw_sq, prior)

            # Oracle prior hop_steps (gold evidence for hops before this one) so
            # only THIS hop's reranking is under test.
            hop_steps = []
            for gj in range(hop_j - 1):
                prior_pi = ex.hop_results[gj].get("gold_para_idx")
                prior_para = next((p for p in ex.paragraphs if int(p.get("idx", -1)) == prior_pi), None)
                if prior_para is None:
                    hop_steps = None
                    break
                hop_steps.append((ex.sub_questions[gj], (prior_para.get("paragraph_text") or "").strip()))
            if hop_steps is None:
                continue
            prefix_before = build_trace_prefix(ex.q_main, hop_steps)

            candidates_all = embed_retrieval(expanded_q, ex.paragraphs, args.retrieve_k, args.cos_model)
            if not candidates_all:
                continue

            texts = [
                f'{prefix_before} Step {hop_j}: {expanded_q}'
                f' Evidence: "{escape_double_quotes((p.get("paragraph_text") or "").strip())}"'
                for p, _s in candidates_all
            ]
            hidden_prev = batch_hidden([prefix_before])[0]
            hidden_cands = batch_hidden(texts)
            deltas = hidden_cands - hidden_prev[None, :]
            emb_scores = np.asarray([s for _p, s in candidates_all], dtype=np.float32)

            model_scores = scorer.score(emb_scores, deltas, K=ex.K, j=hop_j - 1)
            model_ranked = sorted(zip([p for p, _ in candidates_all], model_scores),
                                   key=lambda t: -t[1])
            cosine_ranked = candidates_all  # already sorted by cosine similarity desc

            for name, ranked in (("model", model_ranked), ("cosine", cosine_ranked)):
                rank = find_gold_rank(ranked, int(gold_pi))
                if rank is not None:
                    if rank <= 1:
                        recall_hits[name][1] += 1
                    if rank <= 3:
                        recall_hits[name][3] += 1
                    mrr_sum[name] += 1.0 / rank
            n_hops_scored += 1

    print(f"\nHops scored: {n_hops_scored}")
    for name in ("cosine", "model"):
        r1 = recall_hits[name][1] / n_hops_scored if n_hops_scored else 0.0
        r3 = recall_hits[name][3] / n_hops_scored if n_hops_scored else 0.0
        mrr = mrr_sum[name] / n_hops_scored if n_hops_scored else 0.0
        print(f"[{name:>6}] recall@1={r1:.4f}  recall@3={r3:.4f}  mrr={mrr:.4f}")


if __name__ == "__main__":
    main()
