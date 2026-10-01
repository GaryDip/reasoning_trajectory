#!/usr/bin/env python3
"""Where does each hop's gold passage sit in BGE's ranking over the global corpus, and what does
gate v3 reranking do with it? Uses the global-setting runs' case logs (top10 / top50, both with
comparison_hint), no LLM calls -- only the per-hop queries are re-encoded.

Per hop j (same hop set recall@k is accumulated over: j < min(#hops run, #gold hops)), with the
position-aligned gold g = gold_idxs[j]:
  - query = query_instruction + expand_hop_template(raw_sq[j], prior[:j])  (exactly what the run sent)
  - BGE rank of g over the whole corpus (1 = best)
  - post-rerank position of g in the run's logged pool `hop_ranked_para_ids` (None = not in pool)

Self-checks: "BGE rank <= k" must agree with "g in the logged top-k pool", and recall@1/@3 from the
logged pools must reproduce each run's reported numbers.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_retrieval_exp import _get_pipeline_helpers, load_decompose_index  # noqa: E402
from build_global_corpus import load_global_corpus  # noqa: E402

BUCKETS = [(1, 1), (2, 3), (4, 10), (11, 20), (21, 50), (51, 100), (101, 10**9)]


def latest_run(ds: str, k: int) -> Path:
    return Path(sorted(glob.glob(str(HERE / f"results/*_global_{ds}_k{k}_comparisonhint")))[-1])


def load_run(run: Path):
    summary = json.loads(next(run.glob("retrieval_exp_*.json")).read_text())
    cases = [json.loads(l) for l in next(run.glob("retrieval_cases_*.jsonl")).open() if l.strip()]
    return summary, cases


def hop_records(cases, decomp, expand, instruction):
    """One record per scored hop: (query, gold_global_idx, logged_pool)."""
    out = []
    for c in cases:
        raw_sqs = decomp.get(c["id"]) or []
        pools = c["hop_ranked_para_ids"]
        for j in range(min(len(pools), len(c["gold_idxs"]))):
            q = instruction + expand(raw_sqs[j], c["prior"][:j])
            out.append((q, int(c["gold_idxs"][j]), pools[j]))
    return out


def encode_queries(model, queries):
    qv = model.encode(queries, batch_size=256, normalize_embeddings=True, convert_to_numpy=True,
                      show_progress_bar=False).astype(np.float32)
    return qv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["musique", "2wiki", "hotpot"])
    ap.add_argument("--out", type=Path, default=HERE / "results/global_gold_rank_analysis.json")
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer
    expand = _get_pipeline_helpers()[0]
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda" if torch.cuda.is_available() else "cpu")
    report: dict = {}

    for ds in args.datasets:
        _paras, doc_emb, _ = load_global_corpus(HERE / f"global_corpus/{ds}_dev")
        D = torch.from_numpy(doc_emb.astype(np.float32)).cuda()
        report[ds] = {}
        per_run_hops = {}
        for k in (10, 50):
            summary, cases = load_run(latest_run(ds, k))
            cfg = summary["config"]
            decomp = load_decompose_index(Path(cfg["decompose_file"]))
            hops = hop_records(cases, decomp, expand, cfg["query_instruction"])
            qv = torch.from_numpy(encode_queries(model, [h[0] for h in hops])).cuda()
            gold = torch.tensor([h[1] for h in hops], device="cuda")
            ranks = []
            for s in range(0, len(hops), 2048):
                sims = qv[s:s + 2048] @ D.T                          # [b, N]
                gs = sims.gather(1, gold[s:s + 2048, None])          # [b, 1]
                ranks.append(((sims > gs).sum(1) + 1).cpu())
            bge_rank = torch.cat(ranks).numpy()
            pool_pos = np.array([(h[2].index(h[1]) + 1) if h[1] in h[2] else 0 for h in hops])  # 0 = not in pool

            # self-checks
            in_pool = pool_pos > 0
            agree = float(((bge_rank <= k) == in_pool).mean())
            rep = summary["full_pool_gold_rank_overall"]
            r1, r3 = float((pool_pos == 1).mean()), float(((pool_pos >= 1) & (pool_pos <= 3)).mean())
            print(f"[{ds} k={k}] hops={len(hops)}  check: BGE<=k vs logged pool agree={agree:.4f}  "
                  f"recall@1 {r1:.4f} (reported {rep['recall@1']})  recall@3 {r3:.4f} (reported {rep['recall@3']})")
            per_run_hops[k] = (bge_rank, pool_pos)
            report[ds][f"k{k}_checks"] = {"n_hops": len(hops), "bge_vs_pool_agree": agree,
                                          "recall@1": r1, "recall@3": r3,
                                          "reported_recall@1": rep["recall@1"], "reported_recall@3": rep["recall@3"]}

        # 1) BGE rank distribution (queries of the top10 run)
        bge_rank, _ = per_run_hops[10]
        dist = {f"{lo}-{hi}" if hi < 10**9 else f">{lo - 1}": float(((bge_rank >= lo) & (bge_rank <= hi)).mean())
                for lo, hi in BUCKETS}
        cum = {f"<={t}": float((bge_rank <= t).mean()) for t in (1, 3, 10, 20, 50, 100)}
        report[ds]["bge_rank_distribution"] = dist
        report[ds]["bge_rank_cumulative"] = cum

        # 2) what the gate does in the top50 run, by BGE rank band (queries of the top50 run)
        bge50, pos50 = per_run_hops[50]
        bands = {}
        for name, m in [("bge 1-10", bge50 <= 10), ("bge 11-50", (bge50 > 10) & (bge50 <= 50)), ("bge >50", bge50 > 50)]:
            n = int(m.sum())
            bands[name] = {"share_of_hops": float(m.mean()), "n": n,
                           "gate_top1": float((pos50[m] == 1).mean()) if n else None,
                           "gate_top3": float(((pos50[m] >= 1) & (pos50[m] <= 3)).mean()) if n else None}
        report[ds]["top50_gate_by_bge_band"] = bands

        print(f"  BGE rank distribution (top10-run queries): " + "  ".join(f"{b}={v:.1%}" for b, v in dist.items()))
        print(f"  BGE cumulative: " + "  ".join(f"{b}={v:.1%}" for b, v in cum.items()))
        pct = lambda v: "n/a" if v is None else f"{v:.1%}"
        for name, b in bands.items():
            print(f"  top50 run, gold at {name:<9}: {b['share_of_hops']:.1%} of hops (n={b['n']})  "
                  f"-> after gate: top1 {pct(b['gate_top1'])}  top3 {pct(b['gate_top3'])}")
        del D
        torch.cuda.empty_cache()

    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
