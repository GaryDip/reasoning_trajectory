#!/usr/bin/env python3
"""
Scheme A (pairwise ranking MLP) — training-data builder.

For each (source question, K, hop j) where BOTH a gold-trace instance and a
matching wrong-evidence-at-hop-j counterfactual instance exist in the
existing hidden_states/ data, emit a (correct, wrong) PAIR:
  - Delta_j (h_{j+1} - h_j) for each, already computed and stored — reused
    from the existing .npz files, not re-extracted.
  - emb_score for each: cosine(subquestion, that instance's own evidence
    text), computed here (not stored anywhere yet) with a real BGE encode.

Output is one .npz with parallel arrays (deltas, emb_scores, labels, K, j,
pair_id) — pair_id groups exactly one correct + one wrong row together, so
train_pairwise_mlp.py can reconstruct pairs for the ranking loss.

Does not modify hidden_states/ or gate/ — reads them read-only, reuses
hidden_states/trace_parse.py's quote-parsing helper and gate/hop_labels.py's
wrong-hops parsing (both imported, not copied).

Usage:
  python build_training_data.py --split train
  python build_training_data.py --split dev
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
HIDDEN_STATES_DIR = PROJECT_ROOT / "hidden_states"
GATE_DIR = PROJECT_ROOT / "gate"
sys.path.insert(0, str(HIDDEN_STATES_DIR))
sys.path.insert(0, str(GATE_DIR))

from trace_parse import consume_evidence_quote, parse_trace_structure  # noqa: E402
from hop_labels import parse_wrong_hops  # noqa: E402

STEP_RE = re.compile(r"^Step\s+\d+:\s*")


def split_step_block(block: str) -> tuple[str, str]:
    """'Step 1: <subq> Evidence: "<text>"' -> (subq, text). Reuses the same
    quote-escaping logic trace_parse.py already uses for the whole trace,
    just applied to a single Step block."""
    m = STEP_RE.match(block)
    rest = block[m.end():] if m else block
    marker = 'Evidence: "'
    idx = rest.find(marker)
    if idx < 0:
        raise ValueError(f"no Evidence marker in block: {block[:120]!r}")
    subq = rest[:idx].strip()
    inner_start = idx + len(marker)
    evidence_text, _close_idx = consume_evidence_quote(rest, inner_start)
    return subq, evidence_text


def load_trace_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[str(r["id"])] = r
    return rows


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def hop_subq_and_evidence(trace_row: dict[str, Any], hop_1indexed: int) -> tuple[str, str]:
    _q, blocks, _ans = parse_trace_structure(trace_row["reasoning_trace"])
    return split_step_block(blocks[hop_1indexed - 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["musique"], default="musique")
    ap.add_argument("--split", choices=["train", "dev"], default="train")
    ap.add_argument("--hidden-dir", type=Path, default=None)
    ap.add_argument("--trace-file", type=Path, default=None)
    ap.add_argument("--cos-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--encode-batch-size", type=int, default=256)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit-pairs", type=int, default=0)
    args = ap.parse_args()

    hidden_dir = args.hidden_dir or (HIDDEN_STATES_DIR / args.split)
    trace_file = args.trace_file or (PROJECT_ROOT / "traces" / "merged" / args.dataset / f"{args.split}.jsonl")
    out_path = args.out or (HERE / "data" / f"{args.dataset}_{args.split}_pairwise.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading trace rows from {trace_file} ...")
    trace_rows = load_trace_rows(trace_file)
    print(f"Loading manifest from {hidden_dir}/manifest.jsonl ...")
    manifest = load_manifest(hidden_dir / "manifest.jsonl")

    pos_by_id = {r["id"]: r for r in manifest if r.get("trace_type") == "correct"}
    neg_rows = [r for r in manifest if r.get("trace_type") == "error"]
    print(f"pos: {len(pos_by_id)}  neg: {len(neg_rows)}")

    # Pass 1: figure out which (pos, neg) pairs we can actually build, and
    # collect all (subq, evidence_text) strings that need BGE encoding.
    pending: list[dict[str, Any]] = []
    texts_to_encode: list[str] = []
    text_index: dict[str, int] = {}

    def text_id(t: str) -> int:
        if t not in text_index:
            text_index[t] = len(texts_to_encode)
            texts_to_encode.append(t)
        return text_index[t]

    n_skipped = 0
    for neg in neg_rows:
        wrong_hops = parse_wrong_hops(neg)
        if len(wrong_hops) != 1:
            n_skipped += 1
            continue
        wh = wrong_hops[0]
        j = wh - 1
        source_id = neg.get("source_id")
        pos = pos_by_id.get(source_id)
        if pos is None:
            n_skipped += 1
            continue
        neg_trace = trace_rows.get(neg["id"])
        pos_trace = trace_rows.get(pos["id"])
        if neg_trace is None or pos_trace is None:
            n_skipped += 1
            continue
        try:
            subq, wrong_ev = hop_subq_and_evidence(neg_trace, wh)
            _subq2, gold_ev = hop_subq_and_evidence(pos_trace, wh)
        except (ValueError, IndexError):
            n_skipped += 1
            continue

        pending.append({
            "pair_id": f"{source_id}__K{neg['K']}__j{j}",
            "K": int(neg["K"]),
            "j": j,
            "pos_path": hidden_dir / pos["path"],
            "neg_path": hidden_dir / neg["path"],
            "subq_idx": text_id(subq),
            "gold_ev_idx": text_id(gold_ev),
            "wrong_ev_idx": text_id(wrong_ev),
        })
        if args.limit_pairs and len(pending) >= args.limit_pairs:
            break

    print(f"Buildable pairs: {len(pending)} (skipped {n_skipped})")
    print(f"Distinct texts to encode: {len(texts_to_encode)}")

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(args.cos_model)
    embeddings = st.encode(
        texts_to_encode, batch_size=args.encode_batch_size,
        normalize_embeddings=True, show_progress_bar=True,
    )

    deltas: list[np.ndarray] = []
    emb_scores: list[float] = []
    labels: list[int] = []
    Ks: list[int] = []
    js: list[int] = []
    pair_ids: list[str] = []

    npz_cache: dict[Path, np.ndarray] = {}

    def get_hidden(path: Path) -> np.ndarray:
        if path not in npz_cache:
            npz_cache[path] = np.load(path)["hidden"]
        return npz_cache[path]

    for item in pending:
        j = item["j"]
        pos_hidden = get_hidden(item["pos_path"])
        neg_hidden = get_hidden(item["neg_path"])
        if j + 1 >= pos_hidden.shape[0] or j + 1 >= neg_hidden.shape[0]:
            continue
        delta_pos = (pos_hidden[j + 1] - pos_hidden[j]).astype(np.float32)
        delta_neg = (neg_hidden[j + 1] - neg_hidden[j]).astype(np.float32)

        qv = embeddings[item["subq_idx"]]
        gold_score = float(np.dot(qv, embeddings[item["gold_ev_idx"]]))
        wrong_score = float(np.dot(qv, embeddings[item["wrong_ev_idx"]]))

        for delta, emb_score, label in (
            (delta_pos, gold_score, 1),
            (delta_neg, wrong_score, 0),
        ):
            deltas.append(delta)
            emb_scores.append(emb_score)
            labels.append(label)
            Ks.append(item["K"])
            js.append(j)
            pair_ids.append(item["pair_id"])

    np.savez_compressed(
        out_path,
        deltas=np.stack(deltas).astype(np.float32),
        emb_scores=np.asarray(emb_scores, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        K=np.asarray(Ks, dtype=np.int64),
        j=np.asarray(js, dtype=np.int64),
        pair_id=np.asarray(pair_ids, dtype=object),
    )
    print(f"Wrote {len(deltas)} rows ({len(pending)} pairs) -> {out_path}")


if __name__ == "__main__":
    main()
