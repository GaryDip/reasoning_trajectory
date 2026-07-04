"""Evaluation metrics aligned with official MuSiQue decomposer validation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from transformers import PreTrainedTokenizerBase

from data_utils import target_to_hops


def decode_generated_sequences(
    tokenizer: PreTrainedTokenizerBase,
    token_ids_batch,
) -> list[str]:
    arr = np.asarray(token_ids_batch)
    vocab_size = len(tokenizer)

    if arr.ndim == 3:
        # Logits: (batch, seq_len, vocab); beam outputs: (batch, num_beams, seq_len)
        if arr.shape[-1] >= max(256, vocab_size // 4):
            arr = arr.argmax(axis=-1)
        else:
            arr = arr[:, 0, :]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    cleaned = np.where(arr == -100, pad_id, arr)
    decoded = tokenizer.batch_decode(
        cleaned,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [text.strip() for text in decoded]


def exact_match_rate(predictions: list[str], references: list[str]) -> float:
    if not references:
        return 0.0
    correct = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
    return correct / len(references)


def hop_count_accuracy(predictions: list[str], references: list[str]) -> float:
    if not references:
        return 0.0
    correct = sum(
        len(target_to_hops(p)) == len(target_to_hops(r))
        for p, r in zip(predictions, references)
    )
    return correct / len(references)


def corpus_bleu(predictions: list[str], references: list[str]) -> float:
    if not predictions or not references:
        return 0.0
    try:
        import sacrebleu
    except ImportError as exc:
        raise ImportError(
            "sacrebleu is required for eval BLEU. Install with: pip install sacrebleu"
        ) from exc

    return float(sacrebleu.corpus_bleu(predictions, [references]).score)


def summarize_by_gold_hops(
    predictions: list[str],
    references: list[str],
    gold_hop_counts: list[int] | None = None,
) -> dict[str, Any]:
    if gold_hop_counts is None:
        gold_hop_counts = [len(target_to_hops(r)) for r in references]

    buckets: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0,
            "exact_match": 0,
            "hop_count_match": 0,
            "pred_hop_distribution": defaultdict(int),
        }
    )

    for pred, ref, gold_k in zip(predictions, references, gold_hop_counts):
        bucket = buckets[gold_k]
        bucket["n"] += 1
        if pred.strip() == ref.strip():
            bucket["exact_match"] += 1
        pred_k = len(target_to_hops(pred))
        if pred_k == gold_k:
            bucket["hop_count_match"] += 1
        bucket["pred_hop_distribution"][pred_k] += 1

    by_hop: dict[str, Any] = {}
    for gold_k in sorted(buckets):
        bucket = buckets[gold_k]
        n = bucket["n"]
        by_hop[str(gold_k)] = {
            "n": n,
            "exact_match": bucket["exact_match"] / n if n else 0.0,
            "hop_count_acc": bucket["hop_count_match"] / n if n else 0.0,
            "pred_hop_distribution": dict(sorted(bucket["pred_hop_distribution"].items())),
        }

    return {
        "n": len(predictions),
        "bleu": corpus_bleu(predictions, references),
        "exact_match": exact_match_rate(predictions, references),
        "hop_count_acc": hop_count_accuracy(predictions, references),
        "by_gold_hops": by_hop,
    }


def flatten_metrics_for_trainer(summary: dict[str, Any]) -> dict[str, float]:
    metrics = {
        "bleu": float(summary["bleu"]),
        "exact_match": float(summary["exact_match"]),
        "hop_count_acc": float(summary["hop_count_acc"]),
    }
    for gold_k, bucket in summary.get("by_gold_hops", {}).items():
        metrics[f"exact_match_k{gold_k}"] = float(bucket["exact_match"])
        metrics[f"hop_count_acc_k{gold_k}"] = float(bucket["hop_count_acc"])
    return metrics


def format_eval_summary(summary: dict[str, Any], *, prefix: str = "") -> str:
    lines = [
        f"{prefix}overall: n={summary['n']} | "
        f"bleu={summary['bleu']:.2f} | "
        f"exact_match={summary['exact_match']:.1%} | "
        f"hop_count_acc={summary['hop_count_acc']:.1%}",
        f"{prefix}by gold hops:",
    ]
    for gold_k, bucket in summary.get("by_gold_hops", {}).items():
        lines.append(
            f"{prefix}  k={gold_k}: n={bucket['n']} | "
            f"exact_match={bucket['exact_match']:.1%} | "
            f"hop_count_acc={bucket['hop_count_acc']:.1%} | "
            f"pred_hops={bucket['pred_hop_distribution']}"
        )
    return "\n".join(lines)


def summarize_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = [row.get("predicted_target") or "" for row in rows]
    references = [row.get("gold_target") or "" for row in rows]
    gold_hop_counts = [row.get("num_hops_gold", 0) for row in rows]
    return summarize_by_gold_hops(predictions, references, gold_hop_counts)
