#!/usr/bin/env python3
"""
Build exact-GT reasoning traces without any LLM API call.

Input:
  --nl-decomp   musique_gt_nl_{split}.jsonl   (id, question, sub_questions)
  --musique     musique_ans_v1.0_{split}.jsonl (paragraphs, question_decomposition, answer)

For each example:
  step_reasoning[i] = sub_questions[i]  (from NL decomp file)
  evidence_text[i]  = paragraphs[paragraph_support_idx[i]]  (from MuSiQue gold)
  final_answer      = answer  (from MuSiQue gold)

Output trace format (same as musique_to_reasoning_trace.py):
  Question: {q} Step 1: {r1} Evidence: "{e1}" Step 2: {r2} Evidence: "{e2}" ... Final Answer: {ans}

Usage:
  python build_exact_gt_trace.py \\
      --nl-decomp ../musique_gt_nl_train.jsonl \\
      --musique   ../../../data/raw/musique/musique_ans_v1.0_train.jsonl \\
      --output    ../train_exact_gt_traces.jsonl

  python build_exact_gt_trace.py \\
      --nl-decomp ../musique_gt_nl_dev.jsonl \\
      --musique   ../../../data/raw/musique/musique_ans_v1.0_dev.jsonl \\
      --output    ../dev_exact_gt_traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


# ── trace assembly helpers (mirrors musique_to_reasoning_trace.py) ─────────────

def escape_double_quotes(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def assemble_trace(
    question: str,
    step_reasoning: list[str],
    evidence_texts: list[str],
    final_answer: str,
) -> str:
    if len(step_reasoning) != len(evidence_texts):
        raise ValueError(
            f"step_reasoning ({len(step_reasoning)}) != evidence_texts ({len(evidence_texts)})"
        )
    parts: list[str] = [f"Question: {question.strip()}"]
    for i, (reason, ev) in enumerate(zip(step_reasoning, evidence_texts), start=1):
        parts.append(f"Step {i}: {reason.strip()} Evidence: \"{escape_double_quotes(ev.strip())}\"")
    parts.append(f"Final Answer: {final_answer.strip()}")
    return " ".join(parts)


def validate_trace(trace: str, k: int) -> list[str]:
    errs: list[str] = []
    if "\n" in trace:
        errs.append("trace contains newline")
    for i in range(1, k + 1):
        if f"Step {i}:" not in trace:
            errs.append(f"missing Step {i}:")
    if not re.search(r"Final Answer\s*:", trace, re.IGNORECASE):
        errs.append("missing Final Answer:")
    return errs


def paragraph_by_idx(paragraphs: list[dict], idx: int) -> dict | None:
    for p in paragraphs:
        if int(p.get("idx", -1)) == int(idx):
            return p
    return None


# ── loading ────────────────────────────────────────────────────────────────────

def load_nl_decomp(path: Path) -> dict[str, list[str]]:
    """id -> sub_questions list"""
    index: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rid = str(r.get("id", "")).strip()
            sqs = r.get("sub_questions", [])
            if rid and isinstance(sqs, list):
                index[rid] = [str(q).strip() for q in sqs]
    print(f"Loaded {len(index)} NL decompositions from {path.name}", file=sys.stderr)
    return index


def load_musique_index(path: Path) -> dict[str, dict]:
    """id -> full MuSiQue record"""
    index: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rid = str(r.get("id", "")).strip()
            if rid:
                index[rid] = r
    print(f"Loaded {len(index)} MuSiQue records from {path.name}", file=sys.stderr)
    return index


# ── per-example processing ─────────────────────────────────────────────────────

def process_example(
    rid: str,
    sub_questions: list[str],
    musique_rec: dict,
) -> dict[str, Any]:
    paragraphs = musique_rec.get("paragraphs") or []
    decomp     = musique_rec.get("question_decomposition") or []
    question   = (musique_rec.get("question") or "").strip()
    answer     = (musique_rec.get("answer") or "").strip()
    k = len(decomp)

    if len(sub_questions) != k:
        raise ValueError(
            f"sub_questions length {len(sub_questions)} != decomp length {k}"
        )

    evidence_texts: list[str] = []
    meta_decomp: list[dict] = []
    for i, step in enumerate(decomp):
        pidx = step.get("paragraph_support_idx")
        p = paragraph_by_idx(paragraphs, pidx) if pidx is not None else None
        if not p:
            raise ValueError(
                f"hop {i+1}: paragraph_support_idx={pidx} not found in paragraphs"
            )
        evidence_texts.append((p.get("paragraph_text") or "").strip())
        meta_decomp.append({
            "id":                    step.get("id"),
            "question":              step.get("question"),
            "answer":                step.get("answer"),
            "paragraph_support_idx": pidx,
        })

    # step_reasoning = NL sub_questions (already natural language)
    trace = assemble_trace(question, sub_questions, evidence_texts, answer)
    trace = trace.replace("\n", " ").strip()
    warnings = validate_trace(trace, k)

    return {
        "id":                    rid,
        "question":              question,
        "answer":                answer,
        "question_decomposition": meta_decomp,
        "num_hops":              k,
        "step_reasoning":        sub_questions,
        "reasoning_trace":       trace,
        "validation_warnings":   warnings,
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build exact-GT reasoning traces from NL decompositions (no API call)"
    )
    ap.add_argument(
        "--nl-decomp",
        required=True,
        help="musique_gt_nl_{split}.jsonl (id, sub_questions)",
    )
    ap.add_argument(
        "--musique",
        required=True,
        help="musique_ans_v1.0_{split}.jsonl (gold evidence + answer)",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output JSONL path (e.g. ../train_exact_gt_traces.jsonl)",
    )
    ap.add_argument(
        "--skip-missing",
        action="store_true",
        default=True,
        help="Skip examples missing from either file instead of crashing (default: True)",
    )
    args = ap.parse_args()

    nl_decomp_path = Path(args.nl_decomp)
    musique_path   = Path(args.musique)
    output_path    = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nl_index  = load_nl_decomp(nl_decomp_path)
    mu_index  = load_musique_index(musique_path)

    n_ok = n_warn = n_skip = n_err = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for rid, sub_questions in nl_index.items():
            if rid not in mu_index:
                if args.skip_missing:
                    n_skip += 1
                    print(f"SKIP {rid}: not in MuSiQue file", file=sys.stderr)
                    continue
                else:
                    raise KeyError(f"{rid} not found in {musique_path}")

            try:
                row = process_example(rid, sub_questions, mu_index[rid])
            except Exception as exc:
                n_err += 1
                print(f"ERROR {rid}: {exc}", file=sys.stderr)
                continue

            if row["validation_warnings"]:
                n_warn += 1
                print(
                    f"WARN {rid}: {row['validation_warnings']}", file=sys.stderr
                )

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_ok += 1

    print(
        f"\nDone: ok={n_ok}, warnings={n_warn}, errors={n_err}, skipped={n_skip}",
        file=sys.stderr,
    )
    print(f"Output: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
