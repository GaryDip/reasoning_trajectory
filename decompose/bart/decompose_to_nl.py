#!/usr/bin/env python3
"""
Convert BART-predicted MuSiQue decompositions to natural-language sub-questions
using local Llama-3.1-8B-Instruct.

Input: predict output (v1 hop text preferred), e.g. musique_ans_dev_predictions_v1.jsonl
Output: musique_gt_nl_dev.jsonl schema plus hop counts:
  {"id": "...", "question": "...", "sub_questions": [...],
   "num_hops": 2, "num_hops_pred": 2, "num_hops_gold": 2}

Example:
  python decompose_to_nl.py \\
    --input outputs/musique_ans_dev_predictions_v1.jsonl \\
    --output outputs/musique_pred_nl_dev.jsonl \\
    --device-map cuda:0 --batch-size 8

  # Faster local inference with vLLM:
  python decompose_to_nl.py \\
    --input outputs/2wiki_dev_predictions.jsonl \\
    --output outputs/2wiki_pred_nl_dev.jsonl \\
    --backend vllm --tensor-parallel-size 1 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from tqdm import tqdm

from data_utils import raw_decomposition_to_v1_hop_texts, translate_id
import llama_chat as llama_base

_SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_INPUT = _SCRIPT_DIR / "outputs/musique_ans_dev_predictions_v1.jsonl"
DEFAULT_OUTPUT = _SCRIPT_DIR / "outputs/musique_pred_nl_dev.jsonl"


class ChatBackend(Protocol):
    def chat(self, user_text: str, max_new_tokens: int) -> str: ...

    def chat_batch(self, user_texts: list[str], max_new_tokens: int) -> list[str]: ...

    def unload(self) -> None: ...


class VllmChat:
    def __init__(
        self,
        *,
        model: str,
        dtype: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int | None,
    ) -> None:
        try:
            from vllm import LLM
        except ImportError as exc:
            raise SystemExit(
                "vLLM backend requested but vllm is not installed. "
                "Install vllm first, then rerun with --backend vllm."
            ) from exc

        kwargs: dict[str, Any] = {
            "model": model,
            "dtype": dtype,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
        }
        if max_model_len:
            kwargs["max_model_len"] = max_model_len
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def _prompt(self, user_text: str) -> str:
        messages = [{"role": "user", "content": user_text}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def chat(self, user_text: str, max_new_tokens: int) -> str:
        return self.chat_batch([user_text], max_new_tokens)[0]

    def chat_batch(self, user_texts: list[str], max_new_tokens: int) -> list[str]:
        if not user_texts:
            return []
        from vllm import SamplingParams

        prompts = [self._prompt(text) for text in user_texts]
        params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
        )
        outputs = self.llm.generate(prompts, params)
        return [out.outputs[0].text.strip() if out.outputs else "" for out in outputs]

    def unload(self) -> None:
        del self.llm

REWRITE_SYSTEM = """\
You are an expert in multi-hop question answering.
Your task is to rewrite a sequence of semi-formal KB-style sub-questions into
fluent, natural-language English questions, preserving the sequential chain.

Rules:
1. KB notation like "Entity >> relation" should become a proper English question,
   e.g. "Paris >> country" → "Which country is Paris in?"
2. When a sub-question references a previous answer, that previous answer is
   unknown at query time — replace the reference with [Answer N] (where N is
   the 1-based hop index of the referenced answer).
   Examples of reference patterns you may see:
     "#1", "#2", "#3"  → [Answer 1], [Answer 2], [Answer 3]
     "answer 1", "ans 1" → [Answer 1]
3. The rewritten questions must together answer the original multi-hop question
   in a chain: hop 1 → hop 2 uses [Answer 1] → hop 3 uses [Answer 2], etc.
4. Each rewritten question should be short, specific, and retrievable
   (i.e. a good keyword search query).
5. Do NOT add, remove, or reorder sub-questions.
6. Return ONLY valid JSON — no markdown, no explanation:
   {"subquestions": ["...", "...", ...]}\
"""

REWRITE_USER = """\
Original multi-hop question: {question}
Number of hops: {n_hops}

Predicted sub-questions (semi-formal reference decomposition, in order):
{numbered_subqs}

Rewrite each sub-question into fluent English.
Use [Answer N] to refer to the answer of hop N when needed.
Return JSON: {{"subquestions": [...]}}\
"""

VERIFY_SYSTEM = """\
You are a quality checker for multi-hop question decompositions.
The sub-questions form a sequential chain: each hop may reference [Answer N]
to denote the result of hop N.

Check the proposed rewritten sub-questions against these criteria:
  A. Fluency    : each sub-question is grammatically correct English.
  B. Completeness: every hop in the reference list is represented.
  C. Chain logic: [Answer N] references are used correctly for sequential steps.
  D. Retrievable : each sub-question, used as a search query, would retrieve
                   a useful passage.

Output ONLY valid JSON:
If ALL criteria pass:
  {"pass": true, "issues": null, "subquestions": [...]}
If ANY criterion fails:
  {"pass": false, "issues": "<one sentence per issue>",
   "subquestions": [...corrected sub-questions...]}\
"""

VERIFY_USER = """\
Original question: {question}
Reference sub-questions (semi-formal):
{ref_subqs}

Proposed rewritten sub-questions:
{rewritten_json}

Evaluate criteria A/B/C/D.
Return JSON with "pass", "issues" (null if pass=true), and "subquestions".\
"""


def extract_json(raw: str) -> dict | None:
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def clean_subquestions(sqs: list[str], original_question: str) -> list[str]:
    cleaned = [
        s.strip()
        for s in sqs
        if isinstance(s, str) and s.strip() and s.strip() != original_question.strip()
    ]
    return cleaned or [original_question]


def normalise_semi_formal_subqs(raw_subqs: list[str]) -> list[str]:
    out: list[str] = []
    for sq in raw_subqs:
        sq = re.sub(r"#\s*(\d+)", lambda m: f"[Answer {m.group(1)}]", sq)
        out.append(sq.strip())
    return out


def build_rewrite_prompt(question: str, semi_formal_hops: list[str]) -> str:
    normed = normalise_semi_formal_subqs(semi_formal_hops)
    numbered = "\n".join(f"  Hop {i + 1}: {sq}" for i, sq in enumerate(normed))
    user = REWRITE_USER.format(
        question=question,
        n_hops=len(normed),
        numbered_subqs=numbered,
    )
    return REWRITE_SYSTEM + "\n\n" + user


def parse_rewrite_response(raw: str, question: str, fallback: list[str]) -> list[str]:
    parsed = extract_json(raw)
    if not parsed or "subquestions" not in parsed:
        return fallback
    sqs = clean_subquestions(parsed["subquestions"], question)
    n = len(fallback)
    if len(sqs) > n:
        sqs = sqs[:n]
    elif len(sqs) < n:
        sqs = sqs + fallback[len(sqs) :]
    return sqs


def verify_and_fix(
    llm: ChatBackend,
    question: str,
    reference_hops: list[str],
    rewritten: list[str],
    max_new_tokens: int,
) -> list[str]:
    ref_numbered = "\n".join(
        f"  Hop {i + 1}: {sq}" for i, sq in enumerate(reference_hops)
    )
    prompt = (
        VERIFY_SYSTEM
        + "\n\n"
        + VERIFY_USER.format(
            question=question,
            ref_subqs=ref_numbered,
            rewritten_json=json.dumps(rewritten, ensure_ascii=False),
        )
    )
    raw = llm.chat(prompt, max_new_tokens)
    parsed = extract_json(raw)
    if not parsed or "subquestions" not in parsed:
        return rewritten
    candidate = clean_subquestions(parsed["subquestions"], question)
    if len(candidate) == len(reference_hops):
        return candidate
    return rewritten


def build_verify_prompt(question: str, reference_hops: list[str], rewritten: list[str]) -> str:
    ref_numbered = "\n".join(
        f"  Hop {i + 1}: {sq}" for i, sq in enumerate(reference_hops)
    )
    return (
        VERIFY_SYSTEM
        + "\n\n"
        + VERIFY_USER.format(
            question=question,
            ref_subqs=ref_numbered,
            rewritten_json=json.dumps(rewritten, ensure_ascii=False),
        )
    )


def parse_verify_response(
    raw: str,
    question: str,
    reference_hops: list[str],
    rewritten: list[str],
) -> list[str]:
    parsed = extract_json(raw)
    if not parsed or "subquestions" not in parsed:
        return rewritten
    candidate = clean_subquestions(parsed["subquestions"], question)
    if len(candidate) == len(reference_hops):
        return candidate
    return rewritten


def verify_and_fix_batch(
    llm: ChatBackend,
    rows: list[dict[str, Any]],
    reference_hops_batch: list[list[str]],
    rewritten_batch: list[list[str]],
    max_new_tokens: int,
) -> list[list[str]]:
    prompts = [
        build_verify_prompt(row["question"], reference_hops, rewritten)
        for row, reference_hops, rewritten in zip(
            rows, reference_hops_batch, rewritten_batch
        )
    ]
    raw_outputs = llm.chat_batch(prompts, max_new_tokens)
    return [
        parse_verify_response(raw, row["question"], reference_hops, rewritten)
        for raw, row, reference_hops, rewritten in zip(
            raw_outputs, rows, reference_hops_batch, rewritten_batch
        )
    ]


def semi_formal_hops_from_row(row: dict[str, Any]) -> list[str]:
    if row.get("predicted_question_decomposition"):
        return [
            str(item.get("question", "")).strip()
            for item in row["predicted_question_decomposition"]
            if str(item.get("question", "")).strip()
        ]
    if row.get("predicted_target"):
        return raw_decomposition_to_v1_hop_texts(str(row["predicted_target"]))
    raise ValueError(f"Row {row.get('id')!r} has no predicted decomposition fields")


def normalize_row_id(row: dict[str, Any]) -> str:
    rid = str(row.get("id", "")).strip()
    if rid.startswith("double__") or rid.startswith("triple_") or rid.startswith("quadruple_"):
        return translate_id(rid)
    return rid


def load_input_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = normalize_row_id(row)
            question = (row.get("question") or "").strip()
            hops = semi_formal_hops_from_row(row)
            if not rid or not question or not hops:
                continue
            rows.append(
                {
                    "id": rid,
                    "question": question,
                    "semi_formal_hops": hops,
                    "num_hops_pred": len(hops),
                    "num_hops_gold": row.get("num_hops_gold"),
                }
            )
            if limit and len(rows) >= limit:
                break
    return rows


def build_record(row: dict[str, Any], sub_questions: list[str]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": row["id"],
        "question": row["question"],
        "sub_questions": sub_questions,
        "num_hops": len(sub_questions),
        "num_hops_pred": row["num_hops_pred"],
    }
    if row.get("num_hops_gold") is not None:
        record["num_hops_gold"] = row["num_hops_gold"]
    return record


def summarize_hop_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    nl_counts = Counter(r["num_hops"] for r in records)
    pred_counts = Counter(r["num_hops_pred"] for r in records)
    summary: dict[str, Any] = {
        "n": len(records),
        "num_hops_distribution": dict(sorted(nl_counts.items())),
        "num_hops_pred_distribution": dict(sorted(pred_counts.items())),
    }
    gold_vals = [r["num_hops_gold"] for r in records if r.get("num_hops_gold") is not None]
    if gold_vals:
        gold_counts = Counter(gold_vals)
        summary["num_hops_gold_distribution"] = dict(sorted(gold_counts.items()))
        summary["hop_count_match_rate"] = sum(
            1 for r in records if r.get("num_hops_gold") == r["num_hops"]
        ) / len(records)
    return summary


def format_hop_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"  n={summary['n']}",
        f"  num_hops (NL output): {summary['num_hops_distribution']}",
        f"  num_hops_pred (semi-formal input): {summary['num_hops_pred_distribution']}",
    ]
    if "num_hops_gold_distribution" in summary:
        lines.append(f"  num_hops_gold: {summary['num_hops_gold_distribution']}")
        lines.append(f"  hop_count_match_rate (NL vs gold): {summary['hop_count_match_rate']:.1%}")
    return "\n".join(lines)


def load_done_ids(output_path: Path, checkpoint_path: Path) -> set[str]:
    if checkpoint_path.exists() and output_path.exists():
        with open(checkpoint_path, encoding="utf-8") as handle:
            return set(json.load(handle))
    if output_path.exists() and not checkpoint_path.exists():
        done: set[str] = set()
        with open(output_path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done.add(str(json.loads(line)["id"]))
        return done
    return set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=("transformers", "vllm"), default="transformers")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint is None:
        args.checkpoint = args.output.with_suffix(".checkpoint.json")

    rows = load_input_rows(args.input, args.limit)
    if not rows:
        raise SystemExit(f"No usable rows in {args.input}")

    done_ids = load_done_ids(args.output, args.checkpoint)
    todo = [row for row in rows if row["id"] not in done_ids]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        todo = todo[:3]
        if args.output.exists():
            pass
        else:
            args.output.touch()

    if args.backend == "vllm":
        llm: ChatBackend = VllmChat(
            model=args.model,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len or None,
        )
    else:
        cfg = llama_base.PipelineConfig(
            model_id=args.model,
            device_map=args.device_map,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            do_sample=False,
        )
        llm = llama_base.LlamaChat(cfg)

    pbar = tqdm(total=len(rows), initial=len(done_ids), desc="decompose_to_nl", unit="q")
    written_records: list[dict[str, Any]] = []
    if done_ids and args.output.exists() and not args.dry_run:
        with open(args.output, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    written_records.append(json.loads(line))
    try:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start : start + args.batch_size]
            prompts = [
                build_rewrite_prompt(row["question"], row["semi_formal_hops"])
                for row in batch
            ]
            raw_outputs = llm.chat_batch(prompts, args.max_new_tokens)

            fallback_batch: list[list[str]] = []
            sub_questions_batch: list[list[str]] = []
            for row, raw in zip(batch, raw_outputs):
                fallback = normalise_semi_formal_subqs(row["semi_formal_hops"])
                try:
                    sub_questions = parse_rewrite_response(raw, row["question"], fallback)
                except Exception as exc:
                    tqdm.write(f"ERROR {row['id']}: {exc}")
                    sub_questions = fallback
                fallback_batch.append(fallback)
                sub_questions_batch.append(sub_questions)

            if not args.no_verify:
                try:
                    sub_questions_batch = verify_and_fix_batch(
                        llm,
                        batch,
                        fallback_batch,
                        sub_questions_batch,
                        max_new_tokens=min(args.max_new_tokens, 512),
                    )
                except Exception as exc:
                    tqdm.write(f"ERROR batch verify failed at offset {start}: {exc}")

            for row, sub_questions in zip(batch, sub_questions_batch):
                record = build_record(row, sub_questions)
                hop_tag = f"k={record['num_hops']}"
                if record.get("num_hops_gold") is not None:
                    hop_tag += f"/gold={record['num_hops_gold']}"

                if args.dry_run:
                    print(json.dumps(record, ensure_ascii=False, indent=2))
                else:
                    with open(args.output, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    done_ids.add(row["id"])
                    written_records.append(record)
                    with open(args.checkpoint, "w", encoding="utf-8") as handle:
                        json.dump(sorted(done_ids), handle, ensure_ascii=False)
                    tqdm.write(f"{row['id']}  {hop_tag}")

                pbar.update(1)
    finally:
        pbar.close()
        llm.unload()

    if not args.dry_run and args.checkpoint.exists() and len(done_ids) >= len(rows):
        args.checkpoint.unlink(missing_ok=True)

    summary_path = args.output.with_suffix(".summary.json")
    if written_records and not args.dry_run:
        summary = summarize_hop_counts(written_records)
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(f"Wrote {len(written_records)} records to {args.output}")
        print(format_hop_summary(summary))
        print(f"Summary: {summary_path}")
    else:
        print(f"Wrote {len(done_ids)} records to {args.output}")


if __name__ == "__main__":
    main()
