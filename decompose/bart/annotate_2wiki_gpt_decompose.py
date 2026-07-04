#!/usr/bin/env python3
"""
Create MuSiQue-style BART decomposition training data for 2WikiMultiHopQA.

The script samples 2Wiki records, asks an OpenAI model to rewrite the gold
supporting-title path as ordered MuSiQue-style component questions, and writes
JSONL records compatible with train.py:
  composed_question_text: original complex question
  component_question_texts: ["[[CQS]] 0 [[CQE]] ...", ...]

Example:
  python annotate_2wiki_gpt_decompose.py \
    --input ../2wikimultihopqa/train.json \
    --output outputs/2wiki_gpt_mixed_train.jsonl \
    --preset mixed_train \
    --model gpt-5-mini
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

DEFAULT_INPUT = Path(__file__).resolve().parent.parent.parent / "data/raw/2wikimultihopqa/train.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data/2wiki_gpt_mixed_train.jsonl"
DEFAULT_MODEL = "gpt-5-mini"
TYPE_SAMPLE_PRESETS = {
    # Roughly 45% bridge-comparison, 25% comparison, 15% compositional, 15% inference.
    # This complements the ~20k MuSiQue training examples without swamping them.
    "mixed_train": {
        "bridge_comparison": 5000,
        "comparison": 2500,
        "compositional": 1500,
        "inference": 1500,
    },
    # Small validation set used together with MuSiQue dev.
    "mixed_dev": {
        "bridge_comparison": 500,
        "comparison": 250,
        "compositional": 150,
        "inference": 150,
    },
    "smoke": {
        "bridge_comparison": 5,
        "comparison": 5,
        "compositional": 5,
        "inference": 5,
    },
}

SYSTEM_PROMPT = """\
You create high-quality question decompositions for multi-hop QA.

Given a 2WikiMultiHopQA question, its gold supporting Wikipedia titles, and its
gold Wikidata evidence triples, write MuSiQue-style component questions for the
BART decomposer.

Rules:
1. Return exactly the requested number of component questions. Each component
   is one retrieval hop intended to retrieve one gold supporting Wikipedia title.
2. Use the evidence triples as semantic guidance, but do not create extra
   sub-questions just because multiple triples belong to the same paragraph.
3. Match MuSiQue raw decomposition style:
   - Prefer "Entity [SEP] relation" for KB-style hops.
   - Use fluent questions only when [SEP] relation form is unnatural.
   - Do NOT include [[CQS]], [[CQE]], [[RQS]], or [[RQE]] markers.
4. If an entity was produced by an earlier component question, refer to it
   as [Answer N] where N is the 1-based sub-question index.
5. For bridge-comparison questions, keep the two branches separate:
   first find both bridge entities, then ask the comparable property for each.
6. Do not include the final comparison or final answer as a component question.
7. Return only valid JSON: {"component_questions": ["...", "..."]}.

Examples:
Question: Which film has the director who was born later, A or B?
Triples: A -- director -- X; B -- director -- Y; X -- date of birth -- d1; Y -- date of birth -- d2
Return: {"component_questions": ["A [SEP] director", "B [SEP] director", "[Answer 1] [SEP] date of birth", "[Answer 2] [SEP] date of birth"]}
"""

USER_PROMPT = """\
Question type: {question_type}
Original question: {question}
Final answer: {answer}
Expected number of component questions: {num_hops}
Hop source: {hop_source}

Gold supporting titles, in retrieval order:
{support_title_lines}

Gold evidence triples, in order:
{evidence_lines}

Write the MuSiQue-style decomposition as JSON.
"""


def load_records(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = data.get("data") or data.get("examples") or data.get("records") or []
    if not isinstance(data, list):
        raise ValueError(f"Expected list-like JSON in {path}")
    return [r for r in data if isinstance(r, dict) and r.get("question") and r.get("evidences")]


def evidence_count(record: dict[str, Any]) -> int:
    return len(record.get("evidences") or [])


def support_titles(record: dict[str, Any]) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for item in record.get("supporting_facts") or []:
        if not isinstance(item, (list, tuple)) or not item:
            continue
        title = str(item[0]).strip()
        if title and title not in seen:
            titles.append(title)
            seen.add(title)
    return titles


def hop_count(record: dict[str, Any], hop_source: str) -> int:
    if hop_source == "evidences":
        return evidence_count(record)
    titles = support_titles(record)
    return len(titles) if titles else evidence_count(record)


def sample_records(
    records: list[dict[str, Any]],
    *,
    types: set[str],
    type_samples: dict[str, int],
    hop_source: str,
    min_hops: int,
    max_hops: int,
    samples_per_type: int,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        qtype = str(record.get("type") or "")
        k = hop_count(record, hop_source)
        if types and qtype not in types:
            continue
        if k < min_hops:
            continue
        if max_hops and k > max_hops:
            continue
        buckets[qtype].append(record)

    selected: list[dict[str, Any]] = []
    for qtype in sorted(buckets):
        bucket = buckets[qtype]
        rng.shuffle(bucket)
        n_for_type = type_samples.get(qtype, samples_per_type)
        if n_for_type:
            bucket = bucket[:n_for_type]
        selected.extend(bucket)

    rng.shuffle(selected)
    if limit:
        selected = selected[:limit]
    return selected


def load_done_ids(output_path: Path) -> set[str]:
    done: set[str] = set()
    if not output_path.exists():
        return done
    with open(output_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                done.add(str(json.loads(line).get("id")))
            except json.JSONDecodeError:
                continue
    return done


def parse_type_samples(raw: str) -> dict[str, int]:
    if not raw.strip():
        return {}
    out: dict[str, int] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(
                f"Invalid --type-samples item {part!r}; expected type=count"
            )
        key, value = part.split("=", 1)
        key = key.strip()
        if key not in {"bridge_comparison", "comparison", "compositional", "inference"}:
            raise ValueError(f"Unknown 2Wiki type in --type-samples: {key!r}")
        out[key] = int(value.strip())
    return out


def format_evidences(evidences: list[list[Any]]) -> str:
    lines: list[str] = []
    for i, triple in enumerate(evidences, start=1):
        subj, rel, obj = (list(triple) + ["", "", ""])[:3]
        lines.append(f"{i}. subject={subj!r}; relation={rel!r}; object={obj!r}")
    return "\n".join(lines)


def format_support_titles(titles: list[str]) -> str:
    if not titles:
        return "(none)"
    return "\n".join(f"{i}. {title}" for i, title in enumerate(titles, start=1))


def build_user_prompt(record: dict[str, Any], hop_source: str) -> str:
    evidences = record.get("evidences") or []
    titles = support_titles(record)
    return USER_PROMPT.format(
        question_type=record.get("type") or "",
        question=record.get("question") or "",
        answer=record.get("answer") or "",
        num_hops=hop_count(record, hop_source),
        hop_source=hop_source,
        support_title_lines=format_support_titles(titles),
        evidence_lines=format_evidences(evidences),
    )


def extract_json(raw: str) -> dict[str, Any] | None:
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None


def normalize_component_question(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*\[SEP\]\s*", " [SEP] ", text)
    text = re.sub(
        r"\[\s*Answer\s+(\d+)\s*\]",
        lambda m: f"[[RQS]] {int(m.group(1)) - 1} [[RQE]]",
        text,
        flags=re.IGNORECASE,
    )
    return text


def component_question_texts(component_questions: list[str]) -> list[str]:
    components: list[str] = []
    for i, component in enumerate(component_questions):
        components.append(f"[[CQS]] {i} [[CQE]] {normalize_component_question(component)}")
    return components


def component_query_texts(component_questions: list[str]) -> list[str]:
    queries: list[str] = []
    for component in component_questions:
        query = re.sub(r"\[\s*Answer\s+\d+\s*\]", "", component, flags=re.IGNORECASE)
        query = query.replace("[SEP]", " ")
        query = re.sub(r"\s+", " ", query).strip()
        queries.append(query)
    return queries


def validate_component_questions(component_questions: Any, expected: int) -> list[str]:
    if not isinstance(component_questions, list):
        raise ValueError("component_questions is not a list")
    cleaned = [str(item).strip() for item in component_questions if str(item).strip()]
    if len(cleaned) != expected:
        raise ValueError(f"expected {expected} component_questions, got {len(cleaned)}")
    return cleaned


def call_openai(client: Any, *, model: str, prompt: str, max_output_tokens: int) -> str:
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=prompt,
        max_output_tokens=max_output_tokens,
    )
    text = getattr(response, "output_text", None)
    if text:
        return str(text)

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                chunks.append(str(value))
    return "\n".join(chunks)


def annotate_record(
    record: dict[str, Any],
    *,
    client: Any,
    model: str,
    hop_source: str,
    max_output_tokens: int,
    max_retries: int,
    retry_sleep: float,
) -> list[str]:
    prompt = build_user_prompt(record, hop_source)
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            raw = call_openai(
                client,
                model=model,
                prompt=prompt,
                max_output_tokens=max_output_tokens,
            )
            parsed = extract_json(raw)
            if not parsed:
                raise ValueError(f"model did not return JSON: {raw[:200]!r}")
            return validate_component_questions(
                parsed.get("component_questions"),
                hop_count(record, hop_source),
            )
        except Exception as exc:  # noqa: BLE001 - preserve record-level progress.
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_sleep * (attempt + 1))
    raise RuntimeError(str(last_error))


def annotate_one(
    record: dict[str, Any],
    *,
    model: str,
    hop_source: str,
    setname: str,
    max_output_tokens: int,
    max_retries: int,
    retry_sleep: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    from openai import OpenAI

    rid = str(record.get("_id") or record.get("id"))
    try:
        component_questions = annotate_record(
            record,
            client=OpenAI(),
            model=model,
            hop_source=hop_source,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )
        out = build_output_record(
            record,
            component_questions,
            model,
            hop_source,
            setname,
        )
        return out, None
    except Exception as exc:  # noqa: BLE001 - return record-level failure.
        return None, {
            "id": rid,
            "type": record.get("type"),
            "question": record.get("question"),
            "error": str(exc),
        }


def build_output_record(
    record: dict[str, Any],
    component_questions: list[str],
    model: str,
    hop_source: str,
    setname: str,
) -> dict[str, Any]:
    components = component_question_texts(component_questions)
    titles = support_titles(record)
    return {
        "dataset": "2wiki",
        "setname": setname,
        "id": record.get("_id") or record.get("id"),
        "type": record.get("type"),
        "composed_question_text": record.get("question"),
        "question_text": " ".join(components),
        "component_question_texts": components,
        "component_query_texts": component_query_texts(component_questions),
        "answer_text": record.get("answer"),
        "answerable": True,
        "evidences": record.get("evidences"),
        "supporting_facts": record.get("supporting_facts"),
        "support_titles": titles,
        "num_hops": len(component_questions),
        "num_evidences": evidence_count(record),
        "hop_source": hop_source,
        "annotation_model": model,
    }


def summarize(records: list[dict[str, Any]], hop_source: str) -> dict[str, Any]:
    by_type = Counter(str(r.get("type") or "") for r in records)
    by_hop_count = Counter(hop_count(r, hop_source) for r in records)
    by_evidence_count = Counter(evidence_count(r) for r in records)
    by_support_title_count = Counter(len(support_titles(r)) for r in records)
    return {
        "n": len(records),
        "by_type": dict(sorted(by_type.items())),
        "hop_source": hop_source,
        "by_hop_count": dict(sorted(by_hop_count.items())),
        "by_support_title_count": dict(sorted(by_support_title_count.items())),
        "by_evidence_count": dict(sorted(by_evidence_count.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--setname",
        choices=("auto", "train", "dev"),
        default="auto",
        help="Value written to the output JSONL setname field.",
    )
    parser.add_argument(
        "--preset",
        choices=("none", *TYPE_SAMPLE_PRESETS.keys()),
        default="none",
        help="Convenience per-type sampling preset.",
    )
    parser.add_argument(
        "--type-samples",
        default="",
        help=(
            "Comma-separated per-type counts, e.g. "
            "bridge_comparison=5000,comparison=2500,compositional=1500,inference=1500. "
            "Overrides --samples-per-type and preset counts for listed types."
        ),
    )
    parser.add_argument(
        "--hop-source",
        choices=("support_titles", "evidences"),
        default="support_titles",
        help="support_titles means one hop retrieves one paragraph/title; evidences means one hop per KB triple.",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=None,
        help="2Wiki question types to annotate. Use empty string with shell quoting only if modifying code.",
    )
    parser.add_argument("--min-hops", type=int, default=1)
    parser.add_argument("--max-hops", type=int, default=0, help="0 means no cap")
    parser.add_argument(
        "--min-evidences",
        type=int,
        default=None,
        help="Deprecated alias for --min-hops when --hop-source evidences.",
    )
    parser.add_argument(
        "--max-evidences",
        type=int,
        default=None,
        help="Deprecated alias for --max-hops when --hop-source evidences.",
    )
    parser.add_argument("--samples-per-type", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent OpenAI requests. Start with 4-8 to avoid rate limits.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_evidences is not None:
        args.min_hops = args.min_evidences
        if args.hop_source != "evidences":
            print("Note: --min-evidences was provided; using it as --min-hops.")
    if args.max_evidences is not None:
        args.max_hops = args.max_evidences
        if args.hop_source != "evidences":
            print("Note: --max-evidences was provided; using it as --max-hops.")

    type_samples = (
        dict(TYPE_SAMPLE_PRESETS[args.preset]) if args.preset != "none" else {}
    )
    type_samples.update(parse_type_samples(args.type_samples))
    requested_types = set(args.types or type_samples.keys() or ["bridge_comparison"])

    records = load_records(args.input)
    setname = args.setname
    if setname == "auto":
        setname = "dev" if "dev" in args.input.name.lower() else "train"
    selected = sample_records(
        records,
        types=requested_types,
        type_samples=type_samples,
        hop_source=args.hop_source,
        min_hops=args.min_hops,
        max_hops=args.max_hops,
        samples_per_type=args.samples_per_type,
        limit=args.limit,
        seed=args.seed,
    )
    if not selected:
        raise SystemExit("No records selected")

    print("Selected:")
    print(json.dumps(summarize(selected, args.hop_source), indent=2, ensure_ascii=False))

    if args.dry_run:
        for record in selected[:3]:
            print("\n--- prompt preview ---")
            print(build_user_prompt(record, args.hop_source))
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output.exists():
        args.output.unlink()

    done_ids = load_done_ids(args.output)
    todo = [record for record in selected if str(record.get("_id") or record.get("id")) not in done_ids]
    failures_path = args.output.with_suffix(".failures.jsonl")
    written = 0

    workers = max(1, args.workers)
    with open(args.output, "a", encoding="utf-8") as out_handle, open(
        failures_path, "a", encoding="utf-8"
    ) as failure_handle:
        if workers == 1:
            iterator = tqdm(todo, desc="annotate_2wiki", unit="q")
            for record in iterator:
                out, failure = annotate_one(
                    record,
                    model=args.model,
                    hop_source=args.hop_source,
                    setname=setname,
                    max_output_tokens=args.max_output_tokens,
                    max_retries=args.max_retries,
                    retry_sleep=args.retry_sleep,
                )
                if out:
                    out_handle.write(json.dumps(out, ensure_ascii=False) + "\n")
                    out_handle.flush()
                    written += 1
                elif failure:
                    failure_handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    failure_handle.flush()
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        annotate_one,
                        record,
                        model=args.model,
                        hop_source=args.hop_source,
                        setname=setname,
                        max_output_tokens=args.max_output_tokens,
                        max_retries=args.max_retries,
                        retry_sleep=args.retry_sleep,
                    )
                    for record in todo
                ]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"annotate_2wiki/{workers}w",
                    unit="q",
                ):
                    out, failure = future.result()
                    if out:
                        out_handle.write(json.dumps(out, ensure_ascii=False) + "\n")
                        out_handle.flush()
                        written += 1
                    elif failure:
                        failure_handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                        failure_handle.flush()

    summary_path = args.output.with_suffix(".summary.json")
    completed: list[dict[str, Any]] = []
    if args.output.exists():
        with open(args.output, encoding="utf-8") as handle:
            completed = [json.loads(line) for line in handle if line.strip()]
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "output": str(args.output),
                "model": args.model,
                "hop_source": args.hop_source,
                "preset": args.preset,
                "type_samples": type_samples,
                "workers": workers,
                "written_this_run": written,
                "completed_total": len(completed),
                "selected": summarize(selected, args.hop_source),
                "completed_by_type": dict(sorted(Counter(r.get("type") for r in completed).items())),
                "completed_by_hops": dict(sorted(Counter(r.get("num_hops") for r in completed).items())),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Wrote {written} new records to {args.output}")
    print(f"Summary: {summary_path}")
    if failures_path.exists():
        print(f"Failures: {failures_path}")


if __name__ == "__main__":
    main()
