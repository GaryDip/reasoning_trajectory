#!/usr/bin/env python3
"""
MuSiQue GT decomposition → natural-language sub-questions
using the OpenAI API (async, concurrent).

Usage:
  python gt_decompose_to_nl.py                        # dev, up to 1000 items
  python gt_decompose_to_nl.py --split train
  python gt_decompose_to_nl.py --max 0                # no limit
  python gt_decompose_to_nl.py --concurrency 40       # 40 parallel requests
  python gt_decompose_to_nl.py --no-verify            # skip verify pass
  python gt_decompose_to_nl.py --dry-run              # first 3 items, no write
  python gt_decompose_to_nl.py --split both --max 0
"""

import asyncio
import json
import os
import re
import argparse
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI
from tqdm import tqdm

# ─── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).resolve().parent.parent.parent.parent  # multihop_trace/
MUSIQUE_DIR = BASE_DIR / "data" / "raw" / "musique"
OUTPUT_DIR  = Path(__file__).resolve().parent.parent  # decompose/gt/

DEFAULT_MODEL   = "gpt-4.1-mini"
MAX_PER_DS      = 1000
MAX_RETRIES     = 3
REQUEST_TIMEOUT = 30
DEFAULT_CONCURRENCY = 20

# ─── Prompts ──────────────────────────────────────────────────────────────────

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

GT sub-questions (semi-formal, in order):
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
  B. Completeness: every hop in the original GT list is represented.
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
GT sub-questions (semi-formal):
{gt_subqs}

Proposed rewritten sub-questions:
{rewritten_json}

Evaluate criteria A/B/C/D.
Return JSON with "pass", "issues" (null if pass=true), and "subquestions".\
"""

# ─── Async OpenAI client ───────────────────────────────────────────────────────

def make_client() -> AsyncOpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set.\n"
            "Export it before running:  export OPENAI_API_KEY=sk-..."
        )
    return AsyncOpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT)


async def chat(client: AsyncOpenAI, model: str, system: str, user: str,
               max_tokens: int = 1024) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 * (attempt + 1))
            else:
                raise
    return ""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def extract_json(raw: str) -> Optional[dict]:
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
    s, e = raw.find("{"), raw.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(raw[s: e + 1])
        except json.JSONDecodeError:
            pass
    return None


def clean_subquestions(sqs: list, original_question: str) -> list:
    return [
        s.strip() for s in sqs
        if isinstance(s, str) and s.strip()
        and s.strip() != original_question.strip()
    ] or [original_question]


def normalise_gt_subqs(raw_subqs: list[str]) -> list[str]:
    out = []
    for sq in raw_subqs:
        sq = re.sub(r"#\s*(\d+)", lambda m: f"[Answer {m.group(1)}]", sq)
        out.append(sq.strip())
    return out


# ─── Async two-pass rewrite ───────────────────────────────────────────────────

async def rewrite_gt_decomposition(
    client: AsyncOpenAI,
    model: str,
    question: str,
    gt_decomp: list[dict],
    do_verify: bool = True,
) -> list[str]:
    raw_subqs = [d["question"] for d in gt_decomp]
    normed    = normalise_gt_subqs(raw_subqs)
    n_hops    = len(normed)
    numbered  = "\n".join(f"  Hop {i+1}: {sq}" for i, sq in enumerate(normed))

    # Pass 1: rewrite
    raw1   = await chat(client, model, REWRITE_SYSTEM,
                        REWRITE_USER.format(question=question, n_hops=n_hops,
                                            numbered_subqs=numbered))
    parsed = extract_json(raw1)
    sqs    = clean_subquestions(parsed["subquestions"], question) if (parsed and "subquestions" in parsed) else normed

    if len(sqs) > n_hops:
        sqs = sqs[:n_hops]
    elif len(sqs) < n_hops:
        sqs = sqs + normed[len(sqs):]

    if not do_verify:
        return sqs

    # Pass 2: verify → regen loop
    gt_numbered = "\n".join(f"  Hop {i+1}: {sq}" for i, sq in enumerate(normed))
    for attempt in range(MAX_RETRIES + 1):
        raw_v = await chat(client, model, VERIFY_SYSTEM,
                           VERIFY_USER.format(
                               question=question,
                               gt_subqs=gt_numbered,
                               rewritten_json=json.dumps(sqs, ensure_ascii=False),
                           ))
        v = extract_json(raw_v)
        if not (v and "subquestions" in v):
            break

        passed = bool(v.get("pass", True))
        if passed or attempt == MAX_RETRIES:
            candidate = clean_subquestions(v["subquestions"], question)
            if len(candidate) == n_hops:
                sqs = candidate
            return sqs

        issues = v.get("issues") or ""
        regen_user = (
            f"Original question: {question}\n"
            f"GT sub-questions (semi-formal):\n{gt_numbered}\n\n"
            f"Rejected rewrite:\n{json.dumps(sqs, ensure_ascii=False)}\n\n"
            f"Problems found:\n{issues}\n\n"
            "Produce a corrected rewrite.\n"
            'Return JSON: {"subquestions": [...]}'
        )
        raw_r = await chat(client, model, REWRITE_SYSTEM, regen_user)
        r = extract_json(raw_r)
        if r and "subquestions" in r:
            candidate = clean_subquestions(r["subquestions"], question)
            if len(candidate) == n_hops:
                sqs = candidate

    return sqs


# ─── Record builder ────────────────────────────────────────────────────────────

def build_record(original: dict, nl_subqs: list[str]) -> dict:
    return {
        "id":            original["id"],
        "question":      original["question"],
        "sub_questions": nl_subqs,
    }


# ─── Async dataset processor ──────────────────────────────────────────────────

async def process_dataset(
    client: AsyncOpenAI,
    model: str,
    name: str,
    data: list[dict],
    output_path: Path,
    checkpoint_path: Path,
    do_verify: bool,
    dry_run: bool,
    concurrency: int,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    done_ids: set = set()
    if checkpoint_path.exists() and output_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            done_ids = set(json.load(f))
        print(f"[{name}] Resuming — {len(done_ids)} questions already processed.")
    elif output_path.exists():
        output_path.unlink()

    todo = [item for item in data if item["id"] not in done_ids]
    pbar = tqdm(total=len(data), initial=len(done_ids),
                desc=f"[{name}]", unit="q", dynamic_ncols=True)

    sem      = asyncio.Semaphore(concurrency)
    file_lock = asyncio.Lock()
    dry_count = 0

    async def process_one(item: dict):
        nonlocal dry_count
        orig_id   = item["id"]
        question  = item["question"]
        gt_decomp = item.get("question_decomposition", [])

        if not gt_decomp:
            pbar.update(1)
            return

        try:
            async with sem:
                nl_subqs = await rewrite_gt_decomposition(
                    client, model, question, gt_decomp, do_verify=do_verify
                )
            tqdm.write(f"Q: {question}")
            for i, sq in enumerate(nl_subqs, 1):
                tqdm.write(f"  hop{i}: {sq}")
        except Exception as exc:
            tqdm.write(f"ERROR {orig_id}: {exc}")
            nl_subqs = normalise_gt_subqs([d["question"] for d in gt_decomp])

        record = build_record(item, nl_subqs)

        if dry_run:
            dry_count += 1
            print(json.dumps(record, indent=2, ensure_ascii=False))
            pbar.update(1)
            return

        async with file_lock:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            done_ids.add(orig_id)
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(list(done_ids), f, ensure_ascii=False)

        pbar.update(1)

    tasks = [process_one(item) for item in todo]

    if dry_run:
        # Process sequentially so dry-run stops cleanly after 3
        for item in todo:
            await process_one(item)
            if dry_count >= 3:
                print("[dry-run] Stopping after 3 items.")
                break
    else:
        await asyncio.gather(*tasks)

    pbar.close()

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    total_written = len(done_ids)
    print(f"[{name}] Done — {total_written} records saved to {output_path}\n")


# ─── Main ──────────────────────────────────────────────────────────────────────

async def async_main():
    parser = argparse.ArgumentParser(
        description="Convert MuSiQue GT decompositions to natural-language sub-questions (async)."
    )
    parser.add_argument("--split", choices=["train", "dev", "both"], default="dev")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max", type=int, default=MAX_PER_DS,
                        help="Max questions per split, 0 = no limit")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"Parallel API requests (default: {DEFAULT_CONCURRENCY})")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the verify pass")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process first 3 items, print results, no file writes")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    client = make_client()
    splits = ["train", "dev"] if args.split == "both" else [args.split]

    for split in splits:
        input_path  = MUSIQUE_DIR / f"musique_ans_v1.0_{split}.jsonl"
        output_path = args.output_dir / f"musique_gt_nl_{split}.jsonl"
        checkpoint  = args.output_dir / f"musique_gt_nl_{split}_checkpoint.json"

        print(f"Loading {split} from {input_path} …")
        with open(input_path, encoding="utf-8") as f:
            data = [json.loads(l) for l in f if l.strip()]
        if args.max and args.max > 0:
            data = data[: args.max]
        print(f"  {len(data)} questions  |  concurrency={args.concurrency}\n")

        await process_dataset(
            client      = client,
            model       = args.model,
            name        = split,
            data        = data,
            output_path = output_path,
            checkpoint_path = checkpoint,
            do_verify   = not args.no_verify,
            dry_run     = args.dry_run,
            concurrency = args.concurrency,
        )

    await client.close()


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
