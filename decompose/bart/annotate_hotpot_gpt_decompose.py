#!/usr/bin/env python3
"""
Create MuSiQue-style BART decomposition training data for HotpotQA.

Same purpose as annotate_2wiki_gpt_decompose.py (which added 2Wiki coverage to BART's
training set) -- BART currently has ZERO HotpotQA training examples, it is evaluated on
Hotpot purely zero-shot after training on MuSiQue + 2Wiki only, which is a likely source of
Hotpot's larger GT-vs-BART decompose quality gap. This script samples HotpotQA records,
asks an OpenAI model to rewrite the gold supporting-title path as ordered MuSiQue-style
component questions, and writes JSONL records compatible with train.py (same schema
annotate_2wiki_gpt_decompose.py produces):
  composed_question_text: original complex question
  component_question_texts: ["[[CQS]] 0 [[CQE]] ...", ...]

Unlike 2Wiki, HotpotQA has no KB evidence triples -- supporting_facts only points at
(title, sentence_index) pairs into prose Wikipedia paragraphs (in `context`), and there are
only two question types (bridge, comparison), each almost always exactly 2 hops (one
supporting title each). The gold supporting SENTENCE TEXT (pulled from `context` via
supporting_facts) is given to the annotator as grounding instead of triples, and the system
prompt is adapted to favor fluent natural-language sub-questions (Hotpot's evidence is prose,
not a KB), while keeping the [Answer N] -> [[RQS]]/[[RQE]] bridging convention and the
[[CQS]]/[[CQE]] wrapping so downstream train.py/data_utils.py see an identical shape.

Example:
  python annotate_hotpot_gpt_decompose.py \
    --input ../../data/raw/hotpotqa/hotpot_train_v1.1.json \
    --output outputs/hotpot_gpt_mixed_train.jsonl \
    --preset mixed_train --model gpt-5-mini
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

DEFAULT_INPUT = Path(__file__).resolve().parent.parent.parent / "data/raw/hotpotqa/hotpot_train_v1.1.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data/hotpot_gpt_mixed_train.jsonl"
DEFAULT_MODEL = "gpt-5-mini"
TYPE_SAMPLE_PRESETS = {
    # HotpotQA train is ~90k, roughly 82% bridge / 18% comparison. Sample a chunk comparable
    # in size to the 2Wiki mix (~10.4k) without swamping MuSiQue's ~20k own examples.
    "mixed_train": {
        "bridge": 8000,
        "comparison": 2000,
    },
    "mixed_dev": {
        "bridge": 400,
        "comparison": 100,
    },
    "smoke": {
        "bridge": 5,
        "comparison": 5,
    },
}

SYSTEM_PROMPT = """\
You create high-quality question decompositions for multi-hop QA.

Given a HotpotQA question, its gold supporting Wikipedia titles (in retrieval order), and the
gold supporting sentences from each of those titles, write MuSiQue-style component questions
for the BART decomposer.

Rules:
1. Return exactly the requested number of component questions -- one per gold supporting
   title, in the same order. Each component is one retrieval hop intended to retrieve that
   title's paragraph.
2. Write fluent, natural sub-questions (HotpotQA's evidence is prose, not a knowledge base --
   do not force an "Entity [SEP] relation" style unless it reads naturally).
3. Do NOT include [[CQS]], [[CQE]], [[RQS]], or [[RQE]] markers.
4. "bridge" questions: the first hop finds a bridge entity; the second hop must refer to it as
   [Answer 1] (1-based index of the sub-question that produced it), e.g. "What is the
   nationality of [Answer 1]?". Only bridge to the SPECIFIC entity/fact actually needed by the
   next hop, not the whole first sentence.
5. "comparison" questions: the two entities are usually already named in the original
   question -- write two independent component questions (one per title), asking for the
   comparable property of each. Do NOT include the final comparison itself as a component
   question.
6. Return only valid JSON: {"component_questions": ["...", "..."]}.

Examples:
Question: Which magazine was started first, Arthur's Magazine or First for Women?
Type: comparison
Titles: Arthur's Magazine; First for Women
Return: {"component_questions": ["When was Arthur's Magazine started?", "When was First for Women started?"]}

Question: The director of the romantic comedy "Big Stone Gap" is based in what New York city?
Type: bridge
Titles: Big Stone Gap (film); Adriana Trigiani
Return: {"component_questions": ["Who directed the romantic comedy film Big Stone Gap?", "What New York city is [Answer 1] based in?"]}
"""

USER_PROMPT = """\
Question type: {question_type}
Original question: {question}
Final answer: {answer}
Expected number of component questions: {num_hops}

Gold supporting titles, in retrieval order, with their gold supporting sentences:
{support_lines}

Write the MuSiQue-style decomposition as JSON.
"""


def load_records(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = data.get("data") or data.get("examples") or data.get("records") or []
    if not isinstance(data, list):
        raise ValueError(f"Expected list-like JSON in {path}")
    return [r for r in data if isinstance(r, dict) and r.get("question") and r.get("supporting_facts")]


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


def support_sentences(record: dict[str, Any]) -> dict[str, list[str]]:
    """{title: [gold supporting sentence text, ...]} using supporting_facts' sentence indices
    to pull the actual sentence strings out of `context` (title -> list of sentences)."""
    context_by_title: dict[str, list[str]] = {}
    for item in record.get("context") or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        title, sents = item[0], item[1]
        if isinstance(sents, list):
            context_by_title[str(title).strip()] = [str(s) for s in sents]

    out: dict[str, list[str]] = defaultdict(list)
    for item in record.get("supporting_facts") or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        title, sent_idx = str(item[0]).strip(), item[1]
        sents = context_by_title.get(title) or []
        if isinstance(sent_idx, int) and 0 <= sent_idx < len(sents):
            out[title].append(sents[sent_idx].strip())
    return dict(out)


def hop_count(record: dict[str, Any]) -> int:
    return len(support_titles(record))


# --- Verification: hop-count matching (validate_component_questions, below) only checks
# structure. It says nothing about whether the generated sub-questions are actually USABLE --
# whether [Answer N] points at the right earlier hop, whether the question just leaks the
# thing it's supposed to find, or (most importantly) whether the sub-question, run through the
# SAME retriever the pipeline actually uses downstream, would find the right passage at all.
# No extra LLM call needed for any of this -- all three checks are free (regex + local BGE
# encode against HotpotQA's own `context`, which already ships gold + distractor paragraphs
# for every example, i.e. exactly the pool the sub-question will really be searched against).

_ST_MODEL_CACHE: dict[str, Any] = {}


def _get_st_model(model_name: str):
    if model_name not in _ST_MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        # CPU on purpose: this is a handful of short texts per record, GPU speed buys nothing
        # here, and this script has no other GPU use -- forcing CPU keeps it from competing for
        # scarce GPU memory with the actual training/eval jobs (which is what OOM'd when this
        # picked up the default CUDA device on a shared, already-full GPU during testing).
        _ST_MODEL_CACHE[model_name] = SentenceTransformer(model_name, device="cpu")
    return _ST_MODEL_CACHE[model_name]


def answer_refs_ok(component_questions: list[str]) -> bool:
    """Hop i (1-based) may only reference [Answer N] for N < i -- it can't cite an answer
    that a LATER hop is supposed to produce."""
    for i, sq in enumerate(component_questions, start=1):
        refs = {int(m.group(1)) for m in re.finditer(r"\[Answer\s*(\d+)\]", sq, re.I)}
        if any(r >= i or r < 1 for r in refs):
            return False
    return True


def title_leak(component_questions: list[str], titles: list[str]) -> bool:
    """True if some hop's sub-question text already contains a LATER hop's gold title
    verbatim -- that hop's title is effectively "the answer" that hop is supposed to find, so
    naming it early means the sub-question is trivially answerable without real retrieval,
    not a genuine decomposition step."""
    for i, sq in enumerate(component_questions):
        lower = sq.lower()
        for later_title in titles[i + 1 :]:
            token = later_title.strip()
            if len(token) >= 3 and token.lower() in lower:
                return True
    return False


def fill_answer_placeholders(sq: str, titles: list[str]) -> str:
    """Best-effort stand-in for expand_hop_template() at real inference time: substitute
    [Answer N] with the N-th gold title (the closest available proxy for "the entity found by
    that earlier hop" -- HotpotQA has no separate per-hop answer annotation the way MuSiQue
    does), so hop 2+'s retrieval query is complete enough to actually search with."""

    def _sub(m: "re.Match[str]") -> str:
        idx = int(m.group(1)) - 1
        return titles[idx] if 0 <= idx < len(titles) else m.group(0)

    return re.sub(r"\[Answer\s*(\d+)\]", _sub, sq, flags=re.I)


def retrieval_verify_hops(
    record: dict[str, Any],
    component_questions: list[str],
    titles: list[str],
    *,
    model_name: str,
    top_k: int,
) -> tuple[list[int | None], bool]:
    """For each hop, embed its (placeholder-filled) sub-question and every paragraph in this
    example's own `context` (gold + distractors -- the real candidate pool a retriever would
    see), rank by cosine, and return the 1-based rank of that hop's OWN gold title. All hops
    must land within top_k for the chain to pass -- this directly tests "would this
    sub-question actually retrieve the intended passage", the property that matters most."""
    context = record.get("context") or []
    para_titles = [str(item[0]).strip() for item in context if isinstance(item, (list, tuple)) and item]
    para_texts = [
        f"{item[0]} {' '.join(str(s) for s in item[1])}" if len(item) > 1 and isinstance(item[1], list) else str(item[0])
        for item in context
        if isinstance(item, (list, tuple)) and item
    ]
    if not para_titles:
        return [None] * len(component_questions), False

    model = _get_st_model(model_name)
    doc_vecs = model.encode(para_texts, normalize_embeddings=True, show_progress_bar=False)
    queries = [fill_answer_placeholders(sq, titles) for sq in component_questions]
    query_vecs = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)

    ranks: list[int | None] = []
    for j, gold_title in enumerate(titles):
        if gold_title not in para_titles:
            ranks.append(None)
            continue
        sims = query_vecs[j] @ doc_vecs.T
        order = sims.argsort()[::-1]
        gold_pos = [para_titles[i] for i in order].index(gold_title)
        ranks.append(gold_pos + 1)
    all_pass = all(r is not None and r <= top_k for r in ranks)
    return ranks, all_pass


def sample_records(
    records: list[dict[str, Any]],
    *,
    types: set[str],
    type_samples: dict[str, int],
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
        k = hop_count(record)
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
            raise ValueError(f"Invalid --type-samples item {part!r}; expected type=count")
        key, value = part.split("=", 1)
        key = key.strip()
        if key not in {"bridge", "comparison"}:
            raise ValueError(f"Unknown HotpotQA type in --type-samples: {key!r}")
        out[key] = int(value.strip())
    return out


def format_support_lines(titles: list[str], sentences: dict[str, list[str]]) -> str:
    if not titles:
        return "(none)"
    lines = []
    for i, title in enumerate(titles, start=1):
        sents = " ".join(sentences.get(title, [])) or "(no sentence text found)"
        lines.append(f"{i}. {title}: {sents}")
    return "\n".join(lines)


def build_user_prompt(record: dict[str, Any]) -> str:
    titles = support_titles(record)
    sentences = support_sentences(record)
    return USER_PROMPT.format(
        question_type=record.get("type") or "",
        question=record.get("question") or "",
        answer=record.get("answer") or "",
        num_hops=hop_count(record),
        support_lines=format_support_lines(titles, sentences),
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
    return [f"[[CQS]] {i} [[CQE]] {normalize_component_question(q)}" for i, q in enumerate(component_questions)]


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
    max_output_tokens: int,
    max_retries: int,
    retry_sleep: float,
) -> list[str]:
    prompt = build_user_prompt(record)
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            raw = call_openai(client, model=model, prompt=prompt, max_output_tokens=max_output_tokens)
            parsed = extract_json(raw)
            if not parsed:
                raise ValueError(f"model did not return JSON: {raw[:200]!r}")
            return validate_component_questions(parsed.get("component_questions"), hop_count(record))
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
    setname: str,
    max_output_tokens: int,
    max_retries: int,
    retry_sleep: float,
    verify: bool,
    retrieval_model: str,
    retrieval_top_k: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    from openai import OpenAI

    rid = str(record.get("_id") or record.get("id"))
    try:
        component_questions = annotate_record(
            record,
            client=OpenAI(),
            model=model,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )
        out = build_output_record(
            record, component_questions, model, setname,
            verify=verify, retrieval_model=retrieval_model, retrieval_top_k=retrieval_top_k,
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
    setname: str,
    *,
    verify: bool,
    retrieval_model: str,
    retrieval_top_k: int,
) -> dict[str, Any]:
    components = component_question_texts(component_questions)
    titles = support_titles(record)

    verify_pass = True
    verify_issues: list[str] = []
    retrieval_ranks: list[int | None] = []
    if verify:
        if not answer_refs_ok(component_questions):
            verify_pass = False
            verify_issues.append("forward_answer_ref")
        if title_leak(component_questions, titles):
            verify_pass = False
            verify_issues.append("title_leak")
        retrieval_ranks, retrieval_ok = retrieval_verify_hops(
            record, component_questions, titles, model_name=retrieval_model, top_k=retrieval_top_k
        )
        if not retrieval_ok:
            verify_pass = False
            verify_issues.append("retrieval_rank")

    return {
        "dataset": "hotpotqa",
        "setname": setname,
        "id": record.get("_id") or record.get("id"),
        "type": record.get("type"),
        "composed_question_text": record.get("question"),
        "question_text": " ".join(components),
        "component_question_texts": components,
        "component_query_texts": component_query_texts(component_questions),
        "answer_text": record.get("answer"),
        "answerable": True,
        "supporting_facts": record.get("supporting_facts"),
        "support_titles": titles,
        "num_hops": len(component_questions),
        "annotation_model": model,
        "verify_pass": verify_pass,
        "verify_issues": verify_issues,
        "retrieval_ranks": retrieval_ranks,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(str(r.get("type") or "") for r in records)
    by_hop_count = Counter(hop_count(r) for r in records)
    return {
        "n": len(records),
        "by_type": dict(sorted(by_type.items())),
        "by_hop_count": dict(sorted(by_hop_count.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--setname", choices=("auto", "train", "dev"), default="auto",
                         help="Value written to the output JSONL setname field.")
    parser.add_argument("--preset", choices=("none", *TYPE_SAMPLE_PRESETS.keys()), default="none",
                         help="Convenience per-type sampling preset.")
    parser.add_argument("--type-samples", default="",
                         help="Comma-separated per-type counts, e.g. bridge=8000,comparison=2000. "
                              "Overrides --samples-per-type and preset counts for listed types.")
    parser.add_argument("--types", nargs="+", default=None,
                         help="HotpotQA question types to annotate (bridge, comparison).")
    parser.add_argument("--min-hops", type=int, default=1)
    parser.add_argument("--max-hops", type=int, default=0, help="0 means no cap")
    parser.add_argument("--samples-per-type", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=1,
                         help="Number of concurrent OpenAI requests. Start with 4-8 to avoid rate limits.")
    parser.add_argument("--verify", dest="verify", action="store_true", default=True,
                         help="Free, no-extra-API-call correctness checks per record: [Answer N] "
                              "forward-reference check, gold-title leakage check, and a real BGE "
                              "retrieval check against this example's own context pool (gold + "
                              "distractor paragraphs) -- does each hop's sub-question actually rank "
                              "its gold title within --retrieval-top-k. Sets verify_pass/verify_issues/"
                              "retrieval_ranks on every output record; does not drop failing records "
                              "(filter on verify_pass=true before using for BART training).")
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    parser.add_argument("--retrieval-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--retrieval-top-k", type=int, default=1,
                         help="How strict the retrieval check is: gold title must rank within "
                              "this many positions. Default 1 (must be the literal top match) "
                              "on purpose -- with only ~10 candidates per HotpotQA example "
                              "(gold + distractors), top_k=3 measurably passes garbage queries "
                              "by chance (~10%% in a quick 20-example check); top_k=1 did not "
                              "(0/20) on the same check.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    type_samples = dict(TYPE_SAMPLE_PRESETS[args.preset]) if args.preset != "none" else {}
    type_samples.update(parse_type_samples(args.type_samples))
    requested_types = set(args.types or type_samples.keys() or ["bridge", "comparison"])

    records = load_records(args.input)
    setname = args.setname
    if setname == "auto":
        setname = "dev" if "dev" in args.input.name.lower() else "train"
    selected = sample_records(
        records,
        types=requested_types,
        type_samples=type_samples,
        min_hops=args.min_hops,
        max_hops=args.max_hops,
        samples_per_type=args.samples_per_type,
        limit=args.limit,
        seed=args.seed,
    )
    if not selected:
        raise SystemExit("No records selected")

    print("Selected:")
    print(json.dumps(summarize(selected), indent=2, ensure_ascii=False))

    if args.dry_run:
        for record in selected[:3]:
            print("\n--- prompt preview ---")
            print(build_user_prompt(record))
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
            iterator = tqdm(todo, desc="annotate_hotpot", unit="q")
            for record in iterator:
                out, failure = annotate_one(
                    record, model=args.model, setname=setname,
                    max_output_tokens=args.max_output_tokens,
                    max_retries=args.max_retries, retry_sleep=args.retry_sleep,
                    verify=args.verify, retrieval_model=args.retrieval_model,
                    retrieval_top_k=args.retrieval_top_k,
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
                        annotate_one, record, model=args.model, setname=setname,
                        max_output_tokens=args.max_output_tokens,
                        max_retries=args.max_retries, retry_sleep=args.retry_sleep,
                        verify=args.verify, retrieval_model=args.retrieval_model,
                        retrieval_top_k=args.retrieval_top_k,
                    )
                    for record in todo
                ]
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"annotate_hotpot/{workers}w", unit="q"):
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
                "preset": args.preset,
                "type_samples": type_samples,
                "workers": workers,
                "written_this_run": written,
                "completed_total": len(completed),
                "selected": summarize(selected),
                "completed_by_type": dict(sorted(Counter(r.get("type") for r in completed).items())),
                "completed_by_hops": dict(sorted(Counter(r.get("num_hops") for r in completed).items())),
                "verify_pass_total": sum(1 for r in completed if r.get("verify_pass")),
                "verify_issue_counts": dict(sorted(Counter(
                    issue for r in completed for issue in (r.get("verify_issues") or [])
                ).items())),
            },
            handle, indent=2, ensure_ascii=False,
        )
    n_pass = sum(1 for r in completed if r.get("verify_pass"))
    print(f"Wrote {written} new records to {args.output}")
    print(f"verify_pass: {n_pass}/{len(completed)} of all completed records so far")
    print(f"Summary: {summary_path}")
    if failures_path.exists():
        print(f"Failures: {failures_path}")


if __name__ == "__main__":
    main()
