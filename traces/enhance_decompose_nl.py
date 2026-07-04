#!/usr/bin/env python3
"""
Paraphrase K=3 / K=4 GT NL decompositions (train only) for gate data augmentation.

Input:  data/decompose/musique/gt/train_nl.jsonl + raw MuSiQue train (hop answers)
Output: data/decompose/musique/gt/train_nl_enhance.jsonl

Two-pass vLLM (Llama-3.1):
  1) paraphrase each hop separately (one sentence per call; [Answer N] preserved)
  2) verify full chain (fluency / chain logic / retrievability)

Usage:
  python enhance_decompose_nl.py --device cuda:0
  python enhance_decompose_nl.py --limit 10 --dry-run
  NUM_VARIANTS=2 ./run_enhance_decompose_train.sh
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Protocol

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from dataset_loaders import get_decompose_path, get_raw_dir, iter_musique
from trace_evidence import gold_hop_answers_musique

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_OUT = PROJECT_ROOT / "data" / "decompose" / "musique" / "gt" / "train_nl_enhance.jsonl"
TARGET_K = frozenset({3, 4})


class ChatBackend(Protocol):
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
        temperature: float,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise SystemExit(
                "vLLM required. Install vllm, then rerun with --backend vllm."
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
        self._SamplingParams = SamplingParams
        self.temperature = temperature

    def _prompt(self, user_text: str) -> str:
        messages = [{"role": "user", "content": user_text}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def chat_batch(
        self,
        user_texts: list[str],
        max_new_tokens: int,
        *,
        temperature: float | None = None,
    ) -> list[str]:
        if not user_texts:
            return []
        prompts = [self._prompt(t) for t in user_texts]
        params = self._SamplingParams(
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_new_tokens,
        )
        outputs = self.llm.generate(prompts, params)
        return [out.outputs[0].text.strip() if out.outputs else "" for out in outputs]

    def unload(self) -> None:
        del self.llm


HOP_PARAPHRASE_SYSTEM = """\
You are an expert in multi-hop question answering.
You paraphrase exactly ONE sub-question per request.

CRITICAL — [Answer N] placeholders:
- If the original sub-question contains tokens like [Answer 1] or [Answer 2],
  you MUST copy every such token verbatim into your output (same N, same spelling).
- Do NOT replace [Answer N] with entity names, pronouns, or phrases like
  "that city", "that country", "that faith", etc.
- Do NOT invent new [Answer N] tokens that were not in the original.

Other rules:
- Return exactly ONE fluent English sentence (a good keyword search query).
- Keep the same retrieval intent; only change wording.
- Do NOT wrap the sentence in JSON, quotes, or bullet points.\
"""

HOP_PARAPHRASE_USER = """\
Original multi-hop question: {question}
Hop {hop_idx} of {n_hops}  |  variant {variant_idx}

Original sub-question for this hop:
{orig_sq}
{placeholder_block}

Paraphrase into ONE sentence (different wording, same meaning).
Output ONLY that single sentence.\
"""

VERIFY_SYSTEM = """\
You are a quality checker for paraphrased multi-hop sub-questions.

The chain uses [Answer N] for unknown prior-hop answers. Those placeholders must
appear exactly where the reference chain uses them.

Check criteria:
  A. Fluency: grammatical English.
  B. Completeness: same hop count as the reference list.
  C. Chain logic: [Answer N] indices match the reference chain on each hop.
  D. Retrievable: each hop is a usable search query.

Output ONLY valid JSON:
If ALL pass:
  {"pass": true, "issues": null, "subquestions": [...]}
If ANY fail:
  {"pass": false, "issues": "<brief issues>",
   "subquestions": [...corrected paraphrases...]}\
"""

VERIFY_USER = """\
Original question: {question}

Reference NL sub-questions:
{ref_numbered}

Proposed paraphrase:
{rewritten_json}

Evaluate A/B/C/D. Return JSON with pass, issues, subquestions.\
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


def clean_subquestions(sqs: list[Any], question: str, n_hops: int) -> list[str]:
    cleaned = [
        str(s).strip()
        for s in sqs
        if isinstance(s, (str, int, float)) and str(s).strip()
        and str(s).strip() != question.strip()
    ]
    if len(cleaned) > n_hops:
        cleaned = cleaned[:n_hops]
    return cleaned


def answer_refs_ok(sqs: list[str], n_hops: int) -> bool:
    """Each hop h may only reference [Answer N] with N < h."""
    for i, sq in enumerate(sqs, start=1):
        refs = {int(m.group(1)) for m in re.finditer(r"\[Answer\s*(\d+)\]", sq, re.I)}
        if any(r >= i or r < 1 for r in refs):
            return False
    return len(sqs) == n_hops


def refs_pattern(sq: str) -> tuple[int, ...]:
    return tuple(
        sorted(int(m.group(1)) for m in re.finditer(r"\[Answer\s*(\d+)\]", sq, re.I))
    )


def placeholder_tokens(sq: str) -> list[str]:
    return re.findall(r"\[Answer\s*\d+\]", sq, re.I)


def placeholder_block(orig_sq: str) -> str:
    tokens = placeholder_tokens(orig_sq)
    if not tokens:
        return "This hop has no [Answer N] placeholders."
    joined = ", ".join(tokens)
    return (
        f"Required placeholders (copy verbatim into your output): {joined}\n"
        "Do NOT replace them with entity names or pronouns."
    )


def strip_one_sentence(raw: str) -> str:
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
    parsed = extract_json(raw)
    if parsed:
        if isinstance(parsed.get("subquestion"), str):
            return parsed["subquestion"].strip()
        sqs = parsed.get("subquestions")
        if isinstance(sqs, list) and sqs:
            return str(sqs[0]).strip()
    line = raw.splitlines()[0].strip() if raw else ""
    return line.strip("\"'").strip()


def hop_paraphrase_ok(
    orig_sq: str,
    para_sq: str,
    *,
    question: str,
    hop_idx: int,
) -> bool:
    para_sq = para_sq.strip()
    if not para_sq or para_sq == question.strip():
        return False
    if refs_pattern(orig_sq) != refs_pattern(para_sq):
        return False
    refs = {int(m.group(1)) for m in re.finditer(r"\[Answer\s*(\d+)\]", para_sq, re.I)}
    if any(r >= hop_idx or r < 1 for r in refs):
        return False
    return True


def build_hop_paraphrase_prompt(
    row: dict[str, Any],
    *,
    variant_idx: int,
    hop_idx: int,
) -> str:
    orig_sq = row["sub_questions"][hop_idx]
    user = HOP_PARAPHRASE_USER.format(
        question=row["question"],
        hop_idx=hop_idx + 1,
        n_hops=len(row["sub_questions"]),
        variant_idx=variant_idx,
        orig_sq=orig_sq,
        placeholder_block=placeholder_block(orig_sq),
    )
    return HOP_PARAPHRASE_SYSTEM + "\n\n" + user


def parse_hop_paraphrase(
    raw: str,
    *,
    orig_sq: str,
    question: str,
    hop_idx: int,
) -> str:
    candidate = strip_one_sentence(raw)
    if hop_paraphrase_ok(orig_sq, candidate, question=question, hop_idx=hop_idx):
        return candidate
    return orig_sq


def chain_refs_match(reference: list[str], candidate: list[str]) -> bool:
    if len(reference) != len(candidate):
        return False
    return all(refs_pattern(a) == refs_pattern(b) for a, b in zip(reference, candidate))


def gold_answer_leak(hop_answers: list[str], sqs: list[str]) -> bool:
    for sq in sqs:
        lower = sq.lower()
        for ans in hop_answers:
            token = ans.strip().rstrip(",").strip()
            if len(token) >= 3 and token.lower() in lower:
                return True
    return False


def programmatic_verify_pass(
    reference: list[str],
    candidate: list[str],
    hop_answers: list[str],
) -> bool:
    if not chain_refs_match(reference, candidate):
        return False
    if not answer_refs_ok(candidate, len(candidate)):
        return False
    if gold_answer_leak(hop_answers, candidate):
        return False
    return True


def build_verify_prompt(row: dict[str, Any], rewritten: list[str]) -> str:
    ref_numbered = "\n".join(
        f"  Hop {i + 1}: {sq}" for i, sq in enumerate(row["sub_questions"])
    )
    user = VERIFY_USER.format(
        question=row["question"],
        ref_numbered=ref_numbered,
        rewritten_json=json.dumps(rewritten, ensure_ascii=False),
    )
    return VERIFY_SYSTEM + "\n\n" + user


def parse_verify_response(
    raw: str,
    *,
    question: str,
    n_hops: int,
    reference: list[str],
    hop_answers: list[str],
    fallback: list[str],
) -> tuple[list[str], bool]:
    parsed = extract_json(raw)
    if not parsed or "subquestions" not in parsed:
        return fallback, programmatic_verify_pass(reference, fallback, hop_answers)
    sqs = clean_subquestions(parsed["subquestions"], question, n_hops)
    if len(sqs) < n_hops:
        return fallback, programmatic_verify_pass(reference, fallback, hop_answers)
    if len(sqs) > n_hops:
        sqs = sqs[:n_hops]
    if not answer_refs_ok(sqs, n_hops):
        return fallback, programmatic_verify_pass(reference, fallback, hop_answers)
    passed = programmatic_verify_pass(reference, sqs, hop_answers)
    return sqs, passed


def paraphrase_rows_batch(
    llm: VllmChat,
    batch: list[tuple[dict[str, Any], int]],
    *,
    max_new_tokens: int,
) -> list[list[str]]:
    """Paraphrase each hop separately; batch all hop prompts for vLLM."""
    hop_specs: list[tuple[int, int]] = []  # (batch_row_idx, hop_idx)
    prompts: list[str] = []
    for bi, (row, vi) in enumerate(batch):
        for hi in range(len(row["sub_questions"])):
            hop_specs.append((bi, hi))
            prompts.append(build_hop_paraphrase_prompt(row, variant_idx=vi, hop_idx=hi))

    hop_tokens = min(max_new_tokens, 256)
    raw_hops = llm.chat_batch(prompts, hop_tokens)

    out: list[list[str]] = [list(row["sub_questions"]) for row, _ in batch]
    for (bi, hi), raw in zip(hop_specs, raw_hops):
        row, _vi = batch[bi]
        out[bi][hi] = parse_hop_paraphrase(
            raw,
            orig_sq=row["sub_questions"][hi],
            question=row["question"],
            hop_idx=hi + 1,
        )
    return out


def load_raw_index(split: str) -> dict[str, dict[str, Any]]:
    raw_dir = get_raw_dir("musique")
    return {
        str(r.get("id", "")).strip(): r
        for r in iter_musique(raw_dir, split)
        if str(r.get("id", "")).strip()
    }


def load_source_rows(
    decompose_path: Path,
    raw_index: dict[str, dict[str, Any]],
    *,
    target_k: frozenset[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(decompose_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rid = str(obj.get("id", "")).strip()
            sqs = [str(s).strip() for s in (obj.get("sub_questions") or []) if str(s).strip()]
            if not rid or len(sqs) not in target_k:
                continue
            raw = raw_index.get(rid)
            if raw is None:
                continue
            hop_answers = gold_hop_answers_musique(raw)
            if len(hop_answers) < len(sqs):
                continue
            rows.append({
                "source_id": rid,
                "question": str(obj.get("question") or raw.get("question") or "").strip(),
                "sub_questions": sqs,
                "hop_answers": hop_answers[: len(sqs)],
                "K": len(sqs),
            })
    return rows


def load_done_ids(output_path: Path, checkpoint_path: Path) -> set[str]:
    if checkpoint_path.exists() and output_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            return set(json.load(f))
    if output_path.exists():
        done: set[str] = set()
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(str(json.loads(line)["id"]))
        return done
    return set()


def build_enhance_record(
    row: dict[str, Any],
    *,
    variant_idx: int,
    paraphrased: list[str],
    verify_pass: bool,
) -> dict[str, Any]:
    eid = f"{row['source_id']}__enh{variant_idx}"
    return {
        "id": eid,
        "source_id": row["source_id"],
        "augment": "llm_paraphrase",
        "variant": variant_idx,
        "K": row["K"],
        "question": row["question"],
        "sub_questions": paraphrased,
        "hop_answers": row["hop_answers"],
        "verify_pass": verify_pass,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Paraphrase K=3/4 GT NL decompose (train).")
    ap.add_argument("--split", choices=("train",), default="train")
    ap.add_argument("--decompose-file", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--num-variants", type=int, default=2,
                    help="Paraphrase variants per source example")
    ap.add_argument("--temperature", type=float, default=0.4,
                    help="Sampling temperature for paraphrase pass")
    ap.add_argument("--verify-temperature", type=float, default=0.0)
    ap.add_argument("--target-k", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    target_k = frozenset(args.target_k)
    decompose_path = args.decompose_file or get_decompose_path("musique", args.split, mode="gt")
    if not decompose_path.is_file():
        sys.exit(f"Missing decompose file: {decompose_path}")

    if args.checkpoint is None:
        args.checkpoint = args.output.with_suffix(".checkpoint.json")

    raw_index = load_raw_index(args.split)
    sources = load_source_rows(decompose_path, raw_index, target_k=target_k)
    if args.limit and args.limit > 0:
        sources = sources[: args.limit]

    jobs: list[tuple[dict[str, Any], int]] = []
    for row in sources:
        for vi in range(args.num_variants):
            jobs.append((row, vi))

    done_ids = load_done_ids(args.output, args.checkpoint)
    todo = [
        (row, vi) for row, vi in jobs
        if f"{row['source_id']}__enh{vi}" not in done_ids
    ]

    print(f"Sources K={sorted(target_k)}: {len(sources)}")
    print(f"Jobs (×{args.num_variants} variants): {len(jobs)}, todo: {len(todo)}")
    print(f"Output: {args.output}")

    if not todo:
        print("Nothing to do.")
        return

    if args.dry_run:
        todo = todo[: min(4, len(todo))]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    llm = VllmChat(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len or None,
        temperature=args.temperature,
    )

    pbar = tqdm(total=len(jobs), initial=len(done_ids), desc="enhance_nl", unit="row")
    try:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start : start + args.batch_size]
            paraphrased_batch = paraphrase_rows_batch(
                llm, batch, max_new_tokens=args.max_new_tokens
            )

            verify_pass_batch = [False] * len(batch)
            if not args.no_verify:
                verify_prompts = [
                    build_verify_prompt(row, paraphrased)
                    for (row, _vi), paraphrased in zip(batch, paraphrased_batch)
                ]
                verify_raw = llm.chat_batch(
                    verify_prompts,
                    min(args.max_new_tokens, 512),
                    temperature=args.verify_temperature,
                )
                for i, ((row, _vi), raw) in enumerate(zip(batch, verify_raw)):
                    n = len(row["sub_questions"])
                    fixed, passed = parse_verify_response(
                        raw,
                        question=row["question"],
                        n_hops=n,
                        reference=row["sub_questions"],
                        hop_answers=row["hop_answers"],
                        fallback=paraphrased_batch[i],
                    )
                    paraphrased_batch[i] = fixed
                    verify_pass_batch[i] = passed
            else:
                verify_pass_batch = [
                    programmatic_verify_pass(
                        row["sub_questions"], para, row["hop_answers"]
                    )
                    for (row, _vi), para in zip(batch, paraphrased_batch)
                ]

            for (row, vi), paraphrased, vpass in zip(
                batch, paraphrased_batch, verify_pass_batch
            ):
                record = build_enhance_record(
                    row, variant_idx=vi, paraphrased=paraphrased, verify_pass=vpass
                )
                if args.dry_run:
                    print(json.dumps(record, ensure_ascii=False, indent=2))
                else:
                    with open(args.output, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    done_ids.add(record["id"])
                    with open(args.checkpoint, "w", encoding="utf-8") as f:
                        json.dump(sorted(done_ids), f, ensure_ascii=False)
                pbar.update(1)
    finally:
        pbar.close()
        llm.unload()

    if not args.dry_run and args.checkpoint.exists() and len(done_ids) >= len(jobs):
        args.checkpoint.unlink(missing_ok=True)

    summary = {
        "split": args.split,
        "target_k": sorted(target_k),
        "num_sources": len(sources),
        "num_variants": args.num_variants,
        "num_jobs": len(jobs),
        "num_written": len(done_ids),
        "output": str(args.output),
        "verify": not args.no_verify,
    }
    summary_path = args.output.with_suffix(".summary.json")
    if not args.dry_run:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Done — {len(done_ids)} records → {args.output}")
        print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
