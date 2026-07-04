"""MuSiQue official raw-format helpers for BART decomposer training."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

from transformers import BartTokenizer, PreTrainedTokenizerBase

CONSTITUENT_QUESTION_START = "[[CQS]]"
CONSTITUENT_QUESTION_END = "[[CQE]]"
REPLACEMENT_QUESTION_START = "[[RQS]]"
REPLACEMENT_QUESTION_END = "[[RQE]]"

BART_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BART_ROOT.parent.parent

DEFAULT_MODEL_NAME = "facebook/bart-large"
DEFAULT_RAW_TRAIN_FILE = (
    BART_ROOT / "data/musique_raw/musique_ans_gold_context_version_train.jsonl"
)
DEFAULT_RAW_DEV_FILE = (
    BART_ROOT / "data/musique_raw/musique_ans_gold_context_version_dev.jsonl"
)
DEFAULT_RAW_PREDICT_FILE = (
    PROJECT_ROOT / "data/raw/musique/musique_ans_v1.0_dev.jsonl"
)
DEFAULT_2WIKI_DEV_FILE = PROJECT_ROOT / "data/raw/2wikimultihopqa/dev.json"
DEFAULT_HOTPOT_DEV_FILE = PROJECT_ROOT / "data/raw/hotpotqa/hotpot_dev_distractor_v1.json"
DEFAULT_2WIKI_GPT_TRAIN_FILE = BART_ROOT / "data/2wiki_gpt_mixed_train.jsonl"
DEFAULT_2WIKI_GPT_DEV_FILE = BART_ROOT / "data/2wiki_gpt_mixed_dev.jsonl"

_CQS_HEADER_RE = re.compile(
    rf"^{re.escape(CONSTITUENT_QUESTION_START)}\s+\d+\s+{re.escape(CONSTITUENT_QUESTION_END)}\s*"
)


def extra_special_tokens(max_index: int = 5) -> list[str]:
    tokens: list[str] = []
    for index in range(max_index):
        tokens.append(f"{CONSTITUENT_QUESTION_START} {index} {CONSTITUENT_QUESTION_END}")
        tokens.append(f"{REPLACEMENT_QUESTION_START} {index} {REPLACEMENT_QUESTION_END}")
    return tokens


def setup_bart_tokenizer(model_name: str) -> BartTokenizer:
    """Match official question_translator: register CQS/RQS markers as single tokens."""
    tokenizer = BartTokenizer.from_pretrained(model_name)
    additional = extra_special_tokens()
    num_added = tokenizer.add_tokens(additional)
    if num_added != len(additional):
        raise RuntimeError(
            f"Expected to add {len(additional)} tokens, but only {num_added} were new."
        )
    for token in additional:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded).strip() != token.strip():
            raise RuntimeError(f"Added token {token!r} does not map to a single tokenizer id.")
    return tokenizer


def wrap_target_with_special_tokens(tokenizer: PreTrainedTokenizerBase, target: str) -> str:
    """Same wrapping as official QuestionTranslatorReader."""
    parts = [
        tokenizer.bos_token or "",
        target.strip(),
        tokenizer.eos_token or "",
    ]
    return " ".join(part for part in parts if part).strip()


def decomposed_text_from_record(record: dict[str, Any]) -> str:
    components = record.get("component_question_texts") or []
    if not components:
        raise ValueError(f"Record {record.get('id')!r} missing component_question_texts")
    return " ".join(str(part).strip() for part in components if str(part).strip())


def record_to_pair(record: dict[str, Any]) -> tuple[str, str]:
    source = (record.get("composed_question_text") or "").strip()
    target = decomposed_text_from_record(record)
    if not source or not target:
        raise ValueError(f"Invalid record {record.get('id')!r}: missing source or decomposition")
    return source, target


def target_to_hops(target: str) -> list[str]:
    if not target or not target.strip():
        return []

    hops: list[str] = []
    for chunk in target.split(CONSTITUENT_QUESTION_START)[1:]:
        chunk = chunk.strip()
        if not chunk or CONSTITUENT_QUESTION_END not in chunk:
            continue
        hop = chunk.split(CONSTITUENT_QUESTION_END, 1)[1].strip()
        if hop:
            hops.append(hop)
    return hops


def strip_step_prefix(text: str) -> str:
    return _CQS_HEADER_RE.sub("", text.strip(), count=1)


def step_placeholder(index: int, *, is_prefix: bool, strip: bool = False) -> str:
    if is_prefix:
        text = f"{CONSTITUENT_QUESTION_START} {index} {CONSTITUENT_QUESTION_END} "
    else:
        text = f"{REPLACEMENT_QUESTION_START} {index} {REPLACEMENT_QUESTION_END} "
    return text.strip() if strip else text


def translate_id(raw_id: str) -> str:
    """Same id mapping as official raw_data_to_official_format.py."""
    namechange = {
        "double": "2hop",
        "triple_ii": "3hop1",
        "triple_io": "3hop2",
        "quadruple_iii": "4hop1",
        "quadruple_iot": "4hop2",
        "quadruple_ioh1": "4hop3",
        "quadruple_ioh2": "4hop4",
    }
    if sum(original in raw_id for original in namechange) != 1:
        raise ValueError(f"Unrecognized raw id shape: {raw_id!r}")
    for original, new in namechange.items():
        raw_id = raw_id.replace(original, new)
    return raw_id


def normalize_sep_markers(text: str) -> str:
    """Normalize common model typos before marker conversion."""
    normalized = text
    for typo in ("[SE P]", "[SEp]", "[se p]"):
        normalized = normalized.replace(typo, "[SEP]")
    return normalized


def normalize_hop_text(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text.strip())
    return re.sub(r"\s>>\s", " >> ", collapsed).strip()


def raw_decomposition_to_v1_hop_texts(decomposed_question_text: str) -> list[str]:
    """
    Convert raw [[CQS]]/[SEP]/[[RQS]] decomposition to v1.0 hop strings (>> / #i).

    Same string logic as official get_decomposed_question_texts(); works on predicted text.
    """
    text = normalize_sep_markers(decomposed_question_text.strip())
    relation_separator = ">>"
    if relation_separator in text:
        raise ValueError(f"Unexpected {relation_separator!r} in raw decomposition: {text[:120]!r}")

    text = text.replace("[SEP]", relation_separator)
    hop_break = "<--UNLIKELY_BREAK-->"
    for index in range(5):
        text = text.replace(step_placeholder(index, is_prefix=True, strip=True), hop_break)
        text = text.replace(step_placeholder(index, is_prefix=False, strip=True), f"#{index + 1}")

    return [normalize_hop_text(part) for part in text.split(hop_break) if part.strip()]


def build_v1_decomposition_record(
    prediction_row: dict[str, Any],
    *,
    include_gold: bool = False,
) -> dict[str, Any]:
    pred_hops_v1 = raw_decomposition_to_v1_hop_texts(prediction_row["predicted_target"])
    record = {
        "id": translate_id(str(prediction_row["id"])),
        "raw_id": prediction_row["id"],
        "question": prediction_row.get("question") or "",
        "predicted_question_decomposition": [{"question": hop} for hop in pred_hops_v1],
        "num_hops_pred": len(pred_hops_v1),
    }
    if include_gold:
        gold_hops_v1 = raw_decomposition_to_v1_hop_texts(prediction_row["gold_target"])
        record["gold_question_decomposition"] = [{"question": hop} for hop in gold_hops_v1]
        record["num_hops_gold"] = len(gold_hops_v1)
    return record


def iter_raw_records(path: str | Path, *, limit: int = 0) -> Iterator[dict[str, Any]]:
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("composed_question_text"):
                continue
            if not record.get("component_question_texts"):
                continue
            yield record
            count += 1
            if limit and count >= limit:
                break


def load_raw_records(path: str | Path, *, limit: int = 0) -> list[dict[str, Any]]:
    return list(iter_raw_records(path, limit=limit))


def infer_dataset_name(path: str | Path, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    lower = str(path).lower()
    if "2wiki" in lower:
        return "2wiki"
    if "hotpot" in lower:
        return "hotpot"
    return "musique"


def default_predict_file_for_dataset(dataset: str) -> Path:
    if dataset == "2wiki":
        return DEFAULT_2WIKI_DEV_FILE
    if dataset == "hotpot":
        return DEFAULT_HOTPOT_DEV_FILE
    return DEFAULT_RAW_PREDICT_FILE


def record_id(record: dict[str, Any], dataset: str) -> Any:
    if dataset in {"2wiki", "hotpot"}:
        return record.get("_id") or record.get("id")
    return record.get("id") or record.get("_id")


def record_question(record: dict[str, Any], dataset: str) -> str:
    if dataset in {"2wiki", "hotpot"}:
        return str(record.get("question") or "").strip()
    return str(record.get("composed_question_text") or record.get("question") or "").strip()


def unique_support_titles(record: dict[str, Any]) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for item in record.get("supporting_facts") or []:
        if not isinstance(item, (list, tuple)) or not item:
            continue
        title = str(item[0]).strip()
        if title and title not in seen:
            seen.add(title)
            titles.append(title)
    return titles


def gold_support_hop_count(record: dict[str, Any], dataset: str) -> int | None:
    if dataset == "2wiki":
        evidences = record.get("evidences") or []
        if evidences:
            return len(evidences)
    if dataset in {"2wiki", "hotpot"}:
        titles = unique_support_titles(record)
        if titles:
            return len(titles)
    return None


def iter_prediction_records(
    path: str | Path,
    *,
    dataset: str,
    limit: int = 0,
) -> Iterator[dict[str, Any]]:
    """
    Yield records for decomposer inference.

    MuSiQue raw JSONL can include gold decomposition and is used for evaluation.
    2WikiQA/HotpotQA JSON files provide only the composed question here, so they
    are prediction-only unless gold decomposition fields are added upstream.
    """
    if dataset == "musique":
        yield from iter_raw_records(path, limit=limit)
        return

    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = data.get("data") or data.get("examples") or data.get("records") or []
    if not isinstance(data, list):
        raise ValueError(f"Expected list-like JSON for {dataset}: {path}")

    for record in data:
        if not isinstance(record, dict):
            continue
        if not record_question(record, dataset):
            continue
        yield record
        count += 1
        if limit and count >= limit:
            break


def build_prediction_record(
    record: dict[str, Any],
    predicted_target: str,
) -> dict[str, Any]:
    gold_target = decomposed_text_from_record(record)
    gold_hops = target_to_hops(gold_target)
    pred_hops = target_to_hops(predicted_target)
    return {
        "id": record.get("id"),
        "question": record.get("composed_question_text"),
        "predicted_target": predicted_target,
        "predicted_hops": pred_hops,
        "num_hops_pred": len(pred_hops),
        "gold_target": gold_target,
        "gold_hops": gold_hops,
        "num_hops_gold": len(gold_hops),
    }


def build_inference_prediction_record(
    record: dict[str, Any],
    predicted_target: str,
    *,
    dataset: str,
) -> dict[str, Any]:
    if dataset == "musique":
        row = build_prediction_record(record, predicted_target)
        row["dataset"] = "musique"
        return row

    pred_hops = target_to_hops(predicted_target)
    row = {
        "dataset": dataset,
        "id": record_id(record, dataset),
        "question": record_question(record, dataset),
        "predicted_target": predicted_target,
        "predicted_hops": pred_hops,
        "num_hops_pred": len(pred_hops),
    }
    gold_k = gold_support_hop_count(record, dataset)
    if gold_k is not None:
        row["num_hops_gold"] = gold_k
    support_titles = unique_support_titles(record)
    if support_titles:
        row["gold_support_titles"] = support_titles
        row["num_support_titles"] = len(support_titles)
    if record.get("supporting_facts"):
        row["num_supporting_facts"] = len(record["supporting_facts"])
    if dataset == "2wiki" and record.get("evidences"):
        row["num_evidences"] = len(record["evidences"])
    for key in ("answer", "type", "level"):
        if key in record:
            row[key] = record[key]
    return row
