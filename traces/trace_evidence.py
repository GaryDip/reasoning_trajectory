"""Gold / wrong evidence helpers + GPU batched cosine retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_RANKER_CACHE: dict[tuple[str, str, int], CosinePoolRanker] = {}


def expand_hop_template(template: str, prior_answers: list[str]) -> str:
    def repl(m: re.Match[str]) -> str:
        n = int(m.group(1))
        i = n - 1
        if 0 <= i < len(prior_answers):
            return prior_answers[i]
        return m.group(0)

    return re.sub(r"\[Answer\s*(\d+)\]", repl, template, flags=re.IGNORECASE)


def gold_hop_answers_musique(record: dict[str, Any]) -> list[str]:
    return [
        str(h.get("answer", "")).strip()
        for h in (record.get("question_decomposition") or [])
    ]


def gold_evidence_musique(record: dict[str, Any]) -> tuple[list[str], list[int]]:
    paragraphs = list(record.get("paragraphs") or [])
    texts: list[str] = []
    idxs: list[int] = []
    for hop in record.get("question_decomposition") or []:
        pidx = int(hop.get("paragraph_support_idx", -1))
        idxs.append(pidx)
        if 0 <= pidx < len(paragraphs):
            texts.append(str(paragraphs[pidx].get("paragraph_text", "")))
        else:
            texts.append("")
    return texts, idxs


def paragraph_doc_text(p: dict[str, Any]) -> str:
    return f"{p.get('title') or ''}\n{(p.get('paragraph_text') or '').strip()}".strip()


def build_hop_queries(sub_qs: list[str], hop_answers: list[str]) -> list[str]:
    """Expanded retrieval query per hop (1-indexed order)."""
    prior: list[str] = []
    queries: list[str] = []
    for i, sq in enumerate(sub_qs):
        queries.append(expand_hop_template(sq, prior))
        if i < len(hop_answers):
            prior.append(hop_answers[i])
    return queries


def resolve_device(device: str | None) -> str:
    if device:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass
class ExampleRetrieval:
    rid: str
    k: int
    paragraphs: list[dict[str, Any]]
    gold_idxs: set[int]
    queries: list[str] = field(default_factory=list)
    doc_emb: np.ndarray | None = None  # [n_docs, dim]


class CosinePoolRanker:
    """BGE cosine ranker: one pool encode per example (cached across hops), batched queries."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        *,
        device: str | None = None,
        encode_batch_size: int = 64,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = resolve_device(device)
        self.encode_batch_size = encode_batch_size
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=self.encode_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )

    def encode_paragraph_pool(self, paragraphs: list[dict[str, Any]]) -> np.ndarray:
        docs = [paragraph_doc_text(p) for p in paragraphs]
        return self.encode_texts(docs)

    def topk_non_gold_from_emb(
        self,
        query_emb: np.ndarray,
        doc_emb: np.ndarray,
        paragraphs: list[dict[str, Any]],
        gold_idxs: set[int],
        *,
        topk: int = 10,
        max_wrong: int = 3,
    ) -> list[tuple[str, int, float]]:
        if doc_emb.shape[0] == 0:
            return []
        sim = doc_emb @ query_emb
        order = np.argsort(-sim)[:topk]
        wrong: list[tuple[str, int, float]] = []
        for i in order:
            para = paragraphs[int(i)]
            pidx = int(para.get("idx", int(i)))
            if pidx in gold_idxs:
                continue
            text = str(para.get("paragraph_text", "")).strip()
            if not text:
                continue
            wrong.append((text, pidx, float(sim[int(i)])))
            if len(wrong) >= max_wrong:
                break
        return wrong

    def wrongs_per_hop(
        self,
        ex: ExampleRetrieval,
        *,
        topk: int = 10,
        max_wrong: int = 3,
    ) -> list[list[tuple[str, int, float]]]:
        """Requires ex.doc_emb set; batch-encodes all hop queries for one example."""
        if ex.doc_emb is None:
            ex.doc_emb = self.encode_paragraph_pool(ex.paragraphs)
        q_emb = self.encode_texts(ex.queries)
        out: list[list[tuple[str, int, float]]] = []
        for i in range(len(ex.queries)):
            out.append(
                self.topk_non_gold_from_emb(
                    q_emb[i],
                    ex.doc_emb,
                    ex.paragraphs,
                    ex.gold_idxs,
                    topk=topk,
                    max_wrong=max_wrong,
                )
            )
        return out

    def process_batch(
        self,
        examples: list[ExampleRetrieval],
        *,
        topk: int = 10,
        max_wrong: int = 3,
    ) -> list[list[list[tuple[str, int, float]]]]:
        """
        Batch across examples; cache pool embedding per example for all hops:
          1) flat-batch encode each example's paragraph pool once → ex.doc_emb
          2) flat-batch encode all hop queries
          3) cosine top-k non-gold per (example, hop), reusing cached doc_emb
        No cache across batches (each process_batch call is independent).
        """
        if not examples:
            return []

        flat_docs: list[str] = []
        spans: list[tuple[int, int]] = []
        for ex in examples:
            start = len(flat_docs)
            flat_docs.extend(paragraph_doc_text(p) for p in ex.paragraphs)
            spans.append((start, len(flat_docs)))

        flat_doc_emb = self.encode_texts(flat_docs)
        for ex, (s, e) in zip(examples, spans):
            ex.doc_emb = flat_doc_emb[s:e]

        flat_queries: list[str] = []
        q_spans: list[tuple[int, int]] = []
        for ex in examples:
            s = len(flat_queries)
            flat_queries.extend(ex.queries)
            q_spans.append((s, len(flat_queries)))

        flat_q_emb = self.encode_texts(flat_queries)

        results: list[list[list[tuple[str, int, float]]]] = []
        for ex, (qs, qe) in zip(examples, q_spans):
            hop_wrongs: list[list[tuple[str, int, float]]] = []
            for i in range(qs, qe):
                hop_wrongs.append(
                    self.topk_non_gold_from_emb(
                        flat_q_emb[i],
                        ex.doc_emb,  # type: ignore[arg-type]
                        ex.paragraphs,
                        ex.gold_idxs,
                        topk=topk,
                        max_wrong=max_wrong,
                    )
                )
            results.append(hop_wrongs)
        return results


def get_ranker(
    model_name: str = "BAAI/bge-base-en-v1.5",
    *,
    device: str | None = None,
    encode_batch_size: int = 64,
) -> CosinePoolRanker:
    dev = resolve_device(device)
    key = (model_name, dev, encode_batch_size)
    if key not in _RANKER_CACHE:
        _RANKER_CACHE[key] = CosinePoolRanker(
            model_name, device=dev, encode_batch_size=encode_batch_size
        )
    return _RANKER_CACHE[key]


def cosine_non_gold_wrong_texts(
    query: str,
    paragraphs: list[dict[str, Any]],
    gold_idxs: set[int],
    *,
    topk: int = 10,
    max_wrong: int = 3,
    cos_model: str = "BAAI/bge-base-en-v1.5",
    device: str | None = None,
    encode_batch_size: int = 64,
) -> list[tuple[str, int, float]]:
    """Single-query helper (encodes pool once). Prefer batch API for bulk work."""
    ranker = get_ranker(cos_model, device=device, encode_batch_size=encode_batch_size)
    doc_emb = ranker.encode_paragraph_pool(paragraphs)
    q_emb = ranker.encode_texts([query])[0]
    return ranker.topk_non_gold_from_emb(
        q_emb, doc_emb, paragraphs, gold_idxs, topk=topk, max_wrong=max_wrong
    )
