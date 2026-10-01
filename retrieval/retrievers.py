"""
Retriever abstraction: HOW candidates in a CandidatePool get scored against a
query, independent of where the pool came from (candidate_pool.py) or
whether results get reranked afterward (rerankers.py).

Local-pool scoring (pool.precomputed is None) delegates to
run_retrieval_exp.embed_retrieval / colbert_retrieval unchanged — those
functions are not modified or duplicated, so "does this match the original
script's behavior" is true by construction, not by careful copying.

Global-pool scoring (pool.precomputed is a cache payload from
embedding_cache.py) is new: only the query gets encoded per call; passage
vectors come from the precomputed matrix/ragged-tensor instead of being
recomputed.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from candidate_pool import CandidatePool
from run_retrieval_exp import colbert_retrieval, embed_retrieval

_ST_CACHE: dict[str, Any] = {}
_COLBERT_CKPT_CACHE: dict[str, Any] = {}


class Retriever(ABC):
    @abstractmethod
    def score(self, query: str, pool: CandidatePool, k: int) -> list[tuple[dict, float]]:
        """Return top-k (paragraph, score) sorted descending."""


class CosineRetriever(Retriever):
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name

    def score(self, query: str, pool: CandidatePool, k: int) -> list[tuple[dict, float]]:
        if pool.precomputed is None:
            # Local pool: identical to today's embed_retrieval() call — no
            # logic duplicated, so local-pool methods can't drift from the
            # original script's numbers.
            return embed_retrieval(query, pool.paragraphs, k, self.model_name)
        return self._score_global(query, pool, k)

    def _score_global(self, query: str, pool: CandidatePool, k: int) -> list[tuple[dict, float]]:
        from sentence_transformers import SentenceTransformer

        if self.model_name not in _ST_CACHE:
            _ST_CACHE[self.model_name] = SentenceTransformer(self.model_name)
        st = _ST_CACHE[self.model_name]

        embeddings: np.ndarray = pool.precomputed
        qv = st.encode([query], normalize_embeddings=True)[0]
        sims = embeddings @ qv
        order = np.argsort(-sims)[:k].tolist()
        return [(pool.paragraphs[i], float(sims[i])) for i in order]


class ColbertRetriever(Retriever):
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        self.model_name = model_name

    def score(self, query: str, pool: CandidatePool, k: int) -> list[tuple[dict, float]]:
        if pool.precomputed is None:
            # Local pool: identical to today's colbert_retrieval() call.
            return colbert_retrieval(query, pool.paragraphs, k, self.model_name)
        return self._score_global(query, pool, k)

    def _get_checkpoint(self):
        if self.model_name not in _COLBERT_CKPT_CACHE:
            from colbert.infra import ColBERTConfig
            from colbert.modeling.checkpoint import Checkpoint

            config = ColBERTConfig()
            _COLBERT_CKPT_CACHE[self.model_name] = Checkpoint(self.model_name, colbert_config=config)
        return _COLBERT_CKPT_CACHE[self.model_name]

    def _score_global(
        self, query: str, pool: CandidatePool, k: int, *, chunk_size: int = 4096,
    ) -> list[tuple[dict, float]]:
        """MaxSim of the query's token vectors against the cached ragged
        per-passage token vectors, processed in chunks of passages so the
        whole corpus's token embeddings never need to be materialized on GPU
        at once."""
        import torch
        from colbert.modeling.colbert import colbert_score

        token_embeddings, offsets = pool.precomputed
        n_passages = len(offsets) - 1

        ckpt = self._get_checkpoint()
        Q = ckpt.queryFromText([query], bsize=1)

        scores = np.empty(n_passages, dtype=np.float32)
        for start in range(0, n_passages, chunk_size):
            end = min(start + chunk_size, n_passages)
            lengths = (offsets[start + 1 : end + 1] - offsets[start:end]).tolist()
            max_len = max(lengths) if lengths else 0
            batch = torch.zeros((end - start, max_len, token_embeddings.shape[1]), dtype=torch.float32)
            mask = torch.zeros((end - start, max_len), dtype=torch.bool)
            for i, (s, length) in enumerate(zip(offsets[start:end], lengths)):
                if length <= 0:
                    continue
                batch[i, :length] = torch.from_numpy(
                    token_embeddings[s : s + length].astype(np.float32)
                )
                mask[i, :length] = True
            chunk_scores = colbert_score(Q, batch, mask, config=ckpt.colbert_config)
            scores[start:end] = chunk_scores.detach().cpu().numpy().astype(np.float32)

        order = np.argsort(-scores)[:k].tolist()
        return [(pool.paragraphs[i], float(scores[i])) for i in order]
