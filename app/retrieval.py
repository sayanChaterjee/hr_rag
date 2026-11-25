from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .ingest import Chunk
from .store import Store
from .text import tokenize


@dataclass
class Hit:
    chunk: Chunk
    rrf: float
    vector_score: float
    bm25_score: float


@dataclass
class RetrievalResult:
    hits: list[Hit]
    top_vector_score: float
    passed_gate: bool


def retrieve(store: Store, question: str, top_k: int | None = None) -> RetrievalResult:
    top_k = top_k or config.TOP_K
    chunks, matrix, bm25 = store.snapshot()
    if not chunks:
        return RetrievalResult([], 0.0, False)

    n_cand = min(config.CANDIDATES_PER_RETRIEVER, len(chunks))

    q_vec = store.embedder.encode([question], is_query=True)[0]
    vec_scores = matrix @ q_vec
    vec_rank = np.argsort(-vec_scores)[:n_cand]

    q_tokens = tokenize(question)
    bm25_scores = bm25.get_scores(q_tokens) if q_tokens else np.zeros(len(chunks))
    bm25_rank = [i for i in np.argsort(-bm25_scores)[:n_cand] if bm25_scores[i] > 0]

    fused: dict[int, float] = {}
    for ranking in (vec_rank, bm25_rank):
        for rank, idx in enumerate(ranking):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (config.RRF_K + rank + 1)

    order = sorted(fused, key=fused.get, reverse=True)[:top_k]
    hits = [Hit(chunks[i], fused[i], float(vec_scores[i]), float(bm25_scores[i])) for i in order]

    top_vec = float(vec_scores.max())
    return RetrievalResult(hits, top_vec, top_vec >= config.MIN_VECTOR_SCORE)
