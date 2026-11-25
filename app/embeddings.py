from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Protocol

import numpy as np

from . import config


class Embedder(Protocol):
    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        # bge was trained with this instruction on the query side only; documents are embedded as-is.
        if is_query and config.EMBED_QUERY_PREFIX:
            texts = [config.EMBED_QUERY_PREFIX + t for t in texts]
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


class HashingEmbedder:

    dim = 512

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    if config.EMBED_MODEL == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(config.EMBED_MODEL)
