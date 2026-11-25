import os
from pathlib import Path

os.environ["EMBED_MODEL"] = "hashing"
os.environ["DATA_DIR"] = "/tmp/hr-rag-test"

import pytest

from app import config
from app.embeddings import get_embedder
from app.ingest import parse_document
from app.store import Store

SAMPLES = Path(__file__).parent.parent / "sample_policies"


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(config, "MIN_VECTOR_SCORE", 0.0)
    s = Store(":memory:", get_embedder())
    for f in sorted(SAMPLES.glob("*.md")):
        s.upsert(parse_document(f.name, f.read_bytes()))
    return s
