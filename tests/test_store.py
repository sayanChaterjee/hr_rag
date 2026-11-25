import numpy as np

from app.ingest import parse_document


def assert_consistent(store, snap):
    chunks, matrix, bm25 = snap
    assert matrix.shape[0] == len(chunks) == bm25.corpus_size
    expected = store.embedder.encode([c.embed_text for c in chunks])
    assert np.allclose(matrix, expected)


def test_snapshot_is_consistent_after_upsert(store):
    before = store.snapshot()
    new = b"# Leave Policy\n\n## 1. Only section\n\nA maximum of 5 days may be carried forward."
    assert store.upsert(parse_document("leave-policy.md", new)).status == "replaced"

    after = store.snapshot()
    assert_consistent(store, after)
    assert store.chunks is after[0]
    assert [c.text for c in after[0] if c.doc_id == "leave-policy"] == [
        "A maximum of 5 days may be carried forward."]

    assert_consistent(store, before)
    assert len(before[0]) != len(after[0])
