from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .embeddings import Embedder
from .ingest import Chunk, ParsedDocument
from .text import tokenize

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY, title TEXT NOT NULL, filename TEXT NOT NULL,
    content_hash TEXT NOT NULL, uploaded_at TEXT NOT NULL, n_chunks INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    doc_title TEXT NOT NULL, section TEXT NOT NULL, kind TEXT NOT NULL,
    text TEXT NOT NULL, ordinal INTEGER NOT NULL, vector BLOB NOT NULL
);
"""


@dataclass
class UpsertResult:
    doc_id: str
    title: str
    version: str
    chunks: int
    status: str


class Store:
    def __init__(self, db_path, embedder: Embedder):
        self.embedder = embedder
        self._lock = threading.RLock()
        if str(db_path) != ":memory:":
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self._rebuild_index()

    def upsert(self, doc: ParsedDocument) -> UpsertResult:
        vectors = self.embedder.encode([c.embed_text for c in doc.chunks])
        with self._lock:
            row = self.db.execute(
                "SELECT content_hash FROM documents WHERE doc_id = ?", (doc.doc_id,)
            ).fetchone()
            if row and row[0] == doc.content_hash:
                return UpsertResult(doc.doc_id, doc.title, doc.content_hash, len(doc.chunks), "unchanged")
            with self.db:
                self.db.execute("DELETE FROM documents WHERE doc_id = ?", (doc.doc_id,))
                self.db.execute(
                    "INSERT INTO documents VALUES (?,?,?,?,?,?)",
                    (doc.doc_id, doc.title, doc.filename, doc.content_hash,
                     datetime.now(timezone.utc).isoformat(timespec="seconds"), len(doc.chunks)),
                )
                self.db.executemany(
                    "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)",
                    [(c.chunk_id, c.doc_id, c.doc_title, c.section, c.kind, c.text, c.ordinal,
                      v.astype(np.float32).tobytes()) for c, v in zip(doc.chunks, vectors)],
                )
            self._rebuild_index()
        return UpsertResult(doc.doc_id, doc.title, doc.content_hash, len(doc.chunks),
                            "replaced" if row else "created")

    def delete(self, doc_id: str) -> bool:
        with self._lock, self.db:
            n = self.db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,)).rowcount
            self._rebuild_index()
        return n > 0

    def list_documents(self) -> list[dict]:
        cur = self.db.execute(
            "SELECT doc_id, title, filename, content_hash, uploaded_at, n_chunks FROM documents ORDER BY title")
        keys = ["doc_id", "title", "filename", "version", "uploaded_at", "chunks"]
        return [dict(zip(keys, r)) for r in cur.fetchall()]

    def version_of(self, doc_id: str) -> str:
        row = self.db.execute("SELECT content_hash FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return row[0] if row else ""

    def _rebuild_index(self) -> None:
        rows = self.db.execute(
            "SELECT chunk_id, doc_id, doc_title, section, kind, text, ordinal, vector "
            "FROM chunks ORDER BY doc_id, ordinal").fetchall()
        chunks = [Chunk(*r[:7]) for r in rows]
        matrix = (np.vstack([np.frombuffer(r[7], dtype=np.float32) for r in rows])
                  if rows else np.zeros((0, 1), dtype=np.float32))
        bm25 = BM25Okapi([tokenize(c.embed_text) for c in chunks]) if chunks else None
        self.chunks, self.matrix, self.bm25 = chunks, matrix, bm25

    def snapshot(self):
        return self.chunks, self.matrix, self.bm25
