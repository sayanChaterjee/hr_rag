from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf"}


class IngestError(ValueError):
    pass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    kind: str
    text: str
    ordinal: int

    @property
    def embed_text(self) -> str:
        return f"{self.doc_title} | {self.section}\n{self.text}"


@dataclass
class ParsedDocument:
    doc_id: str
    title: str
    filename: str
    content_hash: str
    chunks: list[Chunk] = field(default_factory=list)


@dataclass
class _Block:
    kind: str
    lines: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


def extract_text(filename: str, raw: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestError(f"Unsupported file type '{ext}'. Upload .md, .txt or .pdf.")
    if not raw:
        raise IngestError("File is empty.")
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise IngestError(f"File is larger than {config.MAX_UPLOAD_BYTES // 1024} KB.")

    if ext == ".pdf":
        text = _pdf_to_text(raw)
    else:
        if b"\x00" in raw[:4096]:
            raise IngestError("File looks binary, not text.")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as e:
            raise IngestError("File is not valid UTF-8 text.") from e

    if not text.strip():
        raise IngestError("File contains no readable text.")
    return text


_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z][^.]{0,80})$")


def _pdf_to_text(raw: bytes) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = [p.extract_text() or "" for p in reader.pages]
    except Exception as e:
        raise IngestError(f"Could not read PDF: {e}") from e
    if not any(p.strip() for p in pages):
        raise IngestError("PDF has no text layer (scanned?). OCR is not supported.")

    out = []
    for line in "\n".join(pages).splitlines():
        m = _NUMBERED_HEADING.match(line.strip())
        if m:
            level = 2 + m.group(1).count(".")
            out.append("#" * min(level, 6) + " " + line.strip())
        else:
            out.append(line)
    return "\n".join(out)


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_TITLE_FIELD = re.compile(r"^\*\*Document title:\*\*\s*(.+)$", re.I)


def _split_sections(text: str) -> tuple[str | None, list[tuple[str, list[_Block]]]]:
    title: str | None = None
    stack: list[tuple[int, str]] = []
    sections: list[tuple[str, list[_Block]]] = []
    blocks: list[_Block] = []
    current: _Block | None = None
    in_fence = False

    def path() -> str:
        return " > ".join(h for _, h in stack) or "Overview"

    def close_block():
        nonlocal current
        if current and current.text:
            blocks.append(current)
        current = None

    def close_section():
        nonlocal blocks
        close_block()
        if blocks:
            sections.append((path(), blocks))
        blocks = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(stripped)
        if m:
            level, heading = len(m.group(1)), m.group(2).strip()
            if level == 1 and title is None:
                close_section()
                title = heading
                continue
            close_section()
            level = max(level, 2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            continue

        kind = "table" if (stripped.startswith("|") and not in_fence) else "para"
        if not stripped and not in_fence:
            close_block()
            continue
        if current is None or current.kind != kind:
            close_block()
            current = _Block(kind, [])
        current.lines.append(line.rstrip())

    close_section()
    return title, sections


def _table_rows(table: _Block) -> list[str]:
    def cells(row: str) -> list[str]:
        return [c.strip().replace("**", "") for c in row.strip().strip("|").split("|")]

    rows = [r for r in table.lines if r.strip()]
    if len(rows) < 3:
        return []
    header = cells(rows[0])
    out = []
    for row in rows[1:]:
        values = cells(row)
        if all(re.fullmatch(r":?-{2,}:?", v) for v in values if v):
            continue
        pairs = [f"{h}: {v}" for h, v in zip(header, values) if h or v]
        out.append("; ".join(pairs))
    return out


def _pack(blocks: list[_Block]) -> list[str]:
    chunks, buf = [], ""
    for b in blocks:
        piece = b.text
        if buf and len(buf) + len(piece) + 2 > config.MAX_CHUNK_CHARS:
            chunks.append(buf)
            buf = ""
        buf = f"{buf}\n\n{piece}" if buf else piece
    if buf:
        chunks.append(buf)
    return chunks


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", Path(name).stem.lower()).strip("-") or "document"


def parse_document(filename: str, raw: bytes) -> ParsedDocument:
    text = extract_text(filename, raw)
    h1, sections = _split_sections(text)

    title = h1
    for line in text.splitlines()[:15]:
        m = _TITLE_FIELD.match(line.strip())
        if m:
            title = m.group(1).strip()
            break
    title = title or Path(filename).stem

    doc = ParsedDocument(
        doc_id=slugify(filename),
        title=title,
        filename=filename,
        content_hash=hashlib.sha256(raw).hexdigest()[:12],
    )

    n = 0
    for section, blocks in sections:
        for body in _pack(blocks):
            doc.chunks.append(Chunk(f"{doc.doc_id}#{n}", doc.doc_id, title, section, "section", body, n))
            n += 1
        for table in (b for b in blocks if b.kind == "table"):
            for row in _table_rows(table):
                doc.chunks.append(Chunk(f"{doc.doc_id}#{n}", doc.doc_id, title, section, "table_row", row, n))
                n += 1

    if not doc.chunks:
        raise IngestError("No content found after parsing.")
    return doc
