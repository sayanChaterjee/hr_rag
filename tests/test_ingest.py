import pytest

from app.ingest import IngestError, parse_document

from .conftest import SAMPLES


def test_heading_path_becomes_section():
    doc = parse_document("leave-policy.md", (SAMPLES / "leave-policy.md").read_bytes())
    assert doc.title == "Leave Policy"
    sections = {c.section for c in doc.chunks}
    assert "4. Carry-forward and encashment > 4.1 Casual leave carry-forward" in sections
    assert "4. Carry-forward and encashment" not in sections


def test_table_rows_keep_their_headers():
    doc = parse_document("benefits-policy.md", (SAMPLES / "benefits-policy.md").read_bytes())
    rows = [c for c in doc.chunks if c.kind == "table_row"]
    standard = next(r for r in rows if "Health tier: Standard" in r.text)
    assert "Dental implants: Not covered" in standard.text
    assert standard.section == "2. Health coverage tiers"
    section = next(c for c in doc.chunks if c.kind == "section" and c.section == standard.section)
    assert "| Standard |" in section.text


def test_long_section_splits_on_block_boundaries(monkeypatch):
    from app import config
    monkeypatch.setattr(config, "MAX_CHUNK_CHARS", 200)
    paras = "\n\n".join(f"Paragraph {i} " + "x" * 120 for i in range(5))
    doc = parse_document("long.md", f"# Long\n\n## 1. Big\n\n{paras}".encode())
    assert len(doc.chunks) == 5
    assert all(c.text.startswith("Paragraph") for c in doc.chunks)


def test_txt_without_headings_still_ingests():
    doc = parse_document("note.txt", b"Office closes at 7pm on Fridays.")
    assert doc.chunks[0].section == "Overview"
    assert doc.title == "note"


@pytest.mark.parametrize("name,raw,msg", [
    ("empty.md", b"", "empty"),
    ("blank.md", b"   \n\n ", "no readable text"),
    ("bin.md", b"\x00\x01\x02binary", "binary"),
    ("latin.txt", "caf\xe9".encode("latin-1"), "UTF-8"),
    ("policy.docx", b"PK\x03\x04", "Unsupported"),
    ("broken.pdf", b"%PDF-1.4 garbage", "PDF"),
])
def test_unreadable_files_raise_clean_errors(name, raw, msg):
    with pytest.raises(IngestError, match=msg):
        parse_document(name, raw)
