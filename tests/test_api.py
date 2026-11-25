import pytest
from fastapi.testclient import TestClient

from app import config, main
from app.embeddings import get_embedder
from app.store import Store

ADMIN = {"X-Admin-Token": config.ADMIN_TOKEN}


@pytest.fixture
def client(monkeypatch):
    s = Store(":memory:", get_embedder())
    monkeypatch.setattr(main, "get_store", lambda: s)
    return TestClient(main.app)


def test_empty_question_is_422_not_500(client):
    for q in ["", "   "]:
        r = client.post("/ask", json={"question": q})
        assert r.status_code == 422


def test_upload_requires_admin(client):
    r = client.post("/documents", files={"file": ("a.md", b"# A\n\nhello")})
    assert r.status_code == 403


def test_unreadable_upload_is_422(client):
    r = client.post("/documents", headers=ADMIN, files={"file": ("a.md", b"\x00\x00\x00")})
    assert r.status_code == 422 and "binary" in r.json()["detail"]


def test_upload_list_delete(client):
    r = client.post("/documents", headers=ADMIN,
                    files={"file": ("travel.md", b"# Travel Policy\n\n## 1. Flights\n\nEconomy only.")})
    assert r.status_code == 200 and r.json()["status"] == "created"
    assert [d["doc_id"] for d in client.get("/documents").json()] == ["travel"]
    assert client.delete("/documents/travel", headers=ADMIN).status_code == 200
    assert client.delete("/documents/travel", headers=ADMIN).status_code == 404


def test_ask_with_no_documents_returns_structured_refusal(client):
    r = client.post("/ask", json={"question": "How many sick days?"}).json()
    assert r["status"] == "refused" and r["refusal_reason"] == "no_documents"
    assert r["citations"] == []
