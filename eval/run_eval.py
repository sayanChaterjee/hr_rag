from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from app import config
from app.answer import answer_question
from app.embeddings import get_embedder
from app.ingest import parse_document
from app.retrieval import retrieve
from app.text import normalize
from app.store import Store

ROOT = Path(__file__).parent.parent


def load_store(*paths: Path) -> Store:
    store = Store(":memory:", get_embedder())
    for path in paths:
        files = sorted(path.iterdir()) if path.is_dir() else [path]
        for f in files:
            if f.suffix.lower() in {".md", ".txt", ".pdf"}:
                store.upsert(parse_document(f.name, f.read_bytes()))
    return store


def expected_hit(case, doc_id, section) -> bool:
    return doc_id == case["doc"] and case["section"].lower() in section.lower()


def retrieval_only(store, cases):
    print(f"{'recall@k':8}  {'top_vec':7}  question")
    ans_scores, ref_scores, hits = [], [], 0
    for c in cases:
        r = retrieve(store, c["q"])
        score = r.top_vector_score
        if c["expect"] == "answer":
            ok = any(expected_hit(c, h.chunk.doc_id, h.chunk.section) for h in r.hits)
            hits += ok
            ans_scores.append(score)
            print(f"{'HIT' if ok else 'MISS':8}  {score:7.3f}  {c['q']}")
        else:
            ref_scores.append(score)
            print(f"{'(refuse)':8}  {score:7.3f}  {c['q']}")

    print(f"\nrecall@{config.TOP_K}: {hits}/{len(ans_scores)}")
    lo_ans, hi_ref = min(ans_scores), max(ref_scores)
    print(f"lowest answerable top_vec = {lo_ans:.3f}; highest refuse top_vec = {hi_ref:.3f}")
    print(f"suggested MIN_VECTOR_SCORE ≈ {lo_ans - 0.05:.2f} (current {config.MIN_VECTOR_SCORE})")


def content_ok(case, answer) -> bool:
    text = normalize(answer)
    if not all(normalize(s) in text for s in case.get("answer_contains", [])):
        return False
    phrases = case.get("answer_any", [])
    return not phrases or any(normalize(p) in text for p in phrases)


def full(store, cases, delay):
    rows, passed = [], 0
    for c in cases:
        resp = answer_question(store, c["q"])
        checks = {"status": resp.status.value == ("answered" if c["expect"] == "answer" else "refused")}
        if c["expect"] == "answer":
            checks["citation"] = any(expected_hit(c, x.doc_id, x.section) for x in resp.citations)
            checks["content"] = content_ok(c, resp.answer)
        ok = all(checks.values())
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {c['q']}")
        if not ok:
            print(f"      checks={checks} reason={resp.refusal_reason} answer={resp.answer!r}")
        rows.append({"q": c["q"], "ok": ok, "checks": checks, "response": resp.model_dump(mode="json")})
        time.sleep(delay)
    print(f"\n{passed}/{len(cases)} passed")
    out = ROOT / "eval" / "results.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"details -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--policies", nargs="+", default=[str(ROOT / "sample_policies")],
                    help="policy folders or files")
    ap.add_argument("--questions", nargs="+", default=[str(ROOT / "eval" / "questions.yaml")])
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between LLM calls")
    args = ap.parse_args()

    cases = [c for q in args.questions for c in yaml.safe_load(Path(q).read_text())]
    store = load_store(*(Path(p) for p in args.policies))
    retrieval_only(store, cases) if args.retrieval_only else full(store, cases, args.delay)


if __name__ == "__main__":
    main()
