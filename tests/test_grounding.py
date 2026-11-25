import re

import httpx
import pytest

from app import config
from app.text import normalize
from app.answer import (answer_question, hr_sentence, numbers_supported,
                        quote_supported, system_prompt)
from app.embeddings import get_embedder
from app.ingest import parse_document
from app.llm import LLMError, RateLimitError, chat_json
from app.models import RefusalReason, Status
from app.store import Store

Q = "What is the casual leave carry-forward limit?"
GOOD_QUOTE = "A maximum of 8 days of unused casual leave may be carried forward"


def fake_llm(payload):
    def _llm(system, user, _p=payload):
        if callable(_p):
            return _p(user)
        return _p
    return _llm


def cite_where(needle, answer, quote=None):
    def build(user):
        blocks = re.findall(r"\[(S\d+)\][^\n]*\n<<<\n(.*?)\n>>>", user, re.S)
        label = next(lbl for lbl, text in blocks if normalize(needle) in normalize(text))
        return {"status": "answered", "answer": answer,
                "citations": [{"source": label, "quote": quote or needle}]}
    return build


def test_grounded_answer_passes(store):
    r = answer_question(store, Q, fake_llm(cite_where(GOOD_QUOTE, "You can carry forward up to 8 days.")))
    assert r.status == Status.answered
    c = r.citations[0]
    assert (c.document, c.section) == ("Leave Policy",
                                       "4. Carry-forward and encashment > 4.1 Casual leave carry-forward")
    assert c.doc_version


def test_citation_to_unknown_source_is_refused(store):
    llm = fake_llm({"status": "answered", "answer": "8 days.",
                    "citations": [{"source": "S99", "quote": GOOD_QUOTE}]})
    r = answer_question(store, Q, llm)
    assert r.status == Status.refused and r.refusal_reason == RefusalReason.ungrounded


def test_fabricated_quote_is_refused(store):
    llm = fake_llm(cite_where(GOOD_QUOTE, "8 days.", quote="Employees may carry forward unlimited leave"))
    r = answer_question(store, Q, llm)
    assert r.refusal_reason == RefusalReason.ungrounded


def test_wrong_number_with_real_quote_is_refused(store):
    llm = fake_llm(cite_where(GOOD_QUOTE, "You can carry forward up to 10 days."))
    r = answer_question(store, Q, llm)
    assert r.refusal_reason == RefusalReason.ungrounded


def test_answer_without_citations_is_refused(store):
    r = answer_question(store, Q, fake_llm({"status": "answered", "answer": "8 days.", "citations": []}))
    assert r.refusal_reason == RefusalReason.ungrounded


def test_model_refusal_maps_to_not_in_policy(store):
    r = answer_question(store, "What is maternity leave?",
                        fake_llm({"status": "refused", "answer": "", "citations": []}))
    assert r.status == Status.refused
    assert r.refusal_reason == RefusalReason.not_in_policy
    assert config.HR_CONTACT in r.answer and r.citations == []


def test_llm_failure_fails_closed(store):
    calls = []

    def broken(system, user):
        calls.append(1)
        raise LLMError("boom")
    r = answer_question(store, Q, broken)
    assert r.refusal_reason == RefusalReason.llm_error
    assert len(calls) == 2


def test_malformed_schema_fails_closed(store):
    r = answer_question(store, Q, fake_llm({"status": "maybe", "answer": 8}))
    assert r.refusal_reason == RefusalReason.llm_error


def test_retrieval_gate_skips_llm(store, monkeypatch):
    monkeypatch.setattr(config, "MIN_VECTOR_SCORE", 0.99)

    def must_not_be_called(system, user):
        raise AssertionError("LLM called despite failed gate")
    r = answer_question(store, "zebra quantum spaceship", must_not_be_called)
    assert r.refusal_reason == RefusalReason.low_relevance


def test_no_documents(monkeypatch):
    empty = Store(":memory:", get_embedder())
    assert answer_question(empty, Q).refusal_reason == RefusalReason.no_documents


def test_reupload_replaces_old_version(store):
    new = b"# Leave Policy\n\n## 4.1 Casual leave carry-forward\n\nA maximum of 5 days may be carried forward."
    r = store.upsert(parse_document("leave-policy.md", new))
    assert r.status == "replaced"
    texts = [c.text for c in store.chunks if c.doc_id == "leave-policy"]
    assert len(texts) == 1 and "5 days" in texts[0]
    assert store.upsert(parse_document("leave-policy.md", new)).status == "unchanged"


def test_quote_matching_tolerates_formatting_only():
    src = "A maximum of **8 days** of unused casual leave may be carried forward."
    assert quote_supported("A maximum of 8 days of unused casual leave", src)
    assert quote_supported("a maximum of 8 days of unused casual leave may be carried forward", src)
    assert not quote_supported("A maximum of 18 days of unused casual leave", src)
    assert not quote_supported("8", src)


def test_number_check_handles_indian_grouping():
    assert numbers_supported("Plus covers implants up to ₹40,000.", ["Covered up to ₹40,000"])
    assert not numbers_supported("Plus covers implants up to ₹50,000.", ["Covered up to ₹40,000"])


def test_llm_json_parsing_tolerates_fences():
    from app.llm import parse_json_object
    assert parse_json_object('```json\n{"status": "refused"}\n```') == {"status": "refused"}
    import pytest
    with pytest.raises(LLMError):
        parse_json_object("sorry, I can't")


LTA_QUOTE = "Unclaimed LTA does **not** carry forward."


def test_exclusion_answer_with_hr_sentence_passes_number_check(store, monkeypatch):
    monkeypatch.setattr(config, "HR_CONTACT", "hr-desk@example.com or ext. 4455")
    answer = f"No, unclaimed LTA does not carry forward. {hr_sentence()}"
    r = answer_question(store, "Does unused LTA carry forward to next year?", fake_llm(cite_where(LTA_QUOTE, answer)))
    assert r.status == Status.answered
    assert r.answer.endswith("contact HR at hr-desk@example.com or ext. 4455.")
    assert r.citations[0].section == "3. Leave travel allowance (LTA)"


def test_hr_sentence_does_not_hide_a_wrong_number(store, monkeypatch):
    monkeypatch.setattr(config, "HR_CONTACT", "ext. 4455")
    answer = f"You can carry forward up to 10 days. {hr_sentence()}"
    r = answer_question(store, Q, fake_llm(cite_where(GOOD_QUOTE, answer)))
    assert r.refusal_reason == RefusalReason.ungrounded


def test_system_prompt_includes_hr_contact(monkeypatch):
    monkeypatch.setattr(config, "HR_CONTACT", "people@acme.test")
    assert "For other options, contact HR at people@acme.test." in system_prompt()
    assert "{hr_sentence}" not in system_prompt()


def _rate_limited_once(retry_after, calls):
    good = cite_where(GOOD_QUOTE, "You can carry forward up to 8 days.")

    def _llm(system, user):
        calls.append(1)
        if len(calls) == 1:
            raise RateLimitError(retry_after)
        return good(user)
    return _llm


@pytest.mark.parametrize("retry_after,expected_sleep", [(3, 3), (60, 10), (None, 5)])
def test_rate_limit_waits_then_retries(store, monkeypatch, retry_after, expected_sleep):
    slept, calls = [], []
    monkeypatch.setattr("app.answer.time.sleep", slept.append)
    r = answer_question(store, Q, _rate_limited_once(retry_after, calls))
    assert r.status == Status.answered
    assert len(calls) == 2
    assert slept == [expected_sleep]


def test_other_errors_retry_without_sleeping(store, monkeypatch):
    slept = []
    monkeypatch.setattr("app.answer.time.sleep", slept.append)

    def broken(system, user):
        raise LLMError("boom")
    answer_question(store, Q, broken)
    assert slept == []


def test_chat_json_raises_rate_limit_error_on_429(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr("app.llm.httpx.post",
                        lambda *a, **k: httpx.Response(429, headers={"retry-after": "7"},
                                                       request=httpx.Request("POST", "http://x")))
    with pytest.raises(RateLimitError) as e:
        chat_json("sys", "user")
    assert e.value.retry_after == 7.0


def test_system_prompt_separates_scoped_and_unscoped_exclusions():
    p = system_prompt()
    assert "Unscoped" in p and "Scoped" in p and "under this policy" in p
