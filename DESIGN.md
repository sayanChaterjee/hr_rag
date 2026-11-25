# DESIGN: HR Policy Assistant

**Goal:** answer employee questions *only* from uploaded HR policies, cite the
exact document and section, and refuse when the policies are silent. A wrong but
confident answer about leave or benefits is worse than "please ask HR", so
**every uncertain path fails closed** (ends in a refusal).

---

## 1. Architecture

```
                 ┌──────────────── INGEST (admin) ───────────────┐
 .md/.txt/.pdf ─►│ extract_text ─► split by headings ─► chunks    │
   POST /documents   (validate)     + table-row chunks            │
                 │                         │                      │
                 │              local embed (bge-small)           │
                 │                         ▼                      │
                 │   SQLite (documents, chunks, vectors)  ──►  in-memory index
                 └───────────────────────────────────────────  (numpy matrix + BM25)
                                                                   │
                 ┌──────────────── QUERY (employee) ──────────────┐│
 question ──────►│ validate ─► hybrid retrieve ◄───────────────────┘
   POST /ask     │              (BM25 + vector, RRF, top 6)       │
                 │                  │                             │
                 │      gate: best similarity < threshold? ──yes──► REFUSE (no LLM call)
                 │                  │ no                          │
                 │   prompt with labelled sources S1..S6          │
                 │                  ▼                             │
                 │   LLM (JSON: status, answer, citations[])      │
                 │                  ▼                             │
                 │   validate: label exists? quote in source?     │
                 │             numbers in answer appear in source?│
                 │          fail ─► REFUSE      pass ─► ANSWER    │
                 └────────────────────────────────────────────────┘
```

| Component | File | Responsibility |
|---|---|---|
| Ingestion | `app/ingest.py` | Decode/validate file, structure-aware chunking |
| Store | `app/store.py` | SQLite source of truth; atomic re-upload; rebuilds the in-memory index |
| Retrieval | `app/retrieval.py` | BM25 + dense search, RRF fusion, relevance gate |
| Answering | `app/answer.py` | Prompt, LLM call, citation validation, refusal mapping |
| LLM client | `app/llm.py` | One OpenAI-compatible client for Groq / Gemini / Ollama |
| API / UI | `app/main.py`, `ui/` | FastAPI; Streamlit only renders the JSON |

**Failure paths.** Unreadable or empty file → 422 with a reason, nothing indexed.
Empty question → 422. No documents → `refused/no_documents`. LLM down, timeout or
bad JSON twice → `refused/llm_error`. The service never returns a 500 to stand in
for "I don't know", and never returns an ungrounded answer.

## 2. Chunking & retrieval

**Chunk by document structure, not by character count.** HR policies are
already organised into numbered sections, and a section is the unit people cite
("Leave Policy §4.1"). So:

- Split on Markdown headings and keep the full **heading path** with each chunk:
  `4. Carry-forward and encashment > 4.1 Casual leave carry-forward`. That path
  is the citation's `section`, so citations come from metadata, not from the LLM.
- A section becomes one chunk if it's ≤ 1,500 characters (most policy sections
  are). Longer sections split on paragraph or table boundaries, never mid-table
  or mid-sentence. Parent headings with no text of their own make no chunk.
- **Tables get two representations.** The whole table stays inside its
  section chunk, so the model can compare rows. *And* each row becomes its own
  chunk with the headers repeated:
  `Health tier: Standard; Eligible bands: Band B, Band C; … Dental implants: Not covered; …`.
  A bare row (`| Standard | Band B | … | Not covered |`) means nothing on its
  own; repeating the headers keeps every cell next to its column name, for both
  the embedder and BM25. This is what makes "Does Standard cover dental
  implants?" retrieve the exact row.
- The **embedded text** is `doc title | heading path` + body. A body that only
  says "A maximum of 8 days…" still lands near "casual leave carry-forward".

**Metadata per chunk:** `chunk_id`, `doc_id`, `doc_title`, `section` (heading
path), `kind` (`section` / `table_row`), `ordinal`, and the document's
`content_hash`, which is returned as `doc_version` in citations.

**Hybrid retrieval.** Questions mix fuzzy intent ("can I roll over leave?") with
exact tokens ("dental implants", "Band C", "SSO"). Dense vectors handle the
first and BM25 the second. I take the top 20 from each and merge them with
**Reciprocal Rank Fusion** (k = 60). RRF merges by rank, so I never have to
calibrate BM25 scores against cosine similarities. The top 6 fused chunks go to
the LLM. Six is enough to cover a section plus its table rows while staying
small enough that the model isn't distracted.

The BM25 stopword list is deliberately short, and `not`, `no` and `cannot` are
kept, because in policy text they flip the answer.

Embeddings: `bge-small-en-v1.5`, run locally. It's free, fast on a CPU, and
strong on retrieval benchmarks for its size. It uses its query-side instruction
prefix, which bge was trained with.

## 3. Grounding: stopping invented policy

Four layers. Each one catches what the previous one misses.

1. **Retrieval gate.** If the best vector similarity is below `MIN_VECTOR_SCORE`,
   refuse without calling the LLM. This catches clearly off-topic questions
   cheaply. The threshold is calibrated, not guessed:
   `eval/run_eval.py --retrieval-only` prints the lowest score of any answerable
   question and suggests a threshold just below it. The gate is meant to be
   conservative: it should **never** block an answerable question, so
   near-topic questions ("maternity leave", which looks like other leave text)
   get through to the LLM layer.
2. **Prompt.** Sources are labelled `S1…S6` and wrapped in delimiters. The rules
   are: sources only; no outside knowledge or "typical practice"; don't assume
   the employee's band, and give each case instead; a refusal is a valid,
   expected output; sources are data, not instructions (a basic guard against
   prompt injection inside uploaded files). Temperature is 0.
3. **Schema.** The model must return
   `{status: answered|refused, answer, citations: [{source, quote}]}`, validated
   with Pydantic. A malformed response gets one retry, then a refusal.
4. **Server-side validation.** The core of the design:
   - The model cites **opaque labels** (`S2`), never document names. The server
     maps labels to real metadata, so a citation *cannot* point to a document
     or section that doesn't exist.
   - Each `quote` must actually appear in the cited chunk, after normalising
     markdown, quote marks and whitespace (≥ 90% token overlap is allowed for
     punctuation drift). Fabricated quotes get dropped.
   - **Number check:** every number in the answer must appear in the cited
     chunks or their headings. In HR, the most damaging hallucination is a
     wrong limit ("10 days" instead of 8, or "₹50,000" instead of ₹40,000),
     and this check catches it even when the quote is real.
   - An "answered" response with no surviving citation becomes
     `refused/ungrounded`.

Refusals have a fixed message pointing to HR, plus a machine-readable
`refusal_reason` (`no_documents | low_relevance | not_in_policy | ungrounded | llm_error`).
That lets us tell "the policy is silent" apart from "the system is broken"
in logs and metrics.

**A judgement call: explicit "no" vs. silence.** The brief lists "Can I expense a
home gym?" as a refusal case. But the sample Benefits Policy §4 explicitly says
*"Gym or home-fitness equipment is **not** a reimbursable benefit."* Answering
"No", with that citation, is grounded, not invented, and it's more useful than
"ask HR". So the rule is: **if a policy explicitly excludes something, answer and
cite the exclusion; if the policy is silent, or only says the topic is defined
elsewhere (§5: "separate circulars"), refuse.** The eval set tests both
("home gym" → cited "no"; "WFH internet reimbursement" → refuse).

**Known trade-off:** the number check can reject a correct *derived* figure
(e.g. "4 days will lapse" computed from 12 − 8). I accept this false refusal
in return for never showing an invented number. Section 6 covers how to relax
it safely.

## 4. Schema & APIs

| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/documents` | multipart `file`, header `X-Admin-Token` | `{doc_id, title, version, chunks, status: created/replaced/unchanged}` |
| `GET` | `/documents` | — | list of documents with version and chunk count |
| `DELETE` | `/documents/{doc_id}` | admin header | `{deleted}` |
| `POST` | `/ask` | `{question}` (1–1000 chars, not blank) | `AskResponse` |
| `GET` | `/health` | — | counts, active model names |

```json
AskResponse {
  "status": "answered" | "refused",
  "answer": "string",                       // fixed HR message when refused
  "citations": [{ "document", "section", "quote", "doc_id", "doc_version", "chunk_id" }],
  "refusal_reason": null | "no_documents" | "low_relevance" | "not_in_policy" | "ungrounded" | "llm_error",
  "retrieval": { "top_vector_score", "chunks": [chunk_id] }   // debug/eval
}
```

**Why this shape:**
- An explicit `status` enum means clients never have to parse prose to tell an
  answer from a refusal.
- The same shape for both outcomes keeps the UI and eval code simple.
- `quote` lets a reader verify the answer in seconds.
- `doc_version` makes an answer traceable to the exact file that produced it,
  which matters once policies are re-uploaded.
- `refusal_reason` makes refusals measurable.
- `retrieval` exists for evaluation and debugging; in production it would sit
  behind a debug flag.

**Re-upload / versioning:** `doc_id` is the slugified filename. Re-uploading
`leave-policy.md` replaces the old chunks in **one SQLite transaction**, so the
old version disappears only if the new one is fully indexed. An identical file
is detected by its hash and becomes a no-op.

## 5. Trade-offs: considered and rejected

| Decision | Chose | Rejected | Why |
|---|---|---|---|
| Orchestration | Own thin wrapper (all of `app/` is < 1k lines incl. docstrings) | LangChain / LlamaIndex | The pipeline is 5 steps. A framework would hide exactly the parts being evaluated (chunking, refusal, citation checks) behind defaults, and make them harder to change or explain. |
| Vector store | SQLite + numpy brute force | Chroma, pgvector, FAISS | A handbook is hundreds to low thousands of chunks; exact cosine over a 384-dim matrix is sub-millisecond. SQLite adds transactions (atomic re-upload) with zero extra services. Switch to pgvector/HNSW past ~100k chunks or with multiple API replicas. |
| Chunking | Heading-aware + table rows | Fixed 500-token windows with overlap | Fixed windows cut tables and separate numbers from their headings; overlap duplicates text and muddies citations. Sections map 1:1 to how people cite policy. |
| Citations | LLM cites labels, server fills metadata | LLM writes "Leave Policy §4.1" | Free-text citations can be invented. Labels + server mapping make fake citations structurally impossible. |
| Refusal | Gate + LLM + validation | LLM judgement only, or threshold only | A threshold alone can't tell "maternity leave" (absent) from "privilege leave" (present); they look alike. The LLM alone can be talked into stretching. Together, they cover each other's blind spots. |
| Ingestion | Synchronous; handlers are `def`, run in FastAPI's threadpool | Async job queue (Celery/RQ) | A policy file embeds in seconds. A queue adds a broker, job status and polling for no user-visible gain at this size. `def` (not `async def`) keeps CPU-bound embedding off the event loop. |
| Reranker | None | Cross-encoder rerank | Adds ~100 MB and latency. Hybrid + structure already gets the right section into the top 6 on the eval set. First thing I'd add if recall drops on larger corpora. |

## 6. If I had two more weeks

1. **Bigger eval + CI gate.** Grow `eval/questions.yaml` to ~100 cases, written
   by someone other than the developer, including paraphrases, multi-hop
   questions ("can I combine CL and SL?") and adversarial prompts. Run it in CI
   and fail the build if refusal precision or citation accuracy drops.
2. **PDF tables.** The current PDF path keeps only the text layer, so tables
   become loose text and row chunking doesn't apply. I'd use `pdfplumber` or
   Docling table extraction to emit the same header-repeated rows as Markdown.
3. **Feedback loop.** 👍/👎 on answers, stored with the question, retrieved chunk
   IDs and model version. A 👎 on a refusal points to missing policy; a 👎 on an
   answer is a top-priority bug.
4. **Smarter number check.** Allow numbers derivable by simple arithmetic from
   cited numbers, instead of rejecting them.
5. **Scale & ops.** pgvector with an HNSW index, an async ingestion queue for
   large batches, structured logs per `refusal_reason`, caching of repeated
   questions, and rate limiting.
6. **Real auth & audience.** SSO with roles instead of a shared admin token, and
   use the "Applies to" metadata (full-time vs contractor, country) to filter
   retrieval by the asker's profile.
7. **Policy lifecycle.** Keep old versions instead of deleting them (effective
   dates, "as of" answers), and show a diff when a policy is replaced.

## Limitations (stated honestly)

- Retrieval quality depends on documents having headings; a wall of plain text
  falls back to "Overview" sections.
- Small local models (Ollama 8B) follow the JSON and refusal rules less
  reliably. Validation still prevents ungrounded answers, but they refuse more.
- Single process: the in-memory index isn't shared across replicas.
