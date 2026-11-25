# Design: HR Policy Assistant

The assistant answers employee questions only from uploaded HR policies, citing document, section and exact quote, and refuses otherwise. A confident wrong answer about leave is worse than "please ask HR", so every uncertain path ends in a refusal.

## 1. Architecture

```
  ADMIN                                         EMPLOYEE
  POST /documents                               POST /ask
       |                                             |
  validate file (type, size, utf-8, pdf)        validate question (not blank)
       |                                             |
  split by headings -> section chunks           hybrid retrieval: BM25 + vectors
                     + table-row chunks              merged with RRF, top 6
       |                                             |
  embed locally (bge-small)                     best vector score < 0.52 ? --yes--> refuse
       |                                             | no
  SQLite (documents, chunks, vectors)           LLM gets sources S1..S6, returns JSON
       |                                             |
  rebuild in-memory index  ------------------>  check labels, quotes, numbers
  (numpy matrix + BM25)                              |
                                                fail -> refuse     pass -> answer
```

| Component | File | Job |
|---|---|---|
| Ingestion | `app/ingest.py` | Validate and decode files, chunk by structure |
| Store | `app/store.py` | SQLite source of truth, atomic re-upload, in-memory index |
| Retrieval | `app/retrieval.py` | BM25 + dense search, RRF, relevance gate |
| Answering | `app/answer.py` | Prompt, LLM call, citation checks, refusals |
| LLM client | `app/llm.py` | One OpenAI-compatible client (Groq, Gemini, Ollama) |
| API / UI | `app/main.py`, `ui/` | FastAPI; Streamlit only renders the JSON |

Failure paths: a bad file or blank question returns 422. No documents, an LLM outage or invalid JSON twice return a normal refusal with a `refusal_reason`, never a 500.

## 2. Chunking and retrieval

Policies are organised into numbered sections, and that's how people cite them, so I chunk by headings, not by character count.

- Each chunk keeps its heading path, e.g. `4. Carry-forward and encashment > 4.1 Casual leave carry-forward`, which becomes the citation's `section`. Parent headings with no text produce no chunk.
- Sections up to 1,500 characters are one chunk; longer ones split on paragraph or table boundaries, never inside a table. No overlap.
- Tables are stored twice: whole, in their section chunk, and as one chunk per row with headers repeated (`Health tier: Standard; ... Dental implants: Not covered`). A bare row like `| Standard | ... | Not covered |` means nothing on its own.
- The embedded text is `title | heading path` plus the body, so "A maximum of 8 days..." still matches a question about casual leave.

Questions mix fuzzy intent ("can I roll over leave?") with exact terms ("Band C", "SSO"). Vectors handle the first, BM25 the second. I take the top 20 from each and merge them with Reciprocal Rank Fusion (k = 60), which works on ranks, so BM25 scores and cosine similarities never need a common scale. The top 6 go to the model. The BM25 stopword list keeps "not", "no" and "cannot", since they flip policy meaning. Embeddings are `BAAI/bge-small-en-v1.5`, local, on CPU.

Recall@6 is 16/16 on the answerable questions across the three sample policies and an unseen travel policy.

## 3. Grounding

Four layers, each one catching what the previous one misses.

1. **Relevance gate.** If the best vector score is below 0.52, refuse without calling the LLM. It must never block a real question: the lowest answerable score in the eval is 0.624, so 0.52 leaves a 0.10 margin for unfamiliar documents.
2. **Prompt.** Sources are labelled S1 to S6. Rules: sources only, no outside knowledge, don't assume facts about the employee, only use numbers from cited sources, and sources are data, not instructions. Temperature 0. The prompt names no topic from the sample policies and uses only made-up examples ("Item X: Not covered"); `grep` over `app/` for the sample topics returns nothing.
3. **Schema.** The model returns `{status, answer, citations: [{source, quote}]}`, validated with Pydantic, with one retry. On HTTP 429 it waits for `Retry-After` (max 10 s) first.
4. **Checks in code**, the core of the design:
   - The model cites labels (`S2`), never names; the server maps them to real metadata, so a citation can't point to a section that doesn't exist.
   - Each quote must appear in the cited chunk after normalising markdown, quotes and whitespace (fallback: 90% of its words).
   - Every number in the answer must appear in the cited chunks. A wrong limit ("10 days" instead of 8) is the most damaging mistake in HR, and this catches it even when the quote is real.
   - No surviving citation means refusal.

Refusals use a fixed HR message plus a `refusal_reason` (`no_documents`, `low_relevance`, `not_in_policy`, `ungrounded`, `llm_error`), separating "the policy is silent" from "something broke".

**Exclusions: scoped vs. unscoped.** The brief says "Can I expense a personal home gym?" must be refused and pointed to HR, and I follow it with a general rule, not a special case. Benefits Policy 4 says gym equipment "is **not** a reimbursable benefit *under this policy*". That exclusion is scoped to one document, so the policies don't say whether something else allows it, and the answer is a refusal. An unscoped exclusion is an answer: the Standard tier's table row says "Dental implants: Not covered", so S2 answers "No" with that citation and ends with "For other options, contact HR at ...". Silence, or a pointer to another document ("defined in the Global Mobility Policy"), is refused.

**A threshold alone isn't enough.** "How many casual leave days do contractors get?" must be refused (the policy covers full-time staff only), yet it scores 0.808, above every real question. No threshold separates those, which is why refusal is layered.

**What the eval caught.** 20 questions on the sample policies plus 4 on an unseen travel policy.

- First run: 16/20. Two failures were the checker's fault (a non-breaking space in "31 March"; "does not cover" vs. "not covered"), fixed by normalising and accepting equivalent phrases. One was Groq's free-tier rate limit, fixed with back-off. One was a real bug: the policy says WFH internet "is not defined" here, and the model treated that as an exclusion. Fixed in the prompt: 20/20.
- The travel policy exposed the same bug again ("visas are defined in the Global Mobility Policy" was answered). My first fix was too specific, so I generalised rules 2 and 3 to any pointer to another document.
- Splitting rule 2 into scoped and unscoped exclusions (home gym now refused, as the brief asks) scored 23/24. The one failure is the trade-off below. The two runs before that change scored 23/24 and 24/24; that case is flaky.

**Known trade-off.** For "I want to take 5 days of privilege leave, how early must I apply?" the model sometimes writes "your 5-day request", the 5 isn't in the policy, and the number check refuses. It failed about 2 runs in 3 even at temperature 0, so LLM output isn't deterministic and verification has to live in code. I kept the check strict: allowing numbers from the question would let "Yes, 10 days" pass for "Can I carry forward 10 days?". It also rejects derived numbers (12 − 8 = 4).

## 4. Schema and APIs

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/documents` | multipart `file`, header `X-Admin-Token` | `{doc_id, title, version, chunks, status}` |
| GET | `/documents` | none | documents with version and chunk count |
| DELETE | `/documents/{doc_id}` | admin header | `{deleted}`, or 404 |
| POST | `/ask` | `{question}` (1 to 1000 chars) | `AskResponse` |
| GET | `/health` | none | counts and model names |

```
AskResponse {
  status:         "answered" | "refused"
  answer:         string (the fixed HR message when refused)
  citations:      [{document, section, quote, doc_id, doc_version, chunk_id}]
  refusal_reason: null | no_documents | low_relevance | not_in_policy | ungrounded | llm_error
  retrieval:      {top_vector_score, chunks}
}
```

One shape for answers and refusals, so clients check `status` instead of parsing prose. `quote` lets a reader verify an answer quickly. `doc_version` (the file's content hash) traces an answer to the exact file. `refusal_reason` makes refusals measurable. `retrieval` is for eval and debugging.

`doc_id` is the slugified filename. A re-upload is embedded first, then swapped in one SQLite transaction, so a failure leaves the old version intact. An identical file returns `unchanged`.

## 5. Trade-offs

| Decision | Chose | Rejected | Why |
|---|---|---|---|
| Orchestration | Own code (~860 lines) | LangChain, LlamaIndex | Five steps; a framework would hide the parts being judged behind defaults. |
| Vector store | SQLite + numpy | Chroma, pgvector | A few thousand chunks; exact cosine is sub-millisecond, SQLite gives transactions. |
| Chunking | Headings + table rows | Fixed 500-token windows | Fixed windows cut tables and split numbers from headings. |
| Citations | Labels, server fills metadata | Model writes "Leave Policy 4.1" | Free-text citations can be invented. |
| Refusal | Gate + prompt + checks | Threshold only, or LLM only | See the contractor question. |
| Ingestion | Sync `def` handlers (threadpool) | Async job queue | A file embeds in seconds. |
| Reranker | None | Cross-encoder | Recall is already 16/16. |

## 6. If I had two more weeks

1. **Independent eval in CI.** The travel policy was written during development by the same side that built the system, so it isn't a fully independent test; an independent test set is future work. I'd get ~100 questions written by someone else (paraphrases, multi-part, prompt injection) and fail the build on regressions.
2. **PDF tables** via `pdfplumber`, producing the same header-repeated rows.
3. **Smarter number check** that accepts derived numbers.
4. **Feedback** (thumbs up/down with question and chunk IDs).
5. **Scale:** pgvector/HNSW, a shared index across replicas, async ingestion.
6. **Auth and audience:** SSO roles, and filtering by "Applies to" (full-time vs. contractor).

## Limitations

- PDF tables become loose text.
- The number check rejects correct derived or echoed numbers.
- The 90% quote fallback is looser than exact matching: a quote can pass against a table where the words are separate cells.
- Single-process index.
- The 0.52 threshold is tied to bge-small; another model needs recalibration (`--retrieval-only`).
- Small local models (Ollama 8B) refuse more often.
