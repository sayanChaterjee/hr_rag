# Design: HR Policy Assistant
 
The assistant answers employee questions **only** from uploaded HR policies, cites the document, section and exact quote, and refuses when the policies don't say. My guiding rule: a confident wrong answer about leave or benefits costs far more than "please contact HR", so **every uncertain path ends in a refusal**.
 
## 1. Architecture
 
```
  ADMIN  POST /documents                      EMPLOYEE  POST /ask
         |                                              |
  validate file (type, size, UTF-8, PDF text)    validate question (not blank)
         |                                              |
  split by headings -> section chunks            hybrid search: BM25 + vectors
                     + one chunk per table row   merged with RRF -> top 6
         |                                              |
  embed locally (bge-small)                      best vector score < 0.52 ? --yes--> refuse
         |                                              | no
  SQLite: documents, chunks, vectors             LLM gets sources S1..S6 -> JSON
         |                                              |
  rebuild in-memory index ---------------------> check labels, quotes, numbers
  (numpy matrix + BM25, swapped atomically)             |
                                                 fail -> refuse    pass -> answer
```
 
| Component | File | Job |
|---|---|---|
| Ingestion | `app/ingest.py` | Validate and decode files; chunk by document structure |
| Store | `app/store.py` | SQLite source of truth; atomic re-upload; in-memory index |
| Retrieval | `app/retrieval.py` | BM25 + dense search, Reciprocal Rank Fusion, relevance gate |
| Answering | `app/answer.py` | Prompt, LLM call, citation checks, refusals |
| LLM client | `app/llm.py` | One OpenAI-compatible client (Groq, Gemini, Ollama) |
| API / UI | `app/main.py`, `ui/` | FastAPI; Streamlit only renders the JSON |
 
**Stack:** Python, FastAPI, Pydantic, SQLite, numpy, `rank-bm25`, local `BAAI/bge-small-en-v1.5` embeddings, and Groq (`openai/gpt-oss-120b`) for answers. Embeddings are free and local; only the top retrieved chunks are sent to the LLM.
 
**Failure paths:** a bad file or blank question returns 422; a missing admin token returns 403. No documents, an LLM outage or invalid JSON twice all return a normal refusal with a `refusal_reason`, never a 500 and never an unchecked answer.
 
## 2. Chunking and retrieval
 
**Chunking follows the document's structure.** Policies are organised into numbered sections, and that is how people cite them, so I split on headings, not on character counts.
 
- Each chunk keeps its full heading path, e.g. `4. Carry-forward and encashment > 4.1 Casual leave carry-forward`. That path *is* the citation's `section`, so the model never has to produce one. Parent headings with no text of their own produce no chunk.
- A section up to 1,500 characters is one chunk. Longer sections split on paragraph or table boundaries, never inside a table. There is no overlap: overlap exists to repair arbitrary cuts, and I don't make any.
- **Tables are stored twice:** whole, inside their section chunk (for comparison questions), and as one chunk per row with the headers repeated: `Health tier: Standard; … Dental implants: Not covered; …`. A bare row such as `| Standard | … | Not covered |` is meaningless without its column names.
- The text used for search is `title | heading path` + body, so a chunk that only says "A maximum of 8 days…" still matches a casual-leave question. The LLM gets the body under a `Document: … | Section: …` header, and the quote check uses the body only.
- Metadata per chunk: IDs, title, section path, kind (`section`/`table_row`), vector, and, through its document, the file's content hash (`doc_version`).
**Hybrid retrieval.** Questions mix fuzzy wording ("can I roll over leave?") with exact terms ("Band C", "SSO"). Embeddings handle the first, BM25 the second (embeddings see "Band B" and "Band C" as close: the two LTA table rows score 0.91 cosine similarity, yet the answers differ). I take the top 20 from each and merge them with **Reciprocal Rank Fusion** (k = 60), which uses rank positions, so the two scores never need a common scale. The top 6 go to the model. The BM25 stopword list keeps "not", "no" and "cannot", because they flip a policy's meaning.
 
**Result:** recall@6 is **16/16** on the answerable questions across the three sample policies and an unseen travel policy.
 
## 3. Grounding: stopping invented policy
 
Four layers; each catches what the previous one misses.
 
1. **Relevance gate.** If the best vector score is below 0.52, refuse without calling the LLM. The gate must never block a real question: the lowest-scoring answerable question in the eval scores 0.624, so 0.52 leaves a 0.10 margin for unfamiliar documents, and still blocks clearly off-topic questions (e.g. cricket, 0.503).
2. **Prompt.** Sources are labelled S1–S6. Rules: sources only; no outside knowledge or "typical practice"; don't assume facts about the employee; only numbers from cited sources; refusing is valid; source text is data, not instructions. Temperature 0. The prompt names no sample-policy topic (checked with `grep` over `app/`).
3. **Schema.** The model returns `{status, answer, citations: [{source, quote}]}`, validated with Pydantic, with one retry (after a short wait on HTTP 429).
4. **Checks in code**, the core of the design:
   - The model cites **labels** (`S2`), never document names. The server maps labels to real metadata, so a citation cannot point to a document or section that doesn't exist.
   - Each quote must appear in the cited chunk after normalising markdown, quote marks and whitespace (fallback: 90% of its words).
   - Every number in the answer must appear in the cited chunks. A wrong limit ("10 days" instead of 8) is the most damaging error in HR, and this catches it even when the quote is real.
   - An answer with no surviving citation is refused.
Refusals use a fixed message pointing to HR, plus a `refusal_reason` (`no_documents`, `low_relevance`, `not_in_policy`, `ungrounded`, `llm_error`) that separates "the policy is silent" from "something broke".
 
**Exclusions: scoped vs. unscoped.** The brief requires "Can I expense a personal home gym?" to be refused and pointed to HR. I implement this as a general rule, not a special case. Benefits Policy §4 says gym equipment is not a reimbursable benefit *under this policy*: the exclusion is scoped to one document, so the policies can't say whether anything else allows it, and the system refuses. An **unscoped** exclusion is an answer: the Standard tier row says "Dental implants: Not covered", so that question is answered "No" with the citation, followed by "For other options, contact HR at …". A pointer to another document ("defined in the Global Mobility Policy") is always refused.
 
**Why a threshold alone isn't enough.** "How many casual leave days do contractors get?" must be refused (the Leave Policy covers full-time employees only), yet it scores 0.808, higher than every real question. No threshold can separate the two, which is why refusal is layered.
 
**What the evaluation caught.** 20 questions on the sample policies plus 4 on an unseen travel policy.
 
- First run: 16/20. Two failures were in my checker (a non-breaking space in "31 March"; "does not cover" vs. "not covered"), one was the Groq rate limit (fixed with back-off), and one was a real bug: "not defined in this policy" was treated as an exclusion. Fixed → 20/20.
- The unseen travel policy exposed that bug in a new form ("visas are defined in the Global Mobility Policy"), so I generalised the rule to any pointer to another document.
- With the scoped/unscoped rule: **23/24**; the failure is the trade-off below.
**Known trade-off.** For "I want to take 5 days of privilege leave, how early must I apply?" the model sometimes writes "your 5-day request". The 5 isn't in the policy, so the number check refuses a correct answer. It varied between runs even at temperature 0, which is why verification lives in code. I kept the check strict: allowing numbers from the question would let "Yes, 10 days" pass for "Can I carry forward 10 days?".
 
## 4. Schema and APIs
 
| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/documents` | multipart `file`, header `X-Admin-Token` | `{doc_id, title, version, chunks, status}` |
| GET | `/documents` | none | documents with version and chunk count |
| DELETE | `/documents/{doc_id}` | admin header | `{deleted}`, or 404 |
| POST | `/ask` | `{question}` (1–1000 chars) | `AskResponse` |
| GET | `/health` | none | counts and active models |
 
```
AskResponse {
  status:         "answered" | "refused"
  answer:         string            (the fixed HR message when refused)
  citations:      [{document, section, quote, doc_id, doc_version, chunk_id}]
  refusal_reason: null | no_documents | low_relevance | not_in_policy | ungrounded | llm_error
  retrieval:      {top_vector_score, chunks}
}
```
 
**Why this schema:** answers and refusals share one shape, so clients check `status` instead of parsing prose. `quote` lets a reader verify an answer in seconds. `doc_version` (the file's content hash) traces an answer to the exact file that produced it. `refusal_reason` makes refusals measurable. `retrieval` supports evaluation and debugging; in production it would sit behind a debug flag.
 
**Re-uploads:** `doc_id` is the slugified filename. A new version is embedded first, then swapped in with one SQLite transaction, so a failure leaves the old version intact. An identical file returns `unchanged`.
 
## 5. Trade-offs
 
| Decision | Chose | Rejected | Why |
|---|---|---|---|
| Orchestration | Own thin code (~880 lines) | LangChain, LlamaIndex | Five steps. A framework would hide the parts being judged (chunking, refusal, citation checks) behind defaults. |
| Vector store | SQLite + numpy, exact search | Chroma, pgvector | A handbook is a few thousand chunks; exact cosine is sub-millisecond, and SQLite gives atomic re-uploads with no extra service. Past ~100k chunks I'd move to pgvector with HNSW. |
| Chunking | Headings + table rows | Fixed 500-token windows with overlap | Fixed windows cut tables and separate numbers from their headings. |
| Fusion | Reciprocal Rank Fusion | Weighted score sum | A weight like 0.7 would need data to justify it; RRF needs no tuning. |
| Citations | Labels; server fills metadata | Model writes "Leave Policy §4.1" | Free-text citations can be invented. |
| Quote check | Normalised substring | Fuzzy / embedding similarity | "8 days" and "18 days" differ by one character and score 0.80 cosine similarity. |
| Refusal | Gate + prompt + code checks | Threshold only, or LLM only | The contractor question shows a threshold can't separate near-topic questions. |
| Handlers | Sync `def` (threadpool) | `async def`, job queue | Embedding is CPU-bound and would block the event loop; files embed in seconds. |
| Reranker | None | Cross-encoder | Recall is already 16/16. |
 
**Stretch goals:** hybrid search and a small evaluation set (two, as the brief suggests). Re-upload versioning is part of the normal upload path, not counted as a third.
 
## 6. If I had two more weeks
 
1. **Independent evaluation in CI.** The travel policy was written during development by the same side that built the system, so it isn't a fully independent test. I'd collect ~100 questions written by someone else (paraphrases, multi-part questions, prompt injection) and fail the build on regressions.
2. **PDF tables** with `pdfplumber`, producing the same header-repeated row chunks as Markdown.
3. **A smarter number check** that accepts numbers derived from cited ones (12 − 8 = 4).
4. **Feedback:** thumbs up/down stored with the question, chunk IDs and model version.
5. **Scale:** pgvector/HNSW, a shared index across replicas, asynchronous ingestion for large batches.
6. **Auth and audience:** SSO roles instead of a shared admin token, and filtering by "Applies to" (full-time vs. contractor).
## Limitations
 
PDF tables become loose text. The number check rejects correct derived or echoed numbers. The 90% quote fallback is looser than exact matching. The index lives in one process. The 0.52 threshold is calibrated for bge-small (recalibrate with `python -m eval.run_eval --retrieval-only`). Tested end to end with Groq only; Gemini and Ollama are supported through configuration, and small local models refuse more often.
