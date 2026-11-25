# HR Policy Assistant (RAG)

Ask HR policy questions in plain English. Answers come **only** from uploaded
policies, each with a verified citation (document + section + exact quote).
If the policies don't say, it refuses and points you to HR.

Design rationale: see **[DESIGN.md](DESIGN.md)**.

## Run locally (≈10 min)

Requires Python 3.10+.

```bash
git clone <this repo> && cd hr-rag
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
# Linux only, avoids a 2 GB CUDA download:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

cp .env.example .env        # then set LLM_API_KEY (see "LLM provider" below)

python -m app.cli ingest sample_policies/        # first run downloads the embedding model (~130 MB)
uvicorn app.main:app --reload                    # API at http://localhost:8000/docs
streamlit run ui/streamlit_app.py                # UI at http://localhost:8501 (second terminal)
```

Quick check without the UI:

```bash
python -m app.cli ask "Does the Standard health tier cover dental implants?"
```

## LLM provider

The app talks to any OpenAI-compatible chat endpoint. Pick one via `LLM_PROVIDER`:

| Provider | Key | Default model | Notes |
|---|---|---|---|
| `groq` (default) | free at console.groq.com | `openai/gpt-oss-120b` | Fast; free-tier rate limits apply |
| `gemini` | free at aistudio.google.com | `gemini-3.7-flash` | Free-tier prompts may be used by Google to improve products |
| `ollama` | none | `llama3.1:8b` | Fully local; small models refuse/cite less reliably |

Model IDs on free tiers change often. If you get a 404 or "model decommissioned",
set `LLM_MODEL` to a current model from the provider's list.

Free tiers are rate-limited. Groq's free tier allows about 8,000 tokens per minute for
`openai/gpt-oss-120b`, which is roughly 4 questions a minute. On a 429 the app waits
(using the `Retry-After` header, capped at 10 s) and retries once; if that also fails it
returns a refusal with `refusal_reason: "llm_error"`. For the full eval use `--delay 15`.

**Embeddings** are always local: `BAAI/bge-small-en-v1.5` via sentence-transformers.
Policy text never leaves your machine except the top retrieved chunks sent to the LLM.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq` / `gemini` / `ollama` |
| `LLM_API_KEY` | — | Provider key (not needed for Ollama) |
| `LLM_MODEL`, `LLM_BASE_URL` | per provider | Overrides |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Any sentence-transformers model (`hashing` = test-only) |
| `MIN_VECTOR_SCORE` | `0.52` | Retrieval gate. Calibrate with the eval (below) |
| `TOP_K` | `6` | Chunks sent to the LLM |
| `ADMIN_TOKEN` | `admin` | Required in `X-Admin-Token` to upload/delete |
| `HR_CONTACT` | `hr@example.com` | Shown in refusals |
| `DATA_DIR` | `data` | SQLite location |
| `API_URL` | `http://localhost:8000` | Used by the Streamlit UI |

## API

```bash
# Upload (admin)
curl -H "X-Admin-Token: admin" -F file=@sample_policies/leave-policy.md localhost:8000/documents
# Ask
curl localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "What is the casual leave carry-forward limit?"}'
```

Response shape (values illustrative):

```json
{
  "status": "answered",
  "answer": "You can carry forward a maximum of 8 days of unused casual leave; it must be used by 31 March.",
  "citations": [{
    "document": "Leave Policy",
    "section": "4. Carry-forward and encashment > 4.1 Casual leave carry-forward",
    "quote": "A maximum of 8 days of unused casual leave may be carried forward",
    "doc_id": "leave-policy", "doc_version": "f22fac9a65dc", "chunk_id": "leave-policy#6"
  }],
  "refusal_reason": null,
  "retrieval": {"top_vector_score": 0.81, "chunks": ["leave-policy#6", "..."]}
}
```

Other endpoints: `GET /documents`, `DELETE /documents/{doc_id}` (admin), `GET /health`.
Interactive docs at `/docs`.

## Tests and evaluation

```bash
pytest -q                                   # 29 tests, offline, no API key, no model download
python -m eval.run_eval --retrieval-only    # recall@k + suggested MIN_VECTOR_SCORE (no LLM calls)
python -m eval.run_eval --delay 15          # full pipeline on 20 questions -> eval/results.json
```

## Project layout

```
app/
  ingest.py      parse + structure-aware chunking (headings, table rows)
  store.py       SQLite store + in-memory vector/BM25 index
  retrieval.py   hybrid search (BM25 + vectors, RRF) + relevance gate
  answer.py      prompt, LLM call, citation/quote/number validation, refusals
  llm.py         OpenAI-compatible client (Groq / Gemini / Ollama)
  models.py      Pydantic request/response schema
  main.py        FastAPI endpoints
  cli.py         ingest / ask from the terminal
ui/streamlit_app.py
eval/questions.yaml, eval/run_eval.py
tests/
sample_policies/
```
