import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
_PROVIDER_DEFAULTS = {
    "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.7-flash"),
    "ollama": ("http://localhost:11434/v1", "llama3.1:8b"),
}
_base, _model = _PROVIDER_DEFAULTS.get(LLM_PROVIDER, _PROVIDER_DEFAULTS["groq"])
LLM_BASE_URL = os.getenv("LLM_BASE_URL", _base)
LLM_MODEL = os.getenv("LLM_MODEL", _model)
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama" if LLM_PROVIDER == "ollama" else "")
LLM_TIMEOUT_S = _f("LLM_TIMEOUT_S", 30)

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_QUERY_PREFIX = os.getenv(
    "EMBED_QUERY_PREFIX", "Represent this sentence for searching relevant passages: "
)

MAX_CHUNK_CHARS = _i("MAX_CHUNK_CHARS", 1500)

CANDIDATES_PER_RETRIEVER = _i("CANDIDATES_PER_RETRIEVER", 20)
TOP_K = _i("TOP_K", 6)
RRF_K = _i("RRF_K", 60)
MIN_VECTOR_SCORE = _f("MIN_VECTOR_SCORE", 0.52)

HR_CONTACT = os.getenv("HR_CONTACT", "hr@example.com")
REFUSAL_MESSAGE = (
    "I don't have enough information in the uploaded HR policies to answer that. "
    f"Please contact HR at {HR_CONTACT}."
)
MAX_UPLOAD_BYTES = _i("MAX_UPLOAD_BYTES", 5 * 1024 * 1024)
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "admin")

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DB_PATH = DATA_DIR / "hr_rag.sqlite3"
