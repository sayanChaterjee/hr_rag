import importlib

from app import config


def test_empty_env_values_fall_back_to_defaults(monkeypatch):
    for name in ["LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL", "LLM_TIMEOUT_S", "TOP_K"]:
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    try:
        c = importlib.reload(config)
        assert c.LLM_API_KEY == "ollama"
        assert c.LLM_MODEL == "llama3.1:8b"
        assert c.LLM_BASE_URL == "http://localhost:11434/v1"
        assert c.LLM_TIMEOUT_S == 30.0
        assert c.TOP_K == 6
    finally:
        monkeypatch.undo()
        importlib.reload(config)
