from __future__ import annotations

import json
import re

import httpx

from . import config


class LLMError(RuntimeError):
    pass


def chat_json(system: str, user: str) -> dict:
    if not config.LLM_API_KEY:
        raise LLMError("LLM_API_KEY is not set")
    try:
        resp = httpx.post(
            f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
            json={
                "model": config.LLM_MODEL,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
            },
            timeout=config.LLM_TIMEOUT_S,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
        raise LLMError(f"LLM request failed: {e}") from e
    return parse_json_object(content)


def parse_json_object(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    raise LLMError("LLM did not return valid JSON")
