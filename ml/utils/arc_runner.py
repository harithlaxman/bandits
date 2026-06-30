import os

from openai import OpenAI

_API_KEY  = os.environ.get("ARC_API_KEY")
_BASE_URL = os.environ.get("ARC_BASE_URL", "https://llm-api.arc.vt.edu/api/v1")

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not _API_KEY:
            raise RuntimeError("ARC_API_KEY must be set")
        _client = OpenAI(
            api_key=_API_KEY,
            base_url=_BASE_URL,
        )
    return _client


def generate(
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.0,
) -> str:
    client = _get_client()
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content
