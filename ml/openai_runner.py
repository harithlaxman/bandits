import os

from openai import AzureOpenAI

_ENDPOINT    = os.environ.get("AZURE_OPENAI_ENDPOINT")
_API_KEY     = os.environ.get("AZURE_OPENAI_API_KEY")
_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")

_client: AzureOpenAI | None = None


def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        if not _API_KEY or not _ENDPOINT:
            raise RuntimeError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set"
            )
        _client = AzureOpenAI(
            api_key=_API_KEY,
            azure_endpoint=_ENDPOINT,
            api_version=_API_VERSION,
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