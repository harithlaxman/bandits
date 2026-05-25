from transformers import pipeline

_pipes = {}


def _get_pipe(model: str):
    if model not in _pipes:
        _pipes[model] = pipeline(
            "text-generation",
            model=model,
            dtype="auto",
            device_map="auto",
        )
    return _pipes[model]


def generate(
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.0,
) -> str:
    pipe = _get_pipe(model)
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})

    do_sample = temperature > 0.0
    out = pipe(
        messages,
        max_new_tokens=512,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        return_full_text=False,
    )
    return out[0]["generated_text"]
