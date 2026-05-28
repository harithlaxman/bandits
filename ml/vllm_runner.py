from vllm import LLM, SamplingParams

_llms = {}


def _get_llm(model: str):
    if model not in _llms:
        _llms[model] = LLM(model=model, dtype="bfloat16")
    return _llms[model]


def generate(
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.0,
) -> str:
    llm = _get_llm(model)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=512,
    )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    outputs = llm.chat(messages, sampling_params)
    return outputs[0].outputs[0].text
