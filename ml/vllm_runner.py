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

    # Build a single text prompt, prepending the system message if provided
    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    outputs = llm.generate([full_prompt], sampling_params)
    return outputs[0].outputs[0].text
