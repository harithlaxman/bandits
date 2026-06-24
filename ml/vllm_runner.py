from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

_llms = {}

CHAT_MODEL_TYPES = {"instruct", "chat"}


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
    top_p: float = 1.0,
    top_k: int = -1,
    model_type: str = "instruct",
    choices: list[str] | None = None,
) -> str:
    llm = _get_llm(model)

    structured_outputs = (
        StructuredOutputsParams(choice=choices) if choices else None
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=512,
        ignore_eos=False,
        structured_outputs=structured_outputs,
    )

    if model_type in CHAT_MODEL_TYPES:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        outputs = llm.chat(messages, sampling_params)
    else:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        outputs = llm.generate(full_prompt, sampling_params)

    return outputs[0].outputs[0].text
