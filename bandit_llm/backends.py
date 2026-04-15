import os
from typing import Any, Dict, Optional, Type

from ollama import chat
from openai import AzureOpenAI
from pydantic import BaseModel

_azure_client = None
_hf_cache: dict = {}
_vllm_cache: dict = {}


def _get_azure_client() -> AzureOpenAI:
    global _azure_client
    if _azure_client is None:
        _azure_client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
    return _azure_client


def get_ollama_response(
    model_config: dict,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    num_predict: int,
    top_p: float,
    response_schema: Optional[Type[BaseModel]] = None,
) -> tuple:
    """Call Ollama and return (response_text, prompt_tokens, completion_tokens)."""
    kwargs: Dict[str, Any] = {
        "model": model_config["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "top_p": top_p,
        },
    }
    if response_schema is not None:
        kwargs["format"] = response_schema.model_json_schema()

    response = chat(**kwargs)
    prompt_tokens = response.get("prompt_eval_count", 0)
    completion_tokens = response.get("eval_count", 0)
    return response["message"]["content"], prompt_tokens, completion_tokens


def get_azure_response(
    model_config: dict,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    num_predict: int,
    top_p: float,
    response_schema: Optional[Type[BaseModel]] = None,
) -> tuple:
    """Call Azure OpenAI and return (response_text, prompt_tokens, completion_tokens)."""
    client = _get_azure_client()
    kwargs: Dict[str, Any] = {
        "model": model_config["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": num_predict,
        "top_p": top_p,
    }
    if response_schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "recommendation",
                "strict": True,
                "schema": response_schema.model_json_schema(),
            },
        }

    response = client.chat.completions.create(**kwargs)
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    return response.choices[0].message.content, prompt_tokens, completion_tokens


def _get_hf_model(model_name: str):
    """Lazy-load and cache a HuggingFace model + tokenizer."""
    if model_name not in _hf_cache:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"  Loading HuggingFace model: {model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _hf_cache[model_name] = (tokenizer, model)
    return _hf_cache[model_name]


def get_hf_response(
    model_config: dict,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    num_predict: int,
    top_p: float,
    response_schema: Optional[Type[BaseModel]] = None,
) -> tuple:
    """Run HuggingFace inference. Free-form text — schema is ignored, parser handles it."""
    import torch

    tokenizer, model = _get_hf_model(model_config["model"])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_tokens = inputs["input_ids"].shape[1]

    gen_kwargs = {
        "max_new_tokens": num_predict,
        "top_p": top_p,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_tokens = output_ids[0][prompt_tokens:]
    completion_tokens = len(new_tokens)
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response_text, prompt_tokens, completion_tokens


def _get_vllm_engine(model_name: str):
    """Lazy-load and cache a vLLM engine."""
    if model_name not in _vllm_cache:
        from vllm import LLM

        print(f"  Loading vLLM model: {model_name} ...")
        _vllm_cache[model_name] = LLM(
            model=model_name,
            dtype="float16",
            gpu_memory_utilization=0.9,
            trust_remote_code=False,
        )
    return _vllm_cache[model_name]


def get_vllm_response(
    model_config: dict,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    num_predict: int,
    top_p: float,
    response_schema: Optional[Type[BaseModel]] = None,
) -> tuple:
    """Run vLLM inference. Free-form text — schema is ignored, parser handles it."""
    from vllm import SamplingParams

    llm = _get_vllm_engine(model_config["model"])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    sampling_params = SamplingParams(
        max_tokens=num_predict,
        top_p=top_p,
        temperature=temperature if temperature > 0 else 0.0,
    )
    outputs = llm.chat(messages, sampling_params, use_tqdm=False)

    result = outputs[0]
    response_text = result.outputs[0].text
    prompt_tokens = len(result.prompt_token_ids)
    completion_tokens = len(result.outputs[0].token_ids)
    return response_text, prompt_tokens, completion_tokens


_BACKEND_REGISTRY = {
    "ollama": get_ollama_response,
    "azure": get_azure_response,
    "hf": get_hf_response,
    "vllm": get_vllm_response,
}


def get_model_response(
    model_config: dict,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    num_predict: int,
    top_p: float,
    response_schema: Optional[Type[BaseModel]] = None,
) -> tuple:
    """Dispatch to the backend named in model_config['provider']."""
    provider = model_config["provider"]
    try:
        backend = _BACKEND_REGISTRY[provider]
    except KeyError as e:
        raise ValueError(
            f"Unknown provider {provider!r}. Known providers: {sorted(_BACKEND_REGISTRY)}"
        ) from e
    return backend(
        model_config,
        system_prompt,
        user_prompt,
        temperature,
        num_predict,
        top_p,
        response_schema=response_schema,
    )
