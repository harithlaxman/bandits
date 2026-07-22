"""vLLM backend. Serves an instruct SLM offline and generates one button name per prompt.

Trajectories are independent, so at each decision step we batch one prompt per active
trajectory into a single ``chat`` call to saturate the GPU. Each prompt is sent as one
user message (the model's chat template is applied by ``LLM.chat``).
"""

from typing import List

from vllm import LLM, SamplingParams


class VLLMRunner:
    def __init__(self, model: str, **llm_kwargs):
        self.llm = LLM(model=model, enable_prefix_caching=True, **llm_kwargs)

    def _sampling_params(self, cfg: dict) -> SamplingParams:
        return SamplingParams(
            temperature=cfg.get("temperature", 1.0),
            top_p=cfg.get("top_p", 1.0),
            top_k=cfg.get("top_k", -1),
            max_tokens=cfg.get("max_tokens", 20),
        )

    def chat(self, prompts: List[str], cfg: dict) -> List[str]:
        """Batch-generate one response per prompt. Each prompt is a single user message."""
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        outputs = self.llm.chat(conversations, self._sampling_params(cfg), use_tqdm=False)
        return [o.outputs[0].text for o in outputs]
