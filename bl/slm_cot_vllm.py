import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bandit_llm import BanditLLM  # noqa: E402
from utils import (  # noqa: E402
    SYSTEM_PROMPT_COT,
    USERS_DF,
    build_prompt_cot,
    get_article_title,
    get_user_rounds,
    get_user_seed_articles,
)

# ─── Config ───────────────────────────────────────────────────────────
TURNS = 50
NUM_PREDICT = 1024
TOP_P = 0.9
RESULTS_DIR = "results"

MODELS = [
    {"name": "llama-3.1-8b", "provider": "vllm", "model": "meta-llama/Llama-3.1-8B-Instruct"},
    # {"name": "mistral-7b", "provider": "vllm", "model": "mistralai/Mistral-7B-Instruct-v0.3"},
    # {"name": "granite-3.3-2b", "provider": "vllm", "model": "ibm-granite/granite-3.3-2b-instruct"},
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp-mode", choices=["fixed", "random", "decreasing"], default="random")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature value for fixed mode")
    args = parser.parse_args()

    bandit = BanditLLM(
        models=MODELS,
        system_prompt=SYSTEM_PROMPT_COT,
        prompt_builder=build_prompt_cot,
        item_title_fn=get_article_title,
        results_dir=RESULTS_DIR,
        turns=TURNS,
        num_predict=NUM_PREDICT,
        top_p=TOP_P,
        prompt_mode="cot",
    )

    bandit.run_all(
        users=USERS_DF.index.unique().to_list(),
        seed_text_fn=get_user_seed_articles,
        rounds_fn=get_user_rounds,
        temp_mode=args.temp_mode,
        fixed_temp=args.temperature,
    )
