import json
import os
import random
from datetime import datetime
from typing import Any, Callable

from .backends import get_model_response
from .plotting import plot_learning_curves
from .prompts import Recommendation, parse_cot_response, parse_model_choice

MAX_RETRIES = 3


class BanditLLM:
    """
    Domain-agnostic orchestrator for the LLM-as-bandit experiments.

    The caller supplies a `prompt_builder` that knows how to render its own
    domain (news / movies / …) and an `item_title_fn` for compact log lines.
    Everything else — model dispatch, temperature schedule, candidate shuffle,
    retry on parse failure, reward bookkeeping, JSON dump and learning-curve
    plot — lives here and is shared.
    """

    def __init__(
        self,
        models: list,
        system_prompt: str,
        prompt_builder: Callable[[str, list, list, int, int], str],
        item_title_fn: Callable[[Any], str],
        results_dir: str = "results",
        turns: int = 50,
        num_predict: int = 20,
        top_p: float = 0.9,
        plot_title_prefix: str = "",
        prompt_mode: str = "plain",
    ):
        if prompt_mode not in ("plain", "cot"):
            raise ValueError(f"prompt_mode must be 'plain' or 'cot', got {prompt_mode!r}")
        self.models = models
        self.system_prompt = system_prompt
        self.prompt_builder = prompt_builder
        self.item_title_fn = item_title_fn
        self.results_dir = results_dir
        self.turns = turns
        self.num_predict = num_predict
        self.top_p = top_p
        self.plot_title_prefix = plot_title_prefix
        self.prompt_mode = prompt_mode
        self.response_schema = Recommendation if prompt_mode == "cot" else None

    def run_user_experiment(
        self,
        user_id: Any,
        user_seed_text: str,
        rounds: list,
        model_config: dict,
        temp_mode: str = "random",
        fixed_temp: float = 0.3,
    ) -> dict:
        """Run the bandit experiment for a single user. Returns a results dict."""
        rec_history = []
        rewards = []
        temperatures = []
        reasoning_log = []
        cumulative_reward = 0
        cumulative_tokens = 0
        token_log = []
        parse_failures = 0

        num_rounds = min(self.turns, len(rounds))

        for round_num in range(1, num_rounds + 1):
            candidates, positives = rounds[round_num - 1]

            if len(candidates) == 0:
                continue

            random.seed(round_num + hash(user_id))
            shuffled_indices = list(range(len(candidates)))
            random.shuffle(shuffled_indices)
            shuffled_candidates = [candidates[i] for i in shuffled_indices]

            prompt = self.prompt_builder(
                user_seed_text,
                rec_history,
                shuffled_candidates,
                round_num,
                cumulative_reward,
            )

            if temp_mode == "fixed":
                round_temperature = fixed_temp
            elif temp_mode == "decreasing":
                round_temperature = round(
                    1.0 - (round_num - 1) / max(num_rounds - 1, 1), 1
                )
            else:
                round_temperature = random.choice(
                    [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
                )

            choice_idx = -1
            reasoning = ""
            round_tokens = 0
            for attempt in range(1, MAX_RETRIES + 1):
                response_text, prompt_tokens, completion_tokens = get_model_response(
                    model_config,
                    self.system_prompt,
                    prompt,
                    round_temperature,
                    self.num_predict,
                    self.top_p,
                    response_schema=self.response_schema,
                )
                round_tokens = prompt_tokens + completion_tokens
                cumulative_tokens += round_tokens
                token_log.append((prompt_tokens, completion_tokens, cumulative_tokens))

                if self.prompt_mode == "cot":
                    choice_idx, reasoning = parse_cot_response(
                        response_text, len(shuffled_candidates)
                    )
                else:
                    choice_idx = parse_model_choice(
                        response_text, len(shuffled_candidates)
                    )

                if choice_idx != -1:
                    break

                parse_failures += 1
                expected_msg = (
                    f"JSON with choice 1-{len(shuffled_candidates)}"
                    if self.prompt_mode == "cot"
                    else f"a number 1-{len(shuffled_candidates)}"
                )
                print(
                    f"  [Parse failure] Round {round_num}, attempt {attempt}/{MAX_RETRIES}: "
                    f"model returned '{response_text.strip()[:200]}' "
                    f"(expected {expected_msg})"
                )

            if choice_idx == -1:
                print(
                    f"  [Skipping] Round {round_num}: all {MAX_RETRIES} attempts failed to parse"
                )
                continue

            chosen_id = shuffled_candidates[choice_idx]
            chosen_title = self.item_title_fn(chosen_id)

            was_positive = chosen_id in positives
            reward = 1 if was_positive else 0
            cumulative_reward += reward
            rewards.append(reward)
            temperatures.append(round_temperature)
            if self.prompt_mode == "cot":
                reasoning_log.append(reasoning)

            rec_history.append((chosen_title, was_positive))

            log_line = (
                f"  Round {round_num}/{num_rounds}: "
                f"Picked '{str(chosen_title)[:60]}' → "
                f"{'HIT' if was_positive else 'MISS'} "
                f"(Cumulative: {cumulative_reward}/{round_num}) "
                f"[Tokens: {round_tokens} | Total: {cumulative_tokens}]"
            )
            if self.prompt_mode == "cot" and reasoning:
                preview = reasoning[:80].replace("\n", " ")
                log_line += f"\n    Reasoning: {preview}..."
            print(log_line)

        result = {
            "user_id": user_id,
            "model": model_config["name"],
            "temperature": fixed_temp if temp_mode == "fixed" else temp_mode,
            "temperatures": temperatures,
            "rewards": rewards,
            "cumulative_reward": cumulative_reward,
            "total_rounds": len(rewards),
            "ctr": cumulative_reward / len(rewards) if rewards else 0,
            "total_tokens": cumulative_tokens,
            "token_log": token_log,
            "parse_failures": parse_failures,
        }
        if self.prompt_mode == "cot":
            result["reasoning_log"] = reasoning_log
        return result

    def run_all(
        self,
        users: list,
        seed_text_fn: Callable[[Any], str],
        rounds_fn: Callable[[Any], list],
        temp_mode: str = "random",
        fixed_temp: float = 0.3,
    ) -> dict:
        """
        Run every model against every user. Saves a JSON per model and a
        single comparison plot. Returns the in-memory `results_by_model` dict.
        """
        temp_label = f"fixed={fixed_temp}" if temp_mode == "fixed" else temp_mode
        os.makedirs(self.results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_by_model: dict = {}

        print(f"Total users available: {len(users)}")

        for model_config in self.models:
            model_name = model_config["name"]
            print(f"\n{'#' * 60}")
            print(f"MODEL: {model_name}")
            print(f"{'#' * 60}")

            all_results = []

            for i, user_id in enumerate(users):
                print(f"\n{'=' * 60}")
                print(f"User {i + 1}/{len(users)}: {user_id}")
                print(f"{'=' * 60}")

                user_seed_text = seed_text_fn(user_id)
                rounds = rounds_fn(user_id)
                result = self.run_user_experiment(
                    user_id=user_id,
                    user_seed_text=user_seed_text,
                    rounds=rounds,
                    model_config=model_config,
                    temp_mode=temp_mode,
                    fixed_temp=fixed_temp,
                )
                all_results.append(result)

                print(
                    f"\n  Summary — CTR: {result['ctr']:.3f} | "
                    f"Parse failures: {result['parse_failures']}"
                )

            results_by_model[model_name] = all_results

            safe_name = model_name.replace(":", "_").replace("/", "_")
            results_file = os.path.join(self.results_dir, f"{safe_name}_{timestamp}.json")
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)

            avg_ctr = sum(r["ctr"] for r in all_results) / len(all_results)
            total_failures = sum(r["parse_failures"] for r in all_results)

            print(f"\n{'=' * 60}")
            print(f"FINAL RESULTS — {model_name}")
            print(f"{'=' * 60}")
            print(f"Users evaluated: {len(all_results)}")
            print(f"Average CTR:     {avg_ctr:.4f}")
            print(f"Total parse failures: {total_failures}")
            print(f"Results saved to: {results_file}")

        plot_path = os.path.join(self.results_dir, f"comparison_{timestamp}.png")
        plot_learning_curves(
            results_by_model,
            plot_path,
            temp_label,
            title_prefix=self.plot_title_prefix,
        )
        return results_by_model
