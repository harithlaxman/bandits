import argparse
import ast
import json
import os
import random
import re
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import List

import httpx
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path("./data/")
METADATA_CSV = DATA_DIR / "metadata.csv"

MAX_PARSE_RETRIES = 3
N_CHOICES = 5
TOP_K_AGGREGATE = 3

BANDIT_PREAMBLE = """You are a recommendation agent acting as a contextual bandit. \
Each round you are shown a user's taste profile and a set of candidate movies, and \
you pick exactly ONE candidate to recommend; you then learn whether the user LIKED \
it. Your objective is to MAXIMIZE the total number of liked recommendations over \
all rounds. To do this, balance two competing pressures:
- EXPLOITATION: recommend movies you are confident this user will like.
- EXPLORATION: sometimes recommend movies whose appeal is uncertain, to learn \
tastes you cannot yet predict — this pays off most when you know little about the \
user.

"""


REQUIRED_CFG_KEYS = {
    "model", "runner", "num_epochs", "num_users",
    "num_steps", "seed", "temperature", "output",
}
VALID_RUNNERS = {"ollama", "vllm", "openai", "random", "huggingface"}
RESUME_MATCH_KEYS = (
    "model", "runner", "num_epochs", "num_users",
    "num_steps", "seed", "temperature",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True,
                   help="path to a JSON config file (see configs/)")
    p.add_argument("--resume", type=Path, default=None,
                   help="path to an existing run dir to resume "
                        "(must contain config.json + epochs.jsonl)")
    p.add_argument("--mode", choices=["stationary", "nonstationary"],
                   default="stationary",
                   help="nonstationary: left-rotate the labels array by one after "
                        "each user's 30th interaction (reward mapping shifts; "
                        "candidate movies unchanged)")
    p.add_argument("--positive-only", action="store_true",
                   help="only include past recommendations the user LIKED in history prompt")
    return p.parse_args()


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    missing = REQUIRED_CFG_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"config {path} missing keys: {sorted(missing)}")
    if cfg["runner"] not in VALID_RUNNERS:
        raise ValueError(
            f"unknown runner: {cfg['runner']!r} (valid: {sorted(VALID_RUNNERS)})"
        )
    cfg["output"] = Path(cfg["output"])
    cfg["config_name"] = path.stem
    return cfg


def get_dataset(num_users: int):
    df = pd.read_pickle(DATA_DIR / "impressions_stationary.pkl")
    users = df["userId"].unique().tolist()
    sampled_users = random.sample(users, num_users)
    impr_df = df[df["userId"].isin(sampled_users)]

    cold_start_df = impr_df[impr_df["phase"] == "cold_start"]
    normal_df = impr_df[impr_df["phase"] == "normal"]

    cold_start_df = cold_start_df.set_index("userId")
    normal_df = normal_df.set_index("impression_id")

    metadata_df = pd.read_csv(METADATA_CSV)

    movie_ids = set()
    for user_id in sampled_users:
        for mid in cold_start_df.loc[user_id].impression:
            movie_ids.add(mid)
    for _, row in normal_df.iterrows():
        for mid in row["impression"]:
            movie_ids.add(mid)

    metadata_df = metadata_df[metadata_df["tmdbId"].isin(movie_ids)]

    mid_to_data = {}
    for row in metadata_df.itertuples():
        mid_to_data[row.tmdbId] = f"""Title: {row.title}
Genres: {row.genres}
Overview: {row.overview}
Tagline: {row.tagline}
Runtime: {row.runtime} minutes
Actors: {", ".join(ast.literal_eval(str(row.top_actors)))}
"""

    return cold_start_df, normal_df, mid_to_data


def get_coldstart_prompt(mids: List[int], labels: List[int], mid_to_data) -> str:
    prompt = "USER HISTORY:\n"
    for mid, label in zip(mids, labels):
        if label == 1:
            prompt += "The user LIKED:\n"
        elif label == -1:
            prompt += "The user DID NOT LIKE:\n"
        else:
            prompt += "The user was NOT A FAN of the movie:\n"
        prompt += mid_to_data[mid] + "\n"
    return prompt


def get_history_prompt(
    interactions: list[dict], mid_to_data, positive_only: bool
) -> str:
    prompt = "YOUR PAST RECOMMENDATIONS:\n"
    for it in interactions:
        if it["chosen_mid"] is None:
            continue
        if positive_only and it["reward"] != 1:
            continue
        prompt += "You recommended:\n"
        prompt += mid_to_data[it["chosen_mid"]]
        if it["reward"] == 1:
            prompt += "and the user LIKED the movie\n"
        else:
            prompt += "but the user DID NOT LIKE the movie\n"
    return prompt


def get_candidates_prompt(mids: List[int], mid_to_data) -> str:
    prompt = "CANDIDATES (choose ONE):\n"
    for i, mid in enumerate(mids, 1):
        prompt += f"[{i}]\n{mid_to_data[mid]}\n"
    prompt += (
        'Reply with ONLY the number of your choice in this exact format: '
        '"CHOICE: <number>". No other text.\n'
    )
    return prompt


def get_aggregate_prompt(
    liked_counts, disliked_counts, mid_to_data, top_k=TOP_K_AGGREGATE
) -> str:
    if not liked_counts and not disliked_counts:
        return ""
    prompt = "ACROSS ALL USERS SO FAR:\n"
    top_liked = [mid for mid, _ in liked_counts.most_common(top_k)]
    top_disliked = [mid for mid, _ in disliked_counts.most_common(top_k)]
    if top_liked:
        prompt += "Users tend to LIKE these movies:\n"
        for mid in top_liked:
            prompt += mid_to_data[mid] + "\n"
    if top_disliked:
        prompt += "Users tend to DISLIKE these movies:\n"
        for mid in top_disliked:
            prompt += mid_to_data[mid] + "\n"
    return prompt


def parse_choice(text: str, n: int = N_CHOICES) -> int | None:
    m = re.search(r"CHOICE:\s*([1-9])", text)
    if not m:
        return None
    idx = int(m.group(1)) - 1
    return idx if 0 <= idx < n else None


def make_get_response(runner: str, model: str, temperature: float):
    if runner == "ollama":
        from ollama_runner import generate

        def _ollama_call(prompt: str) -> str:
            return generate(model, prompt, temperature=temperature)

        return _ollama_call

    if runner == "vllm":
        from vllm_runner import generate as vllm_generate

        def _vllm_call(prompt: str) -> str:
            return vllm_generate(model, prompt, temperature=temperature)

        return _vllm_call

    if runner == "huggingface":
        from huggingface_runner import generate as hf_generate

        def _hf_call(prompt: str) -> str:
            return hf_generate(model, prompt, temperature=temperature)

        return _hf_call

    if runner == "openai":
        from openai_runner import generate as openai_generate

        def _openai_call(prompt: str) -> str:
            return openai_generate(model, prompt, temperature=temperature)

        return _openai_call

    if runner == "random":
        sys_rng = random.SystemRandom()

        def _random_call(_prompt: str) -> str:
            return f"CHOICE: {sys_rng.randint(1, N_CHOICES)}"

        return _random_call

    raise ValueError(f"unknown runner: {runner}")


def get_choice(
    get_response, prompt: str
) -> tuple[int | None, str, float]:
    total_ms = 0.0
    last_resp = ""
    for _ in range(MAX_PARSE_RETRIES):
        t0 = time.perf_counter()
        try:
            last_resp = get_response(prompt)
        except httpx.TimeoutException as e:
            total_ms += (time.perf_counter() - t0) * 1000
            last_resp = f"<timeout: {e}>"
            tqdm.write(last_resp)
            continue
        total_ms += (time.perf_counter() - t0) * 1000
        idx = parse_choice(last_resp)
        if idx is not None:
            return idx, last_resp, total_ms
    return None, last_resp, total_ms


def main():
    args = parse_args()
    cfg = load_config(args.config)
    model       = cfg["model"]
    runner      = cfg["runner"]
    num_epochs  = cfg["num_epochs"]
    num_users   = cfg["num_users"]
    num_steps   = cfg["num_steps"]
    seed        = cfg["seed"]
    temperature = cfg["temperature"]
    output      = cfg["output"]
    config_name = cfg["config_name"]
    positive_only = args.positive_only

    random.seed(seed)
    get_response = make_get_response(runner, model, temperature)

    config_record = {
        "config_name": config_name,
        "model": model,
        "runner": runner,
        "num_epochs": num_epochs,
        "num_users": num_users,
        "num_steps": num_steps,
        "seed": seed,
        "temperature": temperature,
        "output": str(output),
        "max_parse_retries": MAX_PARSE_RETRIES,
        "positive_only": positive_only,
    }

    if args.resume is not None:
        out_dir = args.resume
        cfg_path = out_dir / "config.json"
        epochs_path = out_dir / "epochs.jsonl"
        if not cfg_path.exists():
            raise FileNotFoundError(f"--resume dir missing config.json: {out_dir}")
        with open(cfg_path) as f:
            prior = json.load(f)
        mismatched = {
            k: (prior.get(k), config_record[k])
            for k in RESUME_MATCH_KEYS
            if prior.get(k) != config_record[k]
        }
        if mismatched:
            raise ValueError(
                f"--resume config mismatch on {sorted(mismatched)}: "
                f"prior vs new = {mismatched}"
            )
        n_completed = 0
        if epochs_path.exists():
            with open(epochs_path) as f:
                n_completed = sum(1 for line in f if line.strip())
        print(f"resuming {out_dir}: {n_completed} epoch(s) already completed")
    else:
        model_slug = model.replace(":", "_").replace("/", "_")
        timestamp = datetime.now().strftime("%m%d_%H%M")
        out_dir = output / f"{config_name}_{model_slug}_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        config_record["timestamp"] = timestamp
        with open(out_dir / "config.json", "w") as f:
            json.dump(config_record, f, indent=2)
        epochs_path = out_dir / "epochs.jsonl"
        n_completed = 0

    total_users = 0
    total_steps = 0

    for epoch in range(num_epochs):
        cold_starts_df, imprs_df, mid_to_data = get_dataset(num_users)
        user_queues: dict[int, deque] = {}
        for impr_id, uid in imprs_df["userId"].items():
            user_queues.setdefault(int(uid), deque()).append(int(impr_id))

        if epoch < n_completed:
            tqdm.write(f"resume: skipping completed epoch {epoch}")
            continue

        user_logs: dict[int, dict] = {}
        liked_counts: Counter = Counter()
        disliked_counts: Counter = Counter()
        steps_run = 0

        for step in tqdm(range(num_steps), desc=f"epoch {epoch}"):
            active_users = [u for u, q in user_queues.items() if q]
            if not active_users:
                tqdm.write("No more impressions in dataset")
                break

            uid = random.choice(active_users)
            impr_id = user_queues[uid].popleft()
            row = imprs_df.loc[impr_id]
            cands = [int(m) for m in row["impression"]]
            labels = [int(r) for r in row["labels"]]

            if args.mode == "nonstationary" and step >= num_steps // 2:
                labels = labels[1:] + labels[:1]

            if uid not in user_logs:
                cs = cold_starts_df.loc[uid]
                user_logs[uid] = {
                    "cold_start_prompt": get_coldstart_prompt(
                        cs.impression, cs.labels, mid_to_data
                    ),
                    "cold_start_mids": [int(m) for m in cs.impression],
                    "cold_start_labels": [int(r) for r in cs.labels],
                    "interactions": [],
                }

            prompt = (
                BANDIT_PREAMBLE
                + get_aggregate_prompt(liked_counts, disliked_counts, mid_to_data)
                + user_logs[uid]["cold_start_prompt"]
                + get_history_prompt(user_logs[uid]["interactions"], mid_to_data, positive_only)
                + get_candidates_prompt(cands, mid_to_data)
            )

            idx, raw, ms = get_choice(get_response, prompt)
            chosen_mid = cands[idx] if idx is not None else None
            reward = labels[idx] if idx is not None else None

            user_logs[uid]["interactions"].append({
                "step": step,
                "impression_id": int(impr_id),
                "candidates": cands,
                "labels": labels,
                "prompt": prompt,
                "raw_response": raw,
                "choice_idx": idx,
                "chosen_mid": chosen_mid,
                "reward": reward,
                "latency_ms": ms,
            })
            steps_run = step + 1

            if reward == 1:
                liked_counts[chosen_mid] += 1
            elif reward == -1:
                disliked_counts[chosen_mid] += 1

            tqdm.write(
                f"step={step} runner={runner} model={model} "
                f"choice_idx={idx} reward={reward}"
            )

        epoch_record = {
            "epoch": epoch,
            "steps_run": steps_run,
            "user_logs": {str(k): v for k, v in user_logs.items()},
        }
        with open(epochs_path, "a") as f:
            f.write(json.dumps(epoch_record) + "\n")
            f.flush()
            os.fsync(f.fileno())

        total_users += len(epoch_record["user_logs"])
        total_steps += steps_run

    print(
        f"Wrote {num_epochs - n_completed} new epoch(s) "
        f"({num_epochs} total), {total_steps} new steps, "
        f"{total_users} new user-runs -> {out_dir}"
    )


if __name__ == "__main__":
    main()
