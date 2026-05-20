import argparse
import ast
import json
import os
import random
import re
import time
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


REQUIRED_CFG_KEYS = {
    "model", "runner", "num_epochs", "num_users",
    "num_steps", "seed", "temperature", "output",
}
VALID_RUNNERS = {"ollama", "vllm", "openai", "random"}
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
                   help="which impressions pkl to use; nonstationary preserves "
                        "per-user pre->post order so the regime change is observable")
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


def get_dataset(num_users: int, mode: str = "stationary"):
    df = pd.read_pickle(DATA_DIR / f"impressions_{mode}.pkl")
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


def get_history_prompt(interactions: list[dict], mid_to_data) -> str:
    prompt = "YOUR PAST RECOMMENDATIONS:\n"
    for it in interactions:
        if it["chosen_mid"] is None:
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
        cold_starts_df, imprs_df, mid_to_data = get_dataset(num_users, args.mode)
        impression_ids = imprs_df.index.to_list()
        if args.mode != "nonstationary":
            random.shuffle(impression_ids)

        if epoch < n_completed:
            tqdm.write(f"resume: skipping completed epoch {epoch}")
            continue

        user_logs: dict[int, dict] = {}
        steps_run = 0

        for step in tqdm(range(num_steps), desc=f"epoch {epoch}"):
            if step >= len(impression_ids):
                tqdm.write("No more impressions in dataset")
                break

            impr_id = impression_ids[step]
            row = imprs_df.loc[impr_id]
            uid = int(row["userId"])
            cands = [int(m) for m in row["impression"]]
            labels = [int(r) for r in row["labels"]]

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
                user_logs[uid]["cold_start_prompt"]
                + get_history_prompt(user_logs[uid]["interactions"], mid_to_data)
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
