import argparse
import ast
import json
import os
import random
import re
import time
from collections import deque
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
MAX_CONTEXT = 10

BANDIT_PREAMBLE = """You are a recommendation agent acting as a contextual bandit. \
Each round you are shown a user's taste profile and a set of candidate movies, and \
you pick exactly ONE candidate to recommend; you then learn whether the user LIKED \
it. Your objective is to MAXIMIZE the total number of liked recommendations over \
several rounds. To do this, internally consider two competing pressures:
- EXPLOITATION: recommend movies you are confident this user will like.
- EXPLORATION: recommend movies whose appeal is uncertain, to learn \
tastes you cannot yet predict. This pays off most when you know little about the \
user.

Weigh these two strategies internally, but do NOT reason out loud or explain your \
thinking. Respond with only your final choice.

"""


REQUIRED_CFG_KEYS = {
    "model", "model_name", "runner", "num_epochs", "num_users",
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
    return cfg


def build_temperature_schedule(temperature, num_steps: int, mode: str) -> list[float]:
    """Per-step temperature schedule of length num_steps.

    A numeric `temperature` is constant. The strings "increasing"/"decreasing"
    ramp linearly between 0.0 and 2.0 over the run; in nonstationary mode the
    ramp restarts at the midpoint so exploration ramps up again after the regime
    change.
    """
    if isinstance(temperature, (int, float)):
        return [float(temperature)] * num_steps

    if temperature not in ("increasing", "decreasing"):
        raise ValueError(
            f"temperature must be a number or one of "
            f"['increasing', 'decreasing'], got {temperature!r}"
        )

    def _ramp(length: int) -> list[float]:
        if length <= 1:
            return [0.0] * length
        ramp = [2.0 * i / (length - 1) for i in range(length)]
        return ramp if temperature == "increasing" else ramp[::-1]

    if mode == "nonstationary":
        mid = num_steps // 2
        return _ramp(mid) + _ramp(num_steps - mid)
    return _ramp(num_steps)


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


def get_score_prompt(interactions: list[dict]) -> str:
    made = [it for it in interactions if it["chosen_mid"] is not None]
    if not made:
        return ""
    liked = sum(1 for it in made if it["reward"] == 1)
    return f"SCORE SO FAR: {liked} liked out of {len(made)} recommendations.\n\n"


def is_displayed_interaction(it: dict, positive_only: bool) -> bool:
    """Whether an interaction renders in the history prompt (kept in lockstep
    with get_history_prompt's skip rules)."""
    if it["chosen_mid"] is None:
        return False
    if positive_only and it["reward"] != 1:
        return False
    return True


def select_context_window(
    cs_mids: List[int],
    cs_labels: List[int],
    interactions: list[dict],
    positive_only: bool,
    max_context: int = MAX_CONTEXT,
):
    """Slide a window of at most `max_context` *displayed* items over the
    chronological sequence [cold-start..., recommendations...]. Cold-start
    movies always count as displayed; recommendations count only when they
    render (see is_displayed_interaction). Returns the windowed cold-start
    mids/labels and the windowed interactions slice.
    """
    events = [("cs", i) for i in range(len(cs_mids))]
    events += [("int", j) for j in range(len(interactions))]

    def is_displayed(ev) -> bool:
        kind, i = ev
        if kind == "cs":
            return True
        return is_displayed_interaction(interactions[i], positive_only)

    # walk backward from the most recent event, collecting displayed items
    # until the window holds `max_context` of them (or the sequence runs out)
    start = 0
    count = 0
    for idx in range(len(events) - 1, -1, -1):
        if is_displayed(events[idx]):
            count += 1
            if count == max_context:
                start = idx
                break

    window = events[start:]
    cs_idx = [i for kind, i in window if kind == "cs"]
    int_idx = [i for kind, i in window if kind == "int"]

    if cs_idx:
        # any surviving cold-start movie => every later recommendation survives too
        cs_start = cs_idx[0]
        return cs_mids[cs_start:], cs_labels[cs_start:], interactions
    int_start = int_idx[0] if int_idx else len(interactions)
    return [], [], interactions[int_start:]


def get_candidates_prompt(mids: List[int], mid_to_data) -> str:
    prompt = "CANDIDATES (choose ONE):\n"
    for i, mid in enumerate(mids, 1):
        prompt += f"{i}.\n{mid_to_data[mid]}\n"
    prompt += (
        "Respond with ONLY the number of your choice and nothing else. "
        "Do not explain your reasoning or add any other text.\n"
    )
    return prompt


def parse_choice(text: str, n: int = N_CHOICES) -> int | None:
    m = re.match(r"\s*([1-9])\b", text)
    if not m:
        return None
    idx = int(m.group(1)) - 1
    return idx if 0 <= idx < n else None


def make_get_response(
    runner: str, model: str, temperature: float, model_type: str,
    system: str | None = None, top_p: float = 1.0, top_k: int = -1,
):
    if runner == "ollama":
        from utils.ollama_runner import generate

        def _ollama_call(prompt: str, temperature: float) -> str:
            return generate(model, prompt, system=system, temperature=temperature)

        return _ollama_call

    if runner == "vllm":
        from utils.vllm_runner import generate as vllm_generate

        vllm_choices = [str(i) for i in range(1, N_CHOICES + 1)]

        def _vllm_call(prompt: str, temperature: float) -> str:
            return vllm_generate(
                model, prompt, system=system,
                temperature=temperature, top_p=top_p, top_k=top_k,
                model_type=model_type, choices=vllm_choices,
            )

        return _vllm_call

    if runner == "openai":
        from utils.openai_runner import generate as openai_generate

        def _openai_call(prompt: str, temperature: float) -> str:
            return openai_generate(model, prompt, system=system, temperature=temperature)

        return _openai_call

    if runner == "random":
        sys_rng = random.SystemRandom()

        def _random_call(_prompt: str, _temperature: float) -> str:
            return str(sys_rng.randint(1, N_CHOICES))

        return _random_call

    raise ValueError(f"unknown runner: {runner}")


def get_choice(
    get_response, prompt: str, temperature: float
) -> tuple[int | None, str, float]:
    total_ms = 0.0
    last_resp = ""
    for _ in range(MAX_PARSE_RETRIES):
        t0 = time.perf_counter()
        try:
            last_resp = get_response(prompt, temperature)
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
    top_p       = cfg.get("top_p", 1.0)
    top_k       = cfg.get("top_k", -1)
    output      = cfg["output"]
    model_name  = cfg["model_name"]
    model_type  = cfg.get("model_type", "instruct")
    positive_only = args.positive_only

    random.seed(seed)
    temp_schedule = build_temperature_schedule(temperature, num_steps, args.mode)
    get_response = make_get_response(
        runner, model, temperature, model_type, BANDIT_PREAMBLE,
        top_p=top_p, top_k=top_k,
    )

    config_record = {
        "model_name": model_name,
        "model": model,
        "runner": runner,
        "num_epochs": num_epochs,
        "num_users": num_users,
        "num_steps": num_steps,
        "seed": seed,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "output": str(output),
        "max_parse_retries": MAX_PARSE_RETRIES,
        "max_context": MAX_CONTEXT,
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
        timestamp = datetime.now().strftime("%m%d_%H%M")
        out_dir = output / f"{model_name}_{timestamp}"
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

            cs_mids_w, cs_labels_w, interactions_w = select_context_window(
                user_logs[uid]["cold_start_mids"],
                user_logs[uid]["cold_start_labels"],
                user_logs[uid]["interactions"],
                positive_only,
            )
            prompt = ""
            if cs_mids_w:
                prompt += get_coldstart_prompt(cs_mids_w, cs_labels_w, mid_to_data)
            prompt += get_history_prompt(interactions_w, mid_to_data, positive_only)
            prompt += get_score_prompt(user_logs[uid]["interactions"])
            prompt += get_candidates_prompt(cands, mid_to_data)

            temp = temp_schedule[step]
            idx, raw, ms = get_choice(get_response, prompt, temp)
            chosen_mid = cands[idx] if idx is not None else None
            reward = labels[idx] if idx is not None else None

            user_logs[uid]["interactions"].append({
                "step": step,
                "impression_id": int(impr_id),
                "candidates": cands,
                "labels": labels,
                "temperature": temp,
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
