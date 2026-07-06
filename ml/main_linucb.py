"""Classical contextual-bandit runner (LinUCB / Linear Thompson Sampling).

A non-LLM baseline that learns online from revealed rewards and emits the SAME
epochs.jsonl shape as main_sliding.py, so viz/plot_ctr.py / viz/plot_cum_regret.py /
viz/plot_regime_ctr.py consume its runs unchanged.

It reuses main_sliding.get_dataset, so with a matching `seed` it sees the exact
same sampled users / impression order as a matching LLM run. Each user gets
its own model, warm-started from that user's 5 cold-start ratings; only the chosen
arm's reward is revealed and folded back (true bandit feedback, not full labels).

    uv run python bandit.py --config configs/linucb.json
    uv run python bandit.py --config configs/linucb.json --mode nonstationary
"""
import argparse
import json
import os
import random
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils.bandits import make_model
from utils.features import PHI_DIM, MovieStore, UserState
from main_sliding import get_dataset

REQUIRED_CFG_KEYS = {
    "algo", "model_name", "num_epochs", "num_users", "num_steps", "seed", "output",
}
VALID_ALGOS = {"linucb", "lints"}
RESUME_MATCH_KEYS = (
    "algo", "num_epochs", "num_users", "num_steps", "seed",
    "alpha", "lambda", "use_embeddings", "embed_model", "v",
)
DEFAULTS = {
    "alpha": 1.0,
    "lambda": 1.0,
    "v": 1.0,
    "use_embeddings": True,
    "embed_model": "all-MiniLM-L6-v2",
    "model": None,  # plot label; defaults to algo
}


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
                        "each user's 30th interaction")
    return p.parse_args()


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    missing = REQUIRED_CFG_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"config {path} missing keys: {sorted(missing)}")
    if cfg["algo"] not in VALID_ALGOS:
        raise ValueError(
            f"unknown algo: {cfg['algo']!r} (valid: {sorted(VALID_ALGOS)})"
        )
    for k, default in DEFAULTS.items():
        cfg.setdefault(k, default)
    if cfg["model"] is None:
        cfg["model"] = cfg["algo"]
    cfg["output"] = Path(cfg["output"])
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config)
    algo           = cfg["algo"]
    num_epochs     = cfg["num_epochs"]
    num_users      = cfg["num_users"]
    num_steps      = cfg["num_steps"]
    seed           = cfg["seed"]
    output         = cfg["output"]
    model_name     = cfg["model_name"]
    alpha          = cfg["alpha"]
    lam            = cfg["lambda"]
    v              = cfg["v"]
    use_embeddings = cfg["use_embeddings"]
    embed_model    = cfg["embed_model"]
    model_label    = cfg["model"]
    mode           = args.mode

    random.seed(seed)
    rng = np.random.default_rng(seed)
    store = MovieStore.from_metadata(use_embeddings, embed_model)

    config_record = {
        "model_name": model_name,
        "model": model_label,
        "runner": "bandit",
        "algo": algo,
        "num_epochs": num_epochs,
        "num_users": num_users,
        "num_steps": num_steps,
        "seed": seed,
        "alpha": alpha,
        "lambda": lam,
        "v": v,
        "use_embeddings": use_embeddings,
        "embed_model": embed_model,
        "mode": mode,
        "output": str(output),
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
    total_clicks = 0

    for epoch in range(num_epochs):
        cold_starts_df, imprs_df, _ = get_dataset(num_users)
        user_queues: dict[int, deque] = {}
        for impr_id, uid in imprs_df["userId"].items():
            user_queues.setdefault(int(uid), deque()).append(int(impr_id))

        if epoch < n_completed:
            tqdm.write(f"resume: skipping completed epoch {epoch}")
            continue

        user_logs: dict[int, dict] = {}
        user_state: dict[int, UserState] = {}
        user_model: dict = {}
        steps_run = 0
        epoch_clicks = 0

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

            if mode == "nonstationary" and step >= num_steps // 2:
                labels = labels[1:] + labels[:1]

            if uid not in user_logs:
                cs = cold_starts_df.loc[uid]
                state = UserState(store)
                model = make_model(algo, PHI_DIM, alpha, lam, v, rng)
                for m, l in zip(cs.impression, cs.labels):
                    m, l = int(m), int(l)
                    model.update(state.phi(m), float(l))  # phi before observe
                    state.observe(m, l)
                user_state[uid] = state
                user_model[uid] = model
                user_logs[uid] = {
                    "cold_start_mids": [int(m) for m in cs.impression],
                    "cold_start_labels": [int(r) for r in cs.labels],
                    "interactions": [],
                }

            state = user_state[uid]
            model = user_model[uid]
            phis = [state.phi(m) for m in cands]
            scores = model.scores(phis)
            idx = int(np.argmax(scores))
            chosen_mid = cands[idx]
            reward = labels[idx]

            model.update(phis[idx], float(reward))
            state.observe(chosen_mid, reward)

            user_logs[uid]["interactions"].append({
                "step": step,
                "impression_id": int(impr_id),
                "candidates": cands,
                "labels": labels,
                "choice_idx": idx,
                "chosen_mid": chosen_mid,
                "reward": reward,
                "scores": [round(float(s), 4) for s in scores],
            })
            steps_run = step + 1
            epoch_clicks += int(reward == 1)

        epoch_record = {
            "epoch": epoch,
            "steps_run": steps_run,
            "user_logs": {str(k): v for k, v in user_logs.items()},
        }
        with open(epochs_path, "a") as f:
            f.write(json.dumps(epoch_record) + "\n")
            f.flush()
            os.fsync(f.fileno())

        ctr = epoch_clicks / steps_run if steps_run else 0.0
        tqdm.write(f"epoch {epoch}: steps={steps_run} clicks={epoch_clicks} "
                   f"CTR={ctr:.4f}")
        total_users += len(epoch_record["user_logs"])
        total_steps += steps_run
        total_clicks += epoch_clicks

    overall = total_clicks / total_steps if total_steps else 0.0
    print(f"Wrote {num_epochs - n_completed} new epoch(s) ({num_epochs} total), "
          f"{total_steps} new steps, {total_users} new user-runs, "
          f"overall CTR={overall:.4f} -> {out_dir}")


if __name__ == "__main__":
    main()
