import json

import matplotlib.pyplot as plt
import numpy as np

# ─── Config ─────────────────────────────────────────────────────────
# Edit the files/colors dicts and save_path below, then run:
#     python plot_comparison.py

files = {
    # ─── ml ──────────────────────────────────────────────────────
    # "granite4_tiny-h": "ml/results/granite4_tiny-h.json",
    # "granite4_small-h": "ml/results/granite4_small-h.json",
    # "ministral-3_8b": "ml/results/ministral-3_8b.json",
    # "ministral-3_14b": "ml/results/ministral-3_14b.json",
    # "gemma4_e4b": "ml/results/gemma4_e4b.json",
    # "llama3.1_latest": "ml/results/llama3.1_latest.json",
    # ─── bl ──────────────────────────────────────────────────────
    "granite4-tiny-h": "bl/results/cot/granite-4.0-h-tiny.json",
    "granite4-small-h": "bl/results/cot/granite-4.0-h-small.json",
}

colors = {
    # ─── ml ──────────────────────────────────────────────────────
    "llama3.1_latest": "#F7CAC9",
    "gemma4_e4b": "#92A8D1",
    "ministral-3_8b": "#955251",
    "ministral-3_14b": "#B565A7",
    "granite4_tiny-h": "#009B77",
    "granite4_small-h": "#E8853D",
    # ─── bl ──────────────────────────────────────────────────────
    "granite4-tiny-h": "#B565A7",
    "granite4-small-h": "#009B77",
}

save_path = "bl/results/cot_temps.png"
title = "LLM Bandit Comparison"


# ─── Plot ───────────────────────────────────────────────────────────
def _build_reward_matrix(data):
    max_rounds = max(len(r["rewards"]) for r in data)
    reward_matrix = np.full((len(data), max_rounds), np.nan)
    for i, r in enumerate(data):
        rw = r["rewards"]
        reward_matrix[i, : len(rw)] = rw
    return reward_matrix, max_rounds


results = {}
for name, path in files.items():
    with open(path) as f:
        results[name] = json.load(f)

_, ax = plt.subplots(figsize=(8, 5))

for name, data in results.items():
    reward_matrix, max_rounds = _build_reward_matrix(data)
    timesteps = np.arange(1, max_rounds + 1)
    n_users = np.sum(~np.isnan(reward_matrix), axis=0)

    cum_reward = np.nancumsum(reward_matrix, axis=1)
    cum_avg = cum_reward / timesteps[np.newaxis, :]
    avg_cum = np.nanmean(cum_avg, axis=0)
    se_cum = np.nanstd(cum_avg, axis=0) / np.sqrt(n_users)

    ax.plot(timesteps, avg_cum, color=colors[name], linewidth=1.5, label=name)
    ax.fill_between(
        timesteps,
        avg_cum - se_cum,
        avg_cum + se_cum,
        alpha=0.15,
        color=colors[name],
    )

ax.set_xlabel("Round")
ax.set_ylabel("Cumulative Average Reward (CTR)")
ax.set_title("Cumulative CTR Over Time")
ax.legend()
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)

n_users_total = len(next(iter(results.values())))
plt.suptitle(f"{title} | {n_users_total} users | 50 rounds", fontsize=13)
plt.tight_layout()

plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved to: {save_path}")

for name, data in results.items():
    ctrs = [r["ctr"] for r in data]
    print(
        f"{name:20s}  avg CTR: {np.mean(ctrs):.4f} +/- {np.std(ctrs) / np.sqrt(len(ctrs)):.4f}"
    )
