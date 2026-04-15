import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(
    results_by_model: dict,
    save_path: str,
    temp_label: str = "random",
    title_prefix: str = "",
):
    """Plot cumulative average reward (CTR) over time for multiple models on one graph."""
    colors = [
        "steelblue",
        "darkorange",
        "seagreen",
        "crimson",
        "mediumpurple",
        "goldenrod",
        "deeppink",
        "teal",
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    num_users = None

    for idx, (model_name, all_results) in enumerate(results_by_model.items()):
        color = colors[idx % len(colors)]
        num_users = len(all_results)

        max_rounds = max(len(r["rewards"]) for r in all_results)
        reward_matrix = np.full((len(all_results), max_rounds), np.nan)

        for i, r in enumerate(all_results):
            rewards = r["rewards"]
            reward_matrix[i, : len(rewards)] = rewards

        num_users_per_step = np.sum(~np.isnan(reward_matrix), axis=0)
        cum_reward_matrix = np.nancumsum(reward_matrix, axis=1)
        timesteps = np.arange(1, max_rounds + 1)
        cum_avg_matrix = cum_reward_matrix / timesteps[np.newaxis, :]
        avg_cum_reward = np.nanmean(cum_avg_matrix, axis=0)
        se_cum = np.nanstd(cum_avg_matrix, axis=0) / np.sqrt(num_users_per_step)

        ax.plot(timesteps, avg_cum_reward, color=color, linewidth=1.5, label=model_name)
        ax.fill_between(
            timesteps,
            avg_cum_reward - se_cum,
            avg_cum_reward + se_cum,
            alpha=0.2,
            color=color,
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Average Reward (CTR)")
    title = f"Cumulative Average Reward | {num_users} users | temp={temp_label}"
    if title_prefix:
        title = f"{title_prefix} — {title}"
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {save_path}")
