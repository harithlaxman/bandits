"""Metrics & analysis. Regret is the headline metric and is computed here from scratch
(the original repo only ever plotted average reward).

Everything operates on a list of trajectories, where each trajectory is a list of
per-step dicts with at least ``reward``, ``regret`` and ``is_parse_failure`` keys
(exactly the JSON produced by ``run.py``).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def _matrix(trajs: List[List[dict]], key: str) -> np.ndarray:
    """Stack a per-step field into a (n_trajs, horizon) array."""
    return np.array([[step[key] for step in traj] for traj in trajs], dtype=float)


def _mean_se(mat: np.ndarray):
    mean = mat.mean(axis=0)
    se = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0]) if mat.shape[0] > 1 else np.zeros(mat.shape[1])
    return mean, se


def average_reward_curve(trajs: List[List[dict]]):
    """Running average reward across steps, averaged over trajectories (mean, stderr)."""
    rewards = _matrix(trajs, "reward")
    running = np.cumsum(rewards, axis=1) / np.arange(1, rewards.shape[1] + 1)
    return _mean_se(running)


def cumulative_regret_curve(trajs: List[List[dict]]):
    """Cumulative regret vs. the (possibly moving) optimum (mean, stderr)."""
    cum = np.cumsum(_matrix(trajs, "regret"), axis=1)
    return _mean_se(cum)


def instantaneous_regret_curve(trajs: List[List[dict]], window: int = 10):
    """Windowed mean per-step regret — preferred under drift, where a running average hides
    change-points and moving optima."""
    regret = _matrix(trajs, "regret")
    horizon = regret.shape[1]
    smoothed = np.array([
        regret[:, max(0, t - window + 1):t + 1].mean(axis=1) for t in range(horizon)
    ]).T
    return _mean_se(smoothed)


def _optimal_matrix(trajs: List[List[dict]]) -> np.ndarray:
    """(n_trajs, horizon) boolean array: True where the optimal arm was picked.
    Uses regret==0 so it applies to LLM and baseline trajectories alike (baseline
    per-step dicts don't carry ``best_action``)."""
    return np.isclose(_matrix(trajs, "regret"), 0.0)


def optimal_action_rate_curve(trajs: List[List[dict]]):
    """Fraction of trajectories picking the optimal arm at each step (mean, stderr)."""
    return _mean_se(_optimal_matrix(trajs).astype(float))


def parse_failure_rate(trajs: List[List[dict]]) -> float:
    fails = _matrix(trajs, "is_parse_failure")
    return float(fails.mean())


def summarize(trajs: List[List[dict]]) -> Dict[str, float]:
    reward_mean, _ = average_reward_curve(trajs)
    regret_mean, regret_se = cumulative_regret_curve(trajs)
    return {
        "n_trajectories": len(trajs),
        "horizon": len(trajs[0]) if trajs else 0,
        "final_avg_reward": float(reward_mean[-1]),
        "final_cumulative_regret": float(regret_mean[-1]),
        "final_cumulative_regret_se": float(regret_se[-1]),
        "parse_failure_rate": parse_failure_rate(trajs),
    }
