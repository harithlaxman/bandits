"""Classic reference agents: UCB, Greedy, Thompson Sampling.

Small, self-contained, no forgetting (stationary case only). Each exposes the same
decide -> pull -> observe interface: ``act()`` returns an arm index, ``update(arm, reward)``
folds in the observed Bernoulli reward. These run only as reference baselines against the
SLM; their internals are never surfaced to the model in the prompt.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


class UCBAgent:
    """alpha-UCB. Pulls every arm once, then picks argmax(mean + exploration bonus)."""

    name = "UCB"

    def __init__(self, num_arms: int, alpha: float = 0.5, seed: Optional[int] = None):
        self.k = num_arms
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.counts = [0] * self.k
        self.rewards = [0.0] * self.k

    def exploitation_value(self, arm: int) -> float:
        return self.rewards[arm] / self.counts[arm] if self.counts[arm] > 0 else 0.0

    def exploration_bonus(self, arm: int) -> float:
        total = sum(self.counts)
        return math.sqrt((self.alpha * math.log(total)) / self.counts[arm])

    def _value(self, arm: int) -> float:
        return self.exploitation_value(arm) + self.exploration_bonus(arm)

    def act(self) -> int:
        for arm in range(self.k):
            if self.counts[arm] == 0:
                return arm
        return int(np.argmax([self._value(a) for a in range(self.k)]))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.rewards[arm] += reward


class GreedyAgent(UCBAgent):
    """UCB without the exploration bonus (pure exploitation after one pull each)."""

    name = "Greedy"

    def _value(self, arm: int) -> float:
        return self.exploitation_value(arm)


class ThompsonSamplingAgent:
    """Beta-Bernoulli Thompson Sampling with a uniform (Beta(1,1)) prior."""

    name = "ThompsonSampling"

    def __init__(self, num_arms: int, alpha_prior: float = 1.0, beta_prior: float = 1.0,
                 seed: Optional[int] = None):
        self.k = num_arms
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.alpha = [self.alpha_prior] * self.k
        self.beta = [self.beta_prior] * self.k

    def act(self) -> int:
        samples = [self.rng.beta(self.alpha[a], self.beta[a]) for a in range(self.k)]
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward


AGENTS = {
    "ucb": UCBAgent,
    "greedy": GreedyAgent,
    "thompson": ThompsonSamplingAgent,
}
