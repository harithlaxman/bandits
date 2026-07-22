"""Classic reference agents: UCB, Greedy, Thompson Sampling.

Small, self-contained, no forgetting (stationary case only). Each exposes the same
decide -> pull -> observe interface: ``act()`` returns an arm index, ``update(arm, reward)``
folds in the observed Bernoulli reward. These run only as reference baselines against the
SLM; their internals are never surfaced to the model in the prompt.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np


class UCBAgent:
    """alpha-UCB. Pulls every arm once, then picks argmax(mean + exploration bonus)."""

    name = "UCB"

    def __init__(self, num_arms: int, alpha: float = 2, seed: Optional[int] = None):
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
        return self.alpha * math.sqrt(math.log(total) / self.counts[arm])

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


class DiscountedUCBAgent(UCBAgent):
    """D-UCB. Discount every arm's counts and rewards by ``gamma`` so old evidence decays geometrically;
    ``exploration_bonus`` then reads off the discounted total. This lets the agent track a moving optimum
    (non-stationary case).
    """

    name = "D-UCB"

    def __init__(self, num_arms: int, alpha: float = 2, gamma: float = 0.99,
                 seed: Optional[int] = None):
        self.gamma = gamma
        super().__init__(num_arms, alpha, seed)

    def update(self, arm: int, reward: float):
        for a in range(self.k):
            self.counts[a] *= self.gamma
            self.rewards[a] *= self.gamma
        self.counts[arm] += 1
        self.rewards[arm] += reward


class SlidingWindowUCBAgent(UCBAgent):
    """SW-UCB. Only the last ``window`` (arm, reward) pairs count toward each arm's mean and
    exploration bonus; older observations are dropped. Tracks a moving optimum by forgetting
    stale evidence. Simplified variant of Garivier & Moulines (2011); same bonus form as
    ``UCBAgent`` but over the windowed counts.

    When an arm has not been pulled within the window its count returns to 0, so ``act()``
    re-triggers forced exploration of it — the intended re-exploration under drift.
    """

    name = "SW-UCB"

    def __init__(self, num_arms: int, alpha: float = 0.5, window: int = 200,
                 seed: Optional[int] = None):
        self.window = window
        super().__init__(num_arms, alpha, seed)

    def reset(self):
        super().reset()
        self.buffer: deque = deque()

    def update(self, arm: int, reward: float):
        self.buffer.append((arm, reward))
        self.counts[arm] += 1
        self.rewards[arm] += reward
        if len(self.buffer) > self.window:
            old_arm, old_reward = self.buffer.popleft()
            self.counts[old_arm] -= 1
            self.rewards[old_arm] -= old_reward


class LinUCBAgent:
    """Disjoint LinUCB (Li et al. 2010): one ridge model per arm over a shared context.
    Each arm keeps A^{-1} directly via Sherman-Morrison rank-1 updates, so scoring and
    updating are O(d^2) with no per-step matrix inverse. Contextual variant of the
    ``act``/``update`` interface: both take the current context vector."""

    name = "LinUCB"

    def __init__(self, num_arms: int, dim: int, alpha: float = 1.0, lam: float = 1.0,
                 seed: Optional[int] = None):
        self.k = num_arms
        self.d = dim
        self.alpha = alpha
        self.lam = lam
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.Ainv = [np.eye(self.d) / self.lam for _ in range(self.k)]
        self.b = [np.zeros(self.d) for _ in range(self.k)]

    def _ucb(self, arm: int, x: np.ndarray) -> float:
        Ainv = self.Ainv[arm]
        mean = float((Ainv @ self.b[arm]) @ x)
        var = float(x @ Ainv @ x)
        return mean + self.alpha * math.sqrt(max(var, 0.0))

    def act(self, context: np.ndarray) -> int:
        scores = np.array([self._ucb(a, context) for a in range(self.k)])
        best = np.flatnonzero(scores == scores.max())
        return int(self.rng.choice(best))

    def update(self, context: np.ndarray, arm: int, reward: float):
        Ainv = self.Ainv[arm]
        Ax = Ainv @ context
        Ainv -= np.outer(Ax, Ax) / (1.0 + float(context @ Ax))
        self.b[arm] += reward * context


class SlidingWindowLinUCBAgent:
    """Sliding-window disjoint LinUCB for non-stationary contexts. Each arm keeps a ridge
    model built from only the most recent ``window`` interactions (globally, across all
    arms — same window semantics as ``SlidingWindowUCBAgent``); older (context, reward)
    pairs are subtracted out as they leave the window, so the estimate forgets a stale
    context->reward mapping and tracks the current one. Disjoint (per-arm independent),
    the standard sliding-window LinUCB form; hybrid's shared-model cross terms don't
    decompose per observation, so removal isn't well defined there.

    Contexts here are binary, so each A += xx' is later cancelled by an exact A -= xx'
    with no floating-point drift over the horizon."""

    name = "SW-LinUCB"

    def __init__(self, num_arms: int, dim: int, alpha: float = 1.0, lam: float = 1.0,
                 window: int = 300, seed: Optional[int] = None):
        self.k = num_arms
        self.d = dim
        self.alpha = alpha
        self.lam = lam
        self.window = window
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.A = [self.lam * np.eye(self.d) for _ in range(self.k)]
        self.b = [np.zeros(self.d) for _ in range(self.k)]
        self.buffer: deque = deque()  # (arm, context, reward), most recent last

    def _ucb(self, arm: int, x: np.ndarray) -> float:
        Ainv = np.linalg.inv(self.A[arm])
        mean = float((Ainv @ self.b[arm]) @ x)
        var = float(x @ Ainv @ x)
        return mean + self.alpha * math.sqrt(max(var, 0.0))

    def act(self, context: np.ndarray) -> int:
        scores = np.array([self._ucb(a, context) for a in range(self.k)])
        best = np.flatnonzero(scores == scores.max())
        return int(self.rng.choice(best))

    def update(self, context: np.ndarray, arm: int, reward: float):
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        self.buffer.append((arm, context, reward))
        if len(self.buffer) > self.window:
            old_arm, old_x, old_r = self.buffer.popleft()
            self.A[old_arm] -= np.outer(old_x, old_x)
            self.b[old_arm] -= old_r * old_x


AGENTS = {
    "ucb": UCBAgent,
    "greedy": GreedyAgent,
    "thompson": ThompsonSamplingAgent,
    "ducb": DiscountedUCBAgent,
    "swucb": SlidingWindowUCBAgent,
}

# Context-aware agents; separate registry because act/update take the context vector.
CONTEXTUAL_AGENTS = {
    "linucb": LinUCBAgent,
    "sw_linucb": SlidingWindowLinUCBAgent,
}
