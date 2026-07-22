"""UCI Forest Covertype as a contextual bandit (Féraud et al., AISTATS 2016 setup).

Each round the agent sees one dataset row's binarized features and picks one of the
7 cover-type classes; reward is 1 iff the pick matches the row's label, so the
optimal expected reward is always 1 and per-step regret is 1 - reward.

Preprocessing follows the paper: each of the 10 continuous variables is recoded by
equal frequencies into 5 binary variables (quintile one-hot); the 44 binary
wilderness/soil columns pass through. Context dim: 94 binary features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from mab import Interaction, action_names, parse_action

CACHE = Path(__file__).parent / "data" / "covertype.npz"
NUM_CONTINUOUS = 10
NUM_BINS = 5
NUM_WILDERNESS = 4


def _binarize(features: np.ndarray) -> np.ndarray:
    """(N, 54) raw features -> (N, 94) binary. Continuous columns come first in the
    raw UCI ordering; a fixed width-5 block per column keeps exactly 5 indicators
    even when heavy ties leave some quintile bins rare."""
    cont, binary = features[:, :NUM_CONTINUOUS], features[:, NUM_CONTINUOUS:]
    blocks = []
    for j in range(NUM_CONTINUOUS):
        col = cont[:, j]
        edges = np.quantile(col, [0.2, 0.4, 0.6, 0.8])
        bins = np.searchsorted(edges, col, side="right")
        onehot = np.zeros((len(col), NUM_BINS), dtype=np.uint8)
        onehot[np.arange(len(col)), bins] = 1
        blocks.append(onehot)
    return np.hstack(blocks + [binary.astype(np.uint8)])


def load_covertype() -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y): X is (581012, 94) uint8, y is (581012,) labels in 0..6.
    Downloads via ucimlrepo on first call, then reads the local cache."""
    if CACHE.exists():
        data = np.load(CACHE)
        return data["X"], data["y"]

    from ucimlrepo import fetch_ucirepo
    raw = fetch_ucirepo(id=31)
    X = _binarize(raw.data.features.to_numpy(dtype=np.float64))
    y = raw.data.targets.to_numpy().ravel().astype(np.int64) - 1  # 1..7 -> 0..6

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, X=X, y=y)
    return X, y


class CovertypeBandit:
    """One epoch's trajectory: ``num_steps`` rows sampled without replacement.

    Non-stationary case: from ``switch_step`` on, every row's correct arm rotates by one
    class ((label + 1) % 7), so the context->action mapping shifts while the marginal
    class frequencies stay identical. The optimal expected reward remains 1 throughout,
    so regret is still 1 - reward."""

    num_arms = 7

    def __init__(self, X: np.ndarray, y: np.ndarray, num_steps: int, seed: int,
                 switch_step: int | None = None):
        rng = np.random.default_rng(seed)
        self.rows = rng.choice(len(X), size=num_steps, replace=False)
        self.X, self.y = X, y
        self.switch_step = switch_step

    def context(self, step: int) -> np.ndarray:
        return self.X[self.rows[step]].astype(np.float64)

    def reward(self, arm: int, step: int) -> float:
        return float(arm == self.label(step))

    def label(self, step: int) -> int:
        lbl = int(self.y[self.rows[step]])
        if self.switch_step is not None and step >= self.switch_step:
            lbl = (lbl + 1) % self.num_arms
        return lbl


# ---------------------------------------------------------------------------
# Verbal layer (LLM scenario): terrain descriptions + color buttons
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "Elevation", "Aspect", "Slope", "Horiz dist to water", "Vert dist to water",
    "Horiz dist to road", "Hillshade 9am", "Hillshade noon", "Hillshade 3pm",
    "Horiz dist to fire point",
]

BIN_LABELS = ["very low", "low", "medium", "high", "very high"]

# {0} = number of buttons, {1} = "[blue, green, ...]". Mirrors DETAILED_INSTRUCTION in
# mab.py, adapted to the contextual (terrain-dependent) reward structure.
COVERTYPE_INSTRUCTION = (
    "You are a bandit algorithm with {0} buttons labeled {1}.\n"
    "Each round you are shown a description of a terrain. Exactly one button gives reward 1 "
    "for that terrain and all others give reward 0; which button is correct depends on the "
    "terrain's features, and the same kind of terrain always has the same correct button. "
    "Your goal is to maximize the total reward.\n\n"
    "A good strategy to optimize for reward in these situations requires balancing exploration "
    "and exploitation. You need to explore to learn which buttons work for which terrains, but "
    "you also have to exploit the information that you have to accumulate rewards."
)


def render_context(x: np.ndarray) -> str:
    """94-bit row -> compact semantic description. Each width-5 block is one-hot, so the
    active bin decodes by argmax; wilderness/soil likewise from their one-hot blocks."""
    parts = []
    for j, name in enumerate(FEATURE_NAMES):
        bin_idx = int(np.argmax(x[j * NUM_BINS:(j + 1) * NUM_BINS]))
        parts.append(f"{name}: {BIN_LABELS[bin_idx]}")
    offset = NUM_CONTINUOUS * NUM_BINS
    wilderness = int(np.argmax(x[offset:offset + NUM_WILDERNESS])) + 1
    soil = int(np.argmax(x[offset + NUM_WILDERNESS:])) + 1
    parts.append(f"Wilderness area {wilderness}")
    parts.append(f"Soil type {soil}")
    return ", ".join(parts)


@dataclass
class ContextualInteraction(Interaction):
    context_text: str = ""   # rendered terrain shown for this step
    row: int = -1            # dataset row index


class VerbalCovertypeBandit:
    """Wraps a CovertypeBandit with the button scenario: maps free-text responses to
    arms (random on parse failure, as in VerbalMAB) and records a per-step
    ContextualInteraction. best_action is the row's true class; the optimal expected
    reward is always 1, so regret = 1 - reward."""

    def __init__(self, core: CovertypeBandit, seed: int):
        self.core = core
        self.num_arms = core.num_arms
        self.names = action_names(core.num_arms)
        self.rng = np.random.default_rng(seed)
        self.history: List[ContextualInteraction] = []
        self.h = 0

    def current_context_text(self) -> str:
        return render_context(self.core.context(self.h))

    def step(self, response: str) -> ContextualInteraction:
        step = self.h
        parsed = parse_action(response, self.names)
        is_fail = parsed is None
        arm = parsed if parsed is not None else int(self.rng.integers(0, self.num_arms))

        reward = self.core.reward(arm, step)
        rec = ContextualInteraction(
            step=step,
            raw_response=response,
            action=arm,
            action_name=self.names[arm],
            reward=reward,
            expected_reward=reward,
            is_parse_failure=is_fail,
            best_mean=1.0,
            best_action=self.core.label(step),
            regret=1.0 - reward,
            context_text=self.current_context_text(),
            row=int(self.core.rows[step]),
        )
        self.history.append(rec)
        self.h += 1
        return rec
