"""Bernoulli Multi-Armed Bandit (button-pushing scenario) for in-context RL with SLMs.

This is a lean re-implementation of the EVOLvE / banditbench MAB "button" scenario.
Only the stationary case is implemented here, but the environment is written so that
every "what is optimal / expected right now" question is routed through a method that
takes the current ``step``. Those ``step`` arguments are currently unused (means are
fixed), and are the single seam to extend to non-stationary means later without
touching the agents, prompting, or logging code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Button-pushing scenario (verbatim from the design doc / original repo)
# ---------------------------------------------------------------------------

# The first `num_arms` names are used for an instance. Ordered list of 100.
COLOR_NAMES: List[str] = [
    "blue", "green", "red", "yellow", "orange", "purple", "pink", "brown", "black",
    "white", "gray", "cyan", "magenta", "maroon", "olive", "lime", "teal", "navy",
    "indigo", "violet", "gold", "silver", "bronze", "coral", "turquoise", "lavender",
    "tan", "beige", "ivory", "aqua", "azure", "crimson", "fuchsia", "khaki", "peach",
    "plum", "sienna", "mint", "rose", "ruby", "salmon", "scarlet", "amber", "emerald",
    "jade", "mauve", "ochre", "periwinkle", "sapphire", "topaz", "chartreuse",
    "burgundy", "mustard", "cinnamon", "cobalt", "mint green", "forest green",
    "sky blue", "lilac", "tangerine", "raspberry", "pumpkin", "blush", "eggplant",
    "seafoam", "spruce", "lemon", "denim", "flamingo", "sand", "apricot", "honey",
    "chestnut", "midnight blue", "moss green", "bordeaux", "lavender blush",
    "slate gray", "gunmetal", "mint cream", "dark salmon", "persimmon", "cranberry",
    "wheat", "bistre", "magenta haze", "sepia", "ultramarine", "lime green",
    "steel blue", "blush pink", "pearl", "mulberry", "antique white", "burnt orange",
    "coral pink", "ice blue", "bright yellow", "honeydew", "baby blue",
]

ACTION_UNIT = "button"

# {0} = number of buttons, {1} = "[blue, green, ...]"
BASE_INSTRUCTION = (
    "You are a bandit algorithm with {0} buttons labeled {1}.\n"
    "Each button is associated with a Bernoulli distribution with a fixed but unknown mean; the means for the two buttons could be different.\n"
    "When you press a button, you will get a reward that is sampled from the button's associated distribution. Your goal is to maximize the total reward."
)

# NOTE: the stray space in "total reward ." is copied verbatim from the original.
DETAILED_INSTRUCTION = (
    "You are a bandit algorithm with {0} buttons labeled {1}.\n"
    "Each button is associated with a Bernoulli distribution with a fixed but unknown mean; the means for the two buttons could be different.\n"
    "For either button, when you press it, you will get a reward that is sampled from the button's associated distribution. Your goal is to maximize the total reward .\n\n"
    "A good strategy to optimize for reward in these situations requires balancing exploration"
    "and exploitation. You need to explore to try out all of the buttons and find those with high rewards, but you also have to exploit the information that you have to"
    "accumulate rewards."
)

HISTORY_PREAMBLE = "So far you have interacted {n} times with the following choices and rewards:\n"

QUERY_TEMPLATE = (
    "\n\nWhich {unit} will you choose next? PLEASE RESPOND ONLY WITH {choices} AND NO TEXT EXPLANATION."
)


def action_names(num_arms: int) -> List[str]:
    """First `num_arms` button (color) names, in fixed order."""
    assert num_arms <= len(COLOR_NAMES), f"At most {len(COLOR_NAMES)} arms supported."
    return COLOR_NAMES[:num_arms]


def task_instruction(num_arms: int, instruction_type: str = "detailed") -> str:
    names = action_names(num_arms)
    template = DETAILED_INSTRUCTION if instruction_type == "detailed" else BASE_INSTRUCTION
    return template.format(num_arms, "[" + ", ".join(names) + "]")


def query_prompt(num_arms: int) -> str:
    names = action_names(num_arms)
    return QUERY_TEMPLATE.format(unit=ACTION_UNIT, choices="[" + ", ".join(names) + "]")


# ---------------------------------------------------------------------------
# Instance construction
# ---------------------------------------------------------------------------

def make_means(num_arms: int, difficulty: str = "hard") -> List[float]:
    """Build arm means from a target optimality gap (best - second best).

    'hard' -> small gap (0.2), 'easy' -> large gap (0.5). The best arm is last;
    call sites shuffle per-instance to remove positional bias.
    """
    gap = {"hard": 0.2, "easy": 0.5}[difficulty]
    return [0.5 - gap / 2] * (num_arms - 1) + [0.5 + gap / 2]


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class Interaction:
    step: int
    raw_response: str        # exact text the model returned
    action: int              # mapped arm index that was actually pulled
    action_name: str         # button/color name pulled
    reward: float            # observed 0/1 reward
    expected_reward: float   # true mean of the pulled arm (hidden from agent)
    is_parse_failure: bool   # True if response didn't strictly match a button -> random arm
    best_mean: float         # optimal mean at this step (moving optimum ready)
    best_action: int         # optimal arm index at this step
    regret: float            # best_mean - expected_reward (per-step expected regret)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@dataclass
class BernoulliMAB:
    """N-arm Bernoulli bandit. Means are hidden from the agent; the harness knows them
    for scoring. All optimum/expected queries take a ``step`` (unused while stationary)."""

    means: List[float]
    horizon: int
    seed: Optional[int] = None
    history: List[Interaction] = field(default_factory=list)
    h: int = 0

    def __post_init__(self):
        self.means = list(self.means)
        self.num_arms = len(self.means)
        self.rng = np.random.default_rng(self.seed)
        # instance hardness = optimality gap (best - second best), for scoring/labels
        srt = sorted(self.means)
        self.instance_hardness = srt[-1] - srt[-2] if self.num_arms > 1 else srt[-1]

    # --- moving-optimum seam (step currently ignored) ----------------------
    def means_at(self, step: Optional[int] = None) -> List[float]:
        return self.means

    def expected_reward(self, arm: int, step: Optional[int] = None) -> float:
        return self.means_at(step)[arm]

    def optimal_arm(self, step: Optional[int] = None) -> int:
        return int(np.argmax(self.means_at(step)))

    def optimal_mean(self, step: Optional[int] = None) -> float:
        return float(np.max(self.means_at(step)))

    # --- dynamics ----------------------------------------------------------
    def sample_reward(self, arm: int, step: Optional[int] = None) -> float:
        return 1.0 if self.rng.uniform(0.0, 1.0) < self.expected_reward(arm, step) else 0.0

    def reset(self):
        self.history = []
        self.h = 0

    @property
    def done(self) -> bool:
        return self.h >= self.horizon


def parse_action(response: str, names: List[str]) -> Optional[int]:
    """Map a free-text response to an arm. Prefer an exact match; otherwise take the first
    button name that appears anywhere in the response. This tolerates models that wrap the
    answer in brackets or list several names (e.g. ``[blue]``, ``[yellow, orange]``) —
    behaviour the query prompt actively invites by showing the choices as
    ``[blue, green, ...]``. Returns the arm index, or None if no button name is present."""
    norm = response.strip().lower()
    lowered = [n.strip().lower() for n in names]
    # fast path: response is exactly a button name
    try:
        return lowered.index(norm)
    except ValueError:
        pass
    # otherwise: the earliest-occurring button name (as a whole word) wins
    best_idx, best_pos = None, len(norm) + 1
    for i, n in enumerate(lowered):
        m = re.search(rf"\b{re.escape(n)}\b", norm)
        if m and m.start() < best_pos:
            best_idx, best_pos = i, m.start()
    return best_idx


class VerbalMAB:
    """Wraps a core BernoulliMAB with the button scenario: maps free-text responses to
    arms (strict match, random on failure) and records a per-step Interaction."""

    def __init__(self, core: BernoulliMAB, instruction_type: str = "detailed"):
        self.core = core
        self.num_arms = core.num_arms
        self.names = action_names(core.num_arms)
        self.instruction_type = instruction_type

    # prompt pieces
    def task_instruction(self) -> str:
        return task_instruction(self.num_arms, self.instruction_type)

    def query_prompt(self) -> str:
        return query_prompt(self.num_arms)

    def feedback(self, action_name: str, reward: float) -> str:
        return f"{action_name} {ACTION_UNIT}, reward {reward}"

    def step(self, response: str) -> Interaction:
        step = self.core.h
        parsed = parse_action(response, self.names)
        is_fail = parsed is None
        arm = parsed if parsed is not None else int(self.core.rng.integers(0, self.num_arms))

        reward = self.core.sample_reward(arm, step)
        best_arm = self.core.optimal_arm(step)
        best_mean = self.core.optimal_mean(step)
        exp = self.core.expected_reward(arm, step)

        rec = Interaction(
            step=step,
            raw_response=response,
            action=arm,
            action_name=self.names[arm],
            reward=reward,
            expected_reward=exp,
            is_parse_failure=is_fail,
            best_mean=best_mean,
            best_action=best_arm,
            regret=best_mean - exp,
        )
        self.core.history.append(rec)
        self.core.h += 1
        return rec

    def reset(self):
        self.core.reset()
