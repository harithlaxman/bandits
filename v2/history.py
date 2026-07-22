"""History representations and full-prompt assembly.

How past interactions are rendered into the prompt is the main lever on SLM behavior:

- RH  (raw):        one line per past pull.
- SH  (summary):    one line per arm (count + average reward).  <- primary experiment.

The model is deliberately given no algorithm-derived hints (e.g. UCB numbers) — it must
work out the explore/exploit tradeoff from the raw counts and averages alone.
"""

from __future__ import annotations

from typing import List

from mab import ACTION_UNIT, HISTORY_PREAMBLE, Interaction, action_names, query_prompt, task_instruction


def _per_arm_stats(history: List[Interaction], num_arms: int):
    counts = [0] * num_arms
    rewards = [0.0] * num_arms
    for exp in history:
        counts[exp.action] += 1
        rewards[exp.action] += exp.reward
    return counts, rewards


def render_raw(history: List[Interaction], names: List[str]) -> str:
    snippet = ""
    for exp in history:
        snippet += f"\n{exp.action_name} {ACTION_UNIT}, reward {exp.reward}"
    return snippet


def render_summary(history: List[Interaction], names: List[str]) -> str:
    if not history:
        return ""
    counts, rewards = _per_arm_stats(history, len(names))
    snippet = ""
    for name, n, total in zip(names, counts, rewards):
        avg = total / (n + 1e-6)
        snippet += f"\n{name} {ACTION_UNIT}, {n} times, average reward {avg:.2f}"
    return snippet


RENDERERS = {
    "RH": render_raw,
    "SH": render_summary,
}


def build_prompt(history: List[Interaction], num_arms: int,
                 history_type: str = "SH", instruction_type: str = "detailed") -> str:
    """Assemble the single flat user message: task instruction + history + decision query."""
    names = action_names(num_arms)
    instruction = task_instruction(num_arms, instruction_type)
    preamble = HISTORY_PREAMBLE.format(n=len(history))
    block = RENDERERS[history_type](history, names)
    query = query_prompt(num_arms)
    return instruction + "\n\n" + preamble + block + query
