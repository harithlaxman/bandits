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


def build_covertype_prompt(history, current_context_text: str, num_arms: int,
                           window: int = 30) -> str:
    """Contextual (Covertype) prompt: instruction + per-button summary + a sliding window
    of the last `window` raw interactions (terrain -> button, reward) + current terrain.
    Bounded size at any horizon: summaries carry the long-run stats, the window carries
    the recent context-reward pairs."""
    from covertype import COVERTYPE_INSTRUCTION

    names = action_names(num_arms)
    instruction = COVERTYPE_INSTRUCTION.format(num_arms, "[" + ", ".join(names) + "]")

    parts = [instruction, "", f"So far you have interacted {len(history)} times."]
    if history:
        parts.append("Summary per button:")
        counts, rewards = _per_arm_stats(history, num_arms)
        for name, n, total in zip(names, counts, rewards):
            avg = total / (n + 1e-6)
            parts.append(f"{name} {ACTION_UNIT}: chosen {n} times, average reward {avg:.2f}")
        recent = history[-window:]
        parts.append("")
        parts.append(f"Most recent {len(recent)} interactions:")
        for exp in recent:
            parts.append(f"({exp.context_text}) -> {exp.action_name} {ACTION_UNIT}, "
                         f"reward {exp.reward}")
    parts.append("")
    parts.append("Current terrain:")
    parts.append(current_context_text)
    return "\n".join(parts) + query_prompt(num_arms)
