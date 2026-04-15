import json
import re
from typing import Callable

from pydantic import BaseModel


class Recommendation(BaseModel):
    """Structured output for chain-of-thought prompting."""

    reasoning: str
    choice: int


def build_candidate_list(candidates: list, item_info_fn: Callable) -> str:
    """Format candidates as numbered options with full item info."""
    lines = []
    for i, item_id in enumerate(candidates, 1):
        try:
            info = item_info_fn(item_id)
            lines.append(f"{i}. {info}")
        except KeyError:
            lines.append(f"{i}. [Item unavailable]")
    return "\n\n".join(lines)


def build_history_text(rec_history: list, arrow: str = "→") -> str:
    """
    Format past recommendations compactly.
    rec_history is a list of (title, was_positive) tuples.
    """
    if not rec_history:
        return "No recommendations yet."

    lines = []
    for i, (title, positive) in enumerate(rec_history, 1):
        status = "REWARD: 1" if positive else "REWARD: 0"
        lines.append(f'Round {i}: "{title}" {arrow} {status}')
    return "\n".join(lines)


def parse_impression(impression_str: str, id_cast: Callable = str) -> tuple:
    """
    Parse a MIND-style impression string like '12345-1 12346-0 12347-1 ...'

    Args:
        impression_str: Space-separated 'itemID-label' tokens.
        id_cast: Callable applied to each raw ID (e.g. `str` for MIND, `int` for MovieLens).

    Returns:
        candidates: list of item ids
        positives: set of item ids labelled 1
    """
    candidates = []
    positives = set()
    for item in impression_str.split():
        parts = item.rsplit("-", 1)
        if len(parts) == 2:
            item_id, label = id_cast(parts[0]), parts[1]
            candidates.append(item_id)
            if label == "1":
                positives.add(item_id)
        else:
            candidates.append(id_cast(item))
    return candidates, positives


def parse_model_choice(response_text: str, num_candidates: int) -> int:
    """
    Parse the model's response to extract the chosen number.
    Returns the index (0-based) of the chosen candidate, or -1 if parsing fails.
    """
    text = response_text.strip()
    match = re.search(r"\d+", text)
    if match:
        choice = int(match.group())
        if 1 <= choice <= num_candidates:
            return choice - 1
    return -1


def parse_cot_response(response_text: str, num_candidates: int) -> tuple:
    """
    Parse a CoT JSON response with `reasoning` and `choice` fields.
    Returns (index_0_based, reasoning_str), or (-1, "") if parsing fails.

    Four-stage fallback (in order):
      1. Strict Pydantic validation
      2. Raw json.loads
      3. Regex-extract a JSON object containing a "choice" field
      4. Bare integer regex (reasoning lost)
    """
    text = response_text.strip()

    try:
        rec = Recommendation.model_validate_json(text)
        if 1 <= rec.choice <= num_candidates:
            return rec.choice - 1, rec.reasoning
    except Exception:
        pass

    try:
        data = json.loads(text)
        choice = int(data.get("choice", -1))
        reasoning = str(data.get("reasoning", ""))
        if 1 <= choice <= num_candidates:
            return choice - 1, reasoning
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    json_match = re.search(r'\{[^{}]*"choice"\s*:\s*\d+[^{}]*\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            choice = int(data.get("choice", -1))
            reasoning = str(data.get("reasoning", ""))
            if 1 <= choice <= num_candidates:
                return choice - 1, reasoning
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    match = re.search(r"\d+", text)
    if match:
        choice = int(match.group())
        if 1 <= choice <= num_candidates:
            return choice - 1, ""

    return -1, ""
