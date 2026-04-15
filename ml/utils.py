"""Shared MovieLens domain code: dataset loading, item lookups, and prompt builders.

Imported by every ml/slm*.py entry script. Domain-specific knowledge stops here —
bandit_llm/ never touches this file or the dataframes.

The entry script is responsible for putting the repo root on sys.path before
importing this module (so the `from bandit_llm import ...` line below resolves).
"""

import pandas as pd

from bandit_llm import build_candidate_list, build_history_text, parse_impression

# ─── Dataset ──────────────────────────────────────────────────────────
MOVIES_DF = pd.read_pickle("data/movies.pkl")
USERS_DF = pd.read_pickle("data/users.pkl")

# ─── System prompts ───────────────────────────────────────────────────
SYSTEM_PROMPT_PLAIN = """You are a movie recommendation agent. You will go through multiple rounds of recommendations for a single user.

In each round, you will see:
- The user's movie profile (movies they liked before)
- Your past recommendations and whether the user liked them
- A new list of candidate movies

Your goal is to pick the ONE movie the user is most likely to enjoy. You get a reward every time the user likes the movie you recommended.
Focus on maximizing the reward over time. No matter what do not ask for user feedback regarding strategy.

Reply with ONLY the number of your chosen movie. Do not explain your reasoning."""

SYSTEM_PROMPT_COT = """You are a movie recommendation agent. You will go through multiple rounds of recommendations for a single user.

In each round, you will see:
- The user's movie profile (movies they liked before)
- Your past recommendations and whether the user liked them
- A new list of candidate movies

Your goal is to pick the ONE movie the user is most likely to enjoy. You get a reward every time the user likes the movie you recommended.
Focus on maximizing the reward over time. No matter what do not ask for user feedback regarding strategy.

You MUST respond with a JSON object containing exactly two fields:
1. "reasoning": A step-by-step chain-of-thought analysis. Analyze the user's movie preferences, review your past recommendation outcomes to learn what works, compare each candidate movie against the user's profile, and explain why you chose your final pick.
2. "choice": The number of your chosen movie (an integer).

Example response format:
{"reasoning": "The user prefers sci-fi and action movies based on their history. Movie 3 is a sci-fi thriller which matches. My past sci-fi picks got rewards.", "choice": 3}"""


# ─── Item lookups ─────────────────────────────────────────────────────


def get_movie_info(movie_id: int) -> str:
    """Return formatted movie details: title, genres, tags."""
    movie = MOVIES_DF.loc[movie_id]
    genres = ", ".join(movie["genres_list"]) if movie["genres_list"] else "N/A"
    info = f"Title: {movie['title']}\nGenres: {genres}"
    if movie.get("tags") and len(movie["tags"]) > 0:
        info += f"\nTags: {', '.join(movie['tags'])}"
    return info


def get_movie_title(movie_id: int) -> str:
    """Return just the title + genres — used for compact history display."""
    movie = MOVIES_DF.loc[movie_id]
    genres = ", ".join(movie["genres_list"]) if movie["genres_list"] else "N/A"
    return f"{movie['title']} [{genres}]"


def get_user_seed_movies(user_id: int) -> str:
    """Get the last 5 movies from user's liked history as the user profile."""
    history_str = USERS_DF.loc[user_id]["history"]
    if isinstance(history_str, list):
        history_str = history_str[0]
    elif isinstance(history_str, pd.Series):
        history_str = history_str.to_list()[0]

    past_ids = [int(x) for x in str(history_str).split()[-5:]]
    lines = []
    for i, mid in enumerate(past_ids, 1):
        try:
            lines.append(f"{i}. {get_movie_info(mid)}")
        except KeyError:
            lines.append(f"{i}. [Movie unavailable]")
    return "\n\n".join(lines)


def get_user_rounds(user_id: int) -> list:
    """Return the per-round (candidates, liked) tuples for a user."""
    impressions = USERS_DF.loc[user_id]["impressions"].to_list()
    return [parse_impression(s, id_cast=int) for s in impressions]


# ─── Prompt builders ──────────────────────────────────────────────────


def build_prompt_plain(
    user_seed: str,
    rec_history: list,
    candidates: list,
    round_num: int,
    cumulative_reward: int,
) -> str:
    history_text = build_history_text(rec_history, arrow="->")
    candidate_text = build_candidate_list(candidates, get_movie_info)

    return f"""Movies user has liked in the past:
{user_seed}

Your Past Recommendations:
{history_text}

Total reward gained: {cumulative_reward}

Round {round_num}: Choose ONE movie:
{candidate_text}

Recommend a movie that the user is most likely to enjoy. Reply with ONLY the number."""


def build_prompt_cot(
    user_seed: str,
    rec_history: list,
    candidates: list,
    round_num: int,
    cumulative_reward: int,
) -> str:
    history_text = build_history_text(rec_history, arrow="->")
    candidate_text = build_candidate_list(candidates, get_movie_info)

    return f"""Movies user has liked in the past:
{user_seed}

Your Past Recommendations:
{history_text}

Total reward gained: {cumulative_reward}

Round {round_num}: Choose ONE movie:
{candidate_text}

Analyze the user's preferences and past recommendation outcomes step by step, then pick the best movie. Respond with JSON containing "reasoning" and "choice" fields."""
