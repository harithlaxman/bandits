"""Shared MIND-news domain code: dataset loading, item lookups, and prompt builders.

Imported by every bl/slm*.py entry script. Domain-specific knowledge stops here —
bandit_llm/ never touches this file or the dataframes.

The entry script is responsible for putting the repo root on sys.path before
importing this module (so the `from bandit_llm import ...` line below resolves).
"""

import pandas as pd

from bandit_llm import build_candidate_list, build_history_text, parse_impression

# ─── Dataset ──────────────────────────────────────────────────────────
ARTICLES_DF = pd.read_pickle("data/news_with_embeddings_50.pkl")
USERS_DF = pd.read_pickle("data/users_with_embeddings_50.pkl")

# ─── System prompts ───────────────────────────────────────────────────
SYSTEM_PROMPT_PLAIN = """You are a news recommendation agent. You will go through multiple rounds of recommendations for a single user.

In each round, you will see:
- The user's reading profile (articles they liked before)
- Your past recommendations and whether the user clicked them
- A new list of candidate articles

Your goal is to pick the ONE article the user is most likely to click. You get a reward every time the user clicks the article you recommended.
Focus on maximizing the reward over time. No matter what do not ask for user feedback regarding strategy.

Reply with ONLY the number of your chosen article. Do not explain your reasoning."""

SYSTEM_PROMPT_COT = """You are a news recommendation agent. You will go through multiple rounds of recommendations for a single user.

In each round, you will see:
- The user's reading profile (articles they liked before)
- Your past recommendations and whether the user clicked them
- A new list of candidate articles

Your goal is to pick the ONE article the user is most likely to click. You get a reward every time the user clicks the article you recommended.
Focus on maximizing the reward over time. No matter what do not ask for user feedback regarding strategy.

You MUST respond with a JSON object containing exactly two fields:
1. "reasoning": A step-by-step chain-of-thought analysis. Analyze the user's reading preferences, review your past recommendation outcomes to learn what works, compare each candidate article against the user's profile, and explain why you chose your final pick.
2. "choice": The number of your chosen article (an integer).

Example response format:
{"reasoning": "The user prefers sports articles based on their history. Article 3 is about basketball which matches. My past sports picks got clicks.", "choice": 3}"""


# ─── Item lookups ─────────────────────────────────────────────────────


def get_article_info(article_id: str) -> str:
    """Return formatted article details: category, subcategory, title, abstract."""
    article = ARTICLES_DF.loc[article_id]
    return (
        f"Category: {article['category']} - {article['sub_category']}\n"
        f"Title: {article['title']}\n"
        f"Abstract: {article['abstract']}"
    )


def get_article_title(article_id: str) -> str:
    """Return just the title — used for compact history display."""
    content = ARTICLES_DF.loc[article_id]
    return f"Category: {content['category']} - {content['sub_category']}\n{content['title']}"


def get_user_seed_articles(user_id: str) -> str:
    """Get the last 5 articles from user's click history as the user profile."""
    history_str = USERS_DF.loc[user_id]["history"]
    if isinstance(history_str, list):
        history_str = history_str[0]
    elif isinstance(history_str, pd.Series):
        history_str = history_str.to_list()[0]

    past_ids = str(history_str).split()[-5:]
    lines = []
    for i, aid in enumerate(past_ids, 1):
        lines.append(f"{i}. {get_article_info(aid)}")
    return "\n\n".join(lines)


def get_user_rounds(user_id: str) -> list:
    """Return the per-round (candidates, clicked) tuples for a user."""
    impressions = USERS_DF.loc[user_id]["impressions"].to_list()
    return [parse_impression(s, id_cast=str) for s in impressions]


# ─── Prompt builders ──────────────────────────────────────────────────


def build_prompt_plain(
    user_seed: str,
    rec_history: list,
    candidates: list,
    round_num: int,
    cumulative_reward: int,
) -> str:
    history_text = build_history_text(rec_history)
    candidate_text = build_candidate_list(candidates, get_article_info)

    return f"""Articles user has liked in the past:
{user_seed}

Your Past Recommendations:
{history_text}

Total reward gained: {cumulative_reward}

Round {round_num}: Choose ONE article:
{candidate_text}

Recommend an article that the user is most likely to pick. Reply with ONLY the number."""


def build_prompt_cot(
    user_seed: str,
    rec_history: list,
    candidates: list,
    round_num: int,
    cumulative_reward: int,
) -> str:
    history_text = build_history_text(rec_history)
    candidate_text = build_candidate_list(candidates, get_article_info)

    return f"""Articles user has liked in the past:
{user_seed}

Your Past Recommendations:
{history_text}

Total reward gained: {cumulative_reward}

Round {round_num}: Choose ONE article:
{candidate_text}

Analyze the user's preferences and past recommendation outcomes step by step, then pick the best article. Respond with JSON containing "reasoning" and "choice" fields."""
