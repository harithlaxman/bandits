"""
Impression generator for v2 with 3-tier rewards.

Reward mapping (from raw 0.5-5.0 ratings):
    rating >= 4.0  →  reward  1  (positive / liked)
    3.0 <= rating < 4.0  →  reward  0  (neutral)
    rating < 3.0  →  reward -1  (negative / disliked)

Each impression is anchored on one *positive* movie and paired with
4 neutral-or-negative movies drawn from a sliding window, exactly
like v1 does with its binary liked/disliked split.
"""

from typing import cast

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

ML_DATA_DIR = Path("../datasets/ml-32m/")
OUTPUT_DATA_DIR = Path("./data")
MIN_POSITIVES = 50  # minimum positive movies a user must have
WINDOW_SIZE = 8  # sliding window over non-positive movies
N_NON_POS = 4  # number of neutral/negative per impression
STRIDE = 2  # stride of the sliding window
COLD_START_SIZE = 5  # number of movies used for cold-start context
N_CHOICES = 5  # total movies per impression (1 pos + 4 non-pos)
SEED = 42


def _rating_to_reward(rating: float) -> int:
    """Map a raw 0.5-5.0 rating to a 3-tier reward."""
    if rating >= 4.0:
        return 1
    elif rating >= 3.0:
        return 0
    else:
        return -1


def _prepare_user_data(ratings_csv: str, links_csv: str):
    """
    Read raw ratings, map movieId → tmdbId, assign 3-tier rewards, and
    split into cold-start context + main data.

    Returns
    -------
    cold_data : dict[int, tuple[list[int], list[int]]]
        userId → (tmdbIds, rewards) for the first COLD_START_SIZE movies.
    users_df : pd.DataFrame
        One row per qualifying user with columns
        ``userId``, ``positive`` (list[int]), ``non_positive`` (list[int]),
        ``non_positive_rewards`` (list[int]).
    """
    # --- Load and map ids ------------------------------------------------
    links_df = pd.read_csv(ML_DATA_DIR / links_csv)
    links_df.dropna(subset=["tmdbId"], inplace=True)
    mid_to_tid = links_df.set_index("movieId")["tmdbId"]

    ratings_df = pd.read_csv(ML_DATA_DIR / ratings_csv)
    ratings_df["tmdbId"] = ratings_df["movieId"].map(mid_to_tid)
    ratings_df.dropna(subset=["tmdbId"], inplace=True)
    ratings_df["tmdbId"] = ratings_df["tmdbId"].astype(int)

    # Filter to movies that have metadata
    metadata_csv = OUTPUT_DATA_DIR / "metadata.csv"
    if metadata_csv.exists():
        meta_df = pd.read_csv(metadata_csv)
        ratings_df = ratings_df[ratings_df["tmdbId"].isin(meta_df["tmdbId"])]

    # --- Assign 3-tier rewards -------------------------------------------
    ratings_df["reward"] = ratings_df["rating"].apply(_rating_to_reward)
    ratings_df.sort_values(["userId", "timestamp"], inplace=True)

    # --- Cold-start split ------------------------------------------------
    ratings_df["_n"] = ratings_df.groupby("userId").cumcount()
    cold_start_df = cast(pd.DataFrame, ratings_df[ratings_df["_n"] < COLD_START_SIZE])
    main_df = cast(pd.DataFrame, ratings_df[ratings_df["_n"] >= COLD_START_SIZE])

    cold_data: dict[int, tuple[list[int], list[int]]] = {}
    for uid, g in cold_start_df.groupby("userId"):
        if len(g) < COLD_START_SIZE:
            continue
        cold_data[int(uid)] = (
            g["tmdbId"].astype(int).tolist(),
            g["reward"].astype(int).tolist(),
        )

    # --- Separate positive vs non-positive movies per user ---------------
    positive = main_df[main_df["reward"] == 1].groupby("userId")["tmdbId"].apply(list)
    # For non-positive, keep (tmdbId, reward) pairs so we can carry
    # the reward through to the impression labels.
    non_pos = (
        main_df[main_df["reward"] <= 0]
        .groupby("userId")
        .apply(
            lambda g: list(zip(g["tmdbId"].astype(int), g["reward"].astype(int))),
            include_groups=False,
        )
    )

    users_df = pd.DataFrame({"positive": positive, "non_positive": non_pos})
    users_df["positive"] = users_df["positive"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    users_df["non_positive"] = users_df["non_positive"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Filter to users with enough data
    users_df = users_df[
        (users_df["positive"].str.len() >= MIN_POSITIVES)
        & (
            users_df["non_positive"].str.len()
            >= STRIDE * (users_df["positive"].str.len() - 1) + WINDOW_SIZE
        )
    ].reset_index()

    return cold_data, users_df


def _build_main_impressions(
    users_df: pd.DataFrame, desc: str
) -> dict[int, list[tuple[list[int], list[int]]]]:
    """
    Build impression lists per user.

    For every positive movie, sample N_NON_POS movies from a sliding
    window over the user's non-positive list.  The impression contains
    5 tmdbIds and 5 corresponding rewards.

    Returns
    -------
    per_user : dict[int, list[tuple[list[int], list[int]]]]
        userId → [(impression_ids, impression_rewards), ...]
    """
    rng = np.random.default_rng(SEED)
    per_user: dict[int, list[tuple[list[int], list[int]]]] = {}

    for _, row in tqdm(users_df.iterrows(), total=len(users_df), desc=desc):
        user_id = int(row["userId"])
        positive_list = row["positive"]  # list[int] (tmdbIds)
        non_pos_list = row["non_positive"]  # list[(tmdbId, reward)]
        impressions: list[tuple[list[int], list[int]]] = []

        for i, pos_mid in enumerate(positive_list):
            window = non_pos_list[STRIDE * i : STRIDE * i + WINDOW_SIZE]
            if len(window) < N_NON_POS:
                break

            chosen_indices = rng.choice(len(window), size=N_NON_POS, replace=False)
            chosen = [window[j] for j in chosen_indices]

            # Build the impression: 1 positive + N_NON_POS non-positives
            impression_ids = [int(pos_mid)] + [int(mid) for mid, _ in chosen]
            impression_rewards = [1] + [int(r) for _, r in chosen]

            # Shuffle the order so the positive isn't always first
            order = rng.permutation(len(impression_ids))
            impression_ids = [impression_ids[j] for j in order]
            impression_rewards = [impression_rewards[j] for j in order]

            impressions.append((impression_ids, impression_rewards))

        per_user[user_id] = impressions

    return per_user


def _cold_row(uid: int, tmdbIds: list[int], rewards: list[int]) -> dict:
    return {
        "userId": uid,
        "impression": tmdbIds,
        "labels": rewards,
        "phase": "cold_start",
    }


def _main_row(uid: int, impression: list[int], rewards: list[int], phase: str) -> dict:
    return {
        "userId": uid,
        "impression": impression,
        "labels": rewards,
        "phase": phase,
    }


def build_stationary(n_main: int = 60, output: str = "impressions_stationary.pkl"):
    cold_data, users_df = _prepare_user_data("ratings.csv", "links.csv")
    main_per_user = _build_main_impressions(users_df, desc="stationary main")

    common = sorted(set(cold_data) & set(main_per_user))
    rows = []
    for uid in common:
        tmdbIds, rewards = cold_data[uid]
        rows.append(_cold_row(uid, tmdbIds, rewards))
        for impression_ids, impression_rewards in main_per_user[uid][:n_main]:
            rows.append(_main_row(uid, impression_ids, impression_rewards, "normal"))

    df = pd.DataFrame(rows)
    df.insert(0, "impression_id", range(len(df)))
    df.to_pickle(OUTPUT_DATA_DIR / output)
    print(
        f"Stationary: {len(common)} users, {len(df)} rows -> {OUTPUT_DATA_DIR / output}"
    )


if __name__ == "__main__":
    build_stationary()
