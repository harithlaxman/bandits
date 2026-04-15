"""
MovieLens 25M Preprocessing for LLM-as-Bandit Experiments
=========================================================

1. Download MovieLens 25M from grouplens.org
2. Load ratings.csv, movies.csv, tags.csv, links.csv
3. Enrich movie metadata (parse genres, aggregate tags)
4. Filter users with >= min_liked liked ratings (rating >= 4)
5. Sort liked ratings by timestamp per user
6. First 5 liked movies = seed/profile context; rest = bandit rounds
7. Build MIND-style impression strings (1 positive + N negatives per round)
8. Save as parquet + pickle files

Usage:
    python preprocess.py [--min-liked 55] [--num-users 50] [--num-negatives 9]
"""

import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
import argparse

# ─── Config ───────────────────────────────────────────────────────────
DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
RAW_DIR = "../datasets/ml-25m"
OUTPUT_DIR = "data"
ZIP_PATH = "../datasets/ml-25m.zip"


# ─── Step 1: Download & Extract ──────────────────────────────────────

def download_and_extract():
    """Download MovieLens 25M if not already present."""
    if os.path.exists(RAW_DIR):
        print(f"Dataset already exists at {RAW_DIR}")
        return

    os.makedirs(os.path.dirname(RAW_DIR), exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        print(f"Downloading MovieLens 25M from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        print("Download complete.")

    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(os.path.dirname(RAW_DIR))
    print("Extraction complete.")


# ─── Step 2: Load CSVs ───────────────────────────────────────────────

def load_data():
    """Load all CSVs into DataFrames."""
    print("Loading CSVs...")
    ratings = pd.read_csv(os.path.join(RAW_DIR, "ratings.csv"))
    movies = pd.read_csv(os.path.join(RAW_DIR, "movies.csv"))
    tags = pd.read_csv(os.path.join(RAW_DIR, "tags.csv"))
    links = pd.read_csv(os.path.join(RAW_DIR, "links.csv"))

    print(f"  Ratings: {len(ratings):,}")
    print(f"  Movies:  {len(movies):,}")
    print(f"  Tags:    {len(tags):,}")
    print(f"  Links:   {len(links):,}")

    return ratings, movies, tags, links


# ─── Step 3: Enrich Movie Metadata ───────────────────────────────────

def enrich_movies(movies, tags, links):
    """Parse genres and add aggregated tags."""
    print("Enriching movie metadata...")

    movies = movies.copy()

    # Parse genres from pipe-separated string into list
    movies["genres_list"] = movies["genres"].apply(
        lambda g: g.split("|") if g != "(no genres listed)" else []
    )

    # Aggregate tags per movie (top 5 most common)
    tag_agg = (
        tags.groupby("movieId")["tag"]
        .apply(lambda x: list(x.value_counts().head(5).index))
        .reset_index()
        .rename(columns={"tag": "tags"})
    )
    movies = movies.merge(tag_agg, on="movieId", how="left")
    movies["tags"] = movies["tags"].apply(lambda x: x if isinstance(x, list) else [])

    # Merge tmdbId from links
    movies = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")

    all_genres = set(g for gl in movies["genres_list"] for g in gl)
    print(f"  Unique genres: {len(all_genres)}")
    print(f"  Movies with tags: {(movies['tags'].apply(len) > 0).sum():,}")

    return movies


# ─── Steps 4-5: Filter & Prepare Users ──────────────────────────────

SEED_SIZE = 5  # number of liked movies used as user profile context

def prepare_users(ratings, movies, min_liked=55, num_users=50, seed=42):
    """Filter users by liked-rating count, sort by time, split into seed + rounds."""
    print(f"\nFiltering users with >= {min_liked} liked ratings (rating >= 4)...")

    liked_ratings = ratings[ratings["rating"] >= 4].copy()

    # Count liked ratings per user
    liked_counts = liked_ratings.groupby("userId").size()
    eligible_users = liked_counts[liked_counts >= min_liked].index
    print(f"  Eligible users: {len(eligible_users):,}")

    # Sample users
    rng = np.random.RandomState(seed)
    sampled_users = rng.choice(
        eligible_users, size=min(num_users, len(eligible_users)), replace=False
    )
    print(f"  Sampled users: {len(sampled_users)}")

    # Filter to sampled users, sort by timestamp
    user_liked = liked_ratings[liked_ratings["userId"].isin(sampled_users)].copy()
    user_liked = user_liked.sort_values(["userId", "timestamp"])

    # Merge movie info
    user_liked = user_liked.merge(
        movies[["movieId", "title", "genres", "genres_list", "tags"]],
        on="movieId",
        how="left",
    )

    # Split per user: first SEED_SIZE = seed/profile, rest = round pool
    user_data = {}
    for uid, group in user_liked.groupby("userId"):
        movie_ids = group["movieId"].tolist()
        seed_ids = movie_ids[:SEED_SIZE]
        round_ids = movie_ids[SEED_SIZE:]
        user_data[uid] = {"seed": seed_ids, "rounds": round_ids}

    avg_rounds = np.mean([len(d["rounds"]) for d in user_data.values()])
    print(f"  Seed movies per user: {SEED_SIZE}")
    print(f"  Avg round movies per user: {avg_rounds:.0f}")

    return user_data, sampled_users


# ─── Build Impressions ───────────────────────────────────────────────

def build_impressions(user_data, movies, num_negatives=9, seed=42):
    """
    Build MIND-style impression strings for bandit rounds.

    For each round movie (liked movie after the seed):
      - The movie is the positive (movieId-1)
      - Sample num_negatives random movies as negatives (movieId-0)

    Returns a DataFrame indexed by userId with columns: history, impressions.
    One row per round (multiple rows per user).
    """
    print("\nBuilding impressions...")
    rng = np.random.RandomState(seed)

    all_movie_ids = movies["movieId"].values

    records = []

    for uid, data in user_data.items():
        seed_ids = data["seed"]
        round_ids = data["rounds"]
        all_user_ids = set(seed_ids + round_ids)

        history_str = " ".join(str(mid) for mid in seed_ids)

        for pos_id in round_ids:
            # Sample negatives from movies user never liked
            neg_pool = np.setdiff1d(all_movie_ids, list(all_user_ids))
            if len(neg_pool) < num_negatives:
                neg_pool = np.setdiff1d(all_movie_ids, [pos_id])
            neg_ids = rng.choice(
                neg_pool,
                size=min(num_negatives, len(neg_pool)),
                replace=False,
            )

            items = [f"{pos_id}-1"]
            items.extend(f"{nid}-0" for nid in neg_ids)
            impression_str = " ".join(items)

            records.append({
                "userId": uid,
                "history": history_str,
                "impressions": impression_str,
            })

    impressions_df = pd.DataFrame(records)
    n_users = impressions_df["userId"].nunique()
    print(f"  Total impressions: {len(impressions_df):,}")
    print(f"  Avg per user: {len(impressions_df) / n_users:.0f}")

    return impressions_df


# ─── Step 8: Save ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MovieLens 25M for LLM-as-Bandit experiments"
    )
    parser.add_argument(
        "--min-liked", type=int, default=55,
        help="Minimum liked ratings (>= 4) per user (5 seed + 50 rounds)",
    )
    parser.add_argument(
        "--num-users", type=int, default=50,
        help="Number of users to sample",
    )
    parser.add_argument(
        "--num-negatives", type=int, default=9,
        help="Negative candidates per impression",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Download & extract
    download_and_extract()

    # 2. Load CSVs
    ratings, movies_raw, tags, links = load_data()

    # 3. Enrich metadata
    movies = enrich_movies(movies_raw, tags, links)

    # 4-5. Filter & prepare users (seed + rounds)
    user_data, sampled_users = prepare_users(
        ratings, movies, args.min_liked, args.num_users, args.seed
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save enriched movies
    movies.to_parquet(
        os.path.join(OUTPUT_DIR, "movies_enriched.parquet"), index=False
    )

    # 6-7. Build impressions & save pickle files
    impressions_df = build_impressions(
        user_data, movies, args.num_negatives, args.seed
    )

    movies_indexed = movies.set_index("movieId")
    movies_indexed.to_pickle(os.path.join(OUTPUT_DIR, "movies.pkl"))

    impressions_indexed = impressions_df.set_index("userId")
    impressions_indexed.to_pickle(os.path.join(OUTPUT_DIR, "users.pkl"))

    print(f"\nSaved files to {OUTPUT_DIR}/")
    print(f"  movies.pkl: {len(movies_indexed)} movies")
    print(f"  users.pkl: {len(sampled_users)} users, {len(impressions_df)} impressions")
    print("\nDone!")


if __name__ == "__main__":
    main()
