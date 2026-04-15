"""
MIND Large Train Preprocessing for LLM-as-Bandit Experiments
============================================================

1. Load behaviors.tsv and news.tsv from datasets/mind_large_train/
2. Sort behaviors by user + parsed timestamp
3. Filter to users with > min_interactions impression rows
4. Restrict news_df to articles referenced in those users' history+impressions
5. Encode articles with SentenceTransformer + PCA(100)
6. For each impression row, sample N negatives + 1 positive (fallback to a
   random history article when no positive exists)
7. Compute per-user embedding from the first 5 history article embeddings
8. Save data/news_with_embeddings_<min>.pkl and data/users_with_embeddings_<min>.pkl

Usage:
    python preprocess.py [--min-interactions 50] [--num-negatives 4] [--seed 42]
"""

import argparse
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# ─── Config ───────────────────────────────────────────────────────────
RAW_DIR = "../datasets/mind_large_train"
OUTPUT_DIR = "data"
SEED_HISTORY_SIZE = 5


# ─── Step 1: Load TSVs ───────────────────────────────────────────────

def load_data(raw_dir):
    """Load behaviors.tsv and news.tsv with the canonical MIND column names."""
    print("Loading TSVs...")
    imprs_df = pd.read_table(os.path.join(raw_dir, "behaviors.tsv"), header=None)
    news_df = pd.read_table(os.path.join(raw_dir, "news.tsv"), header=None)

    imprs_df.rename(
        columns={
            0: "impression_id",
            1: "user_id",
            2: "time",
            3: "history",
            4: "impressions",
        },
        inplace=True,
    )
    news_df.rename(
        columns={
            0: "news_id",
            1: "category",
            2: "sub_category",
            3: "title",
            4: "abstract",
            5: "url",
            6: "title_entities",
            7: "abstract_entities",
        },
        inplace=True,
    )

    print(f"  Behaviors: {len(imprs_df):,}")
    print(f"  News:      {len(news_df):,}")

    return imprs_df, news_df


# ─── Step 2-3: Sort & Filter Users ──────────────────────────────────

def sort_and_filter_users(imprs_df, min_interactions):
    """Parse timestamps, sort per user, keep users with strictly more than `min_interactions` rows."""
    imprs_df = imprs_df.copy()
    imprs_df["_parsed_time"] = pd.to_datetime(
        imprs_df["time"], format="%m/%d/%Y %I:%M:%S %p"
    )
    imprs_df = (
        imprs_df.sort_values(["user_id", "_parsed_time"])
        .drop(columns="_parsed_time")
        .reset_index(drop=True)
    )

    filtered_df = imprs_df.groupby("user_id").filter(
        lambda g: len(g) > min_interactions
    )
    n_users = filtered_df["user_id"].nunique()
    print(f"Users with > {min_interactions} interactions: {n_users}")

    return filtered_df


# ─── Step 4: Filter News ────────────────────────────────────────────

def filter_news(news_df, filtered_imprs_df):
    """Restrict news_df to articles referenced in users' history or impressions."""
    history_ids = filtered_imprs_df["history"].dropna().str.split().explode()
    impression_ids = (
        filtered_imprs_df["impressions"]
        .dropna()
        .str.split()
        .explode()
        .str.rsplit("-", n=1)
        .str[0]
    )
    referenced_ids = set(history_ids) | set(impression_ids)

    filtered_news_df = news_df[news_df["news_id"].isin(referenced_ids)].copy()
    print(f"News: {len(news_df):,} -> {len(filtered_news_df):,} (referenced)")

    return filtered_news_df


# ─── Step 5: Encode Articles ────────────────────────────────────────

def encode_articles(filtered_news_df):
    """Embed articles with SentenceTransformer and reduce to 100 dims with PCA."""
    print("Encoding articles...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = (
        "category: " + filtered_news_df["category"].fillna("")
        + "\nsub-category: " + filtered_news_df["sub_category"].fillna("")
        + "\ntitle: " + filtered_news_df["title"].fillna("")
        + "\nabstract: " + filtered_news_df["abstract"].fillna("")
    ).tolist()

    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    pca = PCA(n_components=100)
    reduced_embeddings = pca.fit_transform(embeddings)

    filtered_news_df = filtered_news_df.copy()
    filtered_news_df["embedding"] = list(embeddings)
    filtered_news_df["pca_embedding"] = list(reduced_embeddings)
    filtered_news_df.set_index("news_id", inplace=True)

    return filtered_news_df


# ─── Step 6-7: Build User Impressions ───────────────────────────────

def build_user_impressions(filtered_imprs_df, news_with_embeddings, num_negatives, rng):
    """
    Compress each impression row to `num_negatives` rejected + 1 clicked article,
    and compute a per-user embedding from the first 5 history articles.
    """
    print("Building user impressions...")
    df = filtered_imprs_df.reset_index(drop=True).copy()

    user_embeddings = {}
    for idx, r in df.iterrows():
        if r["user_id"] not in user_embeddings:
            history = r["history"].split(" ")[:SEED_HISTORY_SIZE]
            embeddings = [news_with_embeddings["embedding"][a] for a in history]
            user_embeddings[idx] = np.mean(np.array(embeddings), axis=0)

        article_ids = r["impressions"].split()
        clicked = []
        rejected = []
        for article in article_ids:
            if article[-1:] == "1":
                clicked.append(article)
            else:
                rejected.append(article)

        if len(clicked) == 0:
            history_articles = r["history"].split(" ")
            chosen = rng.choice(history_articles)
            chosen = f"{chosen}-1"
        else:
            chosen = rng.choice(clicked)

        negatives = list(rng.choice(rejected, size=num_negatives, replace=True))
        final = negatives + [chosen]

        df.at[idx, "impressions"] = " ".join(final)

    df.set_index("user_id", inplace=True)
    df["embedding"] = df.index.map(user_embeddings)

    return df


# ─── Step 8: Main ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MIND Large Train for LLM-as-Bandit experiments"
    )
    parser.add_argument("--raw-dir", default=RAW_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--min-interactions", type=int, default=50)
    parser.add_argument("--num-negatives", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    imprs_df, news_df = load_data(args.raw_dir)
    filtered_imprs_df = sort_and_filter_users(imprs_df, args.min_interactions)
    filtered_news_df = filter_news(news_df, filtered_imprs_df)
    news_with_embeddings = encode_articles(filtered_news_df)
    users_df = build_user_impressions(
        filtered_imprs_df, news_with_embeddings, args.num_negatives, rng
    )

    os.makedirs(args.output_dir, exist_ok=True)
    news_path = os.path.join(
        args.output_dir, f"news_with_embeddings_{args.min_interactions}.pkl"
    )
    users_path = os.path.join(
        args.output_dir, f"users_with_embeddings_{args.min_interactions}.pkl"
    )
    news_with_embeddings.to_pickle(news_path)
    users_df.to_pickle(users_path)

    print(f"\nSaved files to {args.output_dir}/")
    print(f"  {os.path.basename(news_path)}: {len(news_with_embeddings)} articles")
    print(f"  {os.path.basename(users_path)}: {users_df.index.nunique()} users, {len(users_df)} impressions")
    print("\nDone!")


if __name__ == "__main__":
    main()
