# ml — bandit experiments on MovieLens

LLM as Contextual-bandit experiments over MovieLens-32M. Raw data expected to be at `../datasets/ml-32m/`. But not necessary.
Python deps via `uv`.

## Pre-processing

Two-step pipeline; metadata must exist before impressions (impressions filter to movies present in `metadata.csv`).

1. **`get_metadata.py` → `./data/metadata.csv`** — fetches TMDB `/movie/{id}` (with credits) for every movie in `links.csv`. Needs `TMDB_BEARER_TOKEN` or `TMDB_API_KEY` in env. Resumable.

2. **`get_impressions.py` → `./data/impressions_stationary.pkl`** — per user, anchors each impression on 1 liked movie (`rating ≥ 4.0`) + 4 non-liked from a sliding window over their chronological non-liked history. First 5 ratings are split off as a `cold_start` row.

## Experiment

LLM as Contextual Bandits.

Users with at least 60 impressions are sampled at random from the impressions.pkl file created earlier. At each iteration step, an LLM is given the first 5 ratings of the user and then is presented with 5 movies to pick from, to recommend to the user. The reward is then fed back to the LLM for the next iteration for the same user.
