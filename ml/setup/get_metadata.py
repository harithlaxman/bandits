"""
Fetch TMDB metadata for every MovieLens movie and write a CSV with
the schema:

    tmdbId, origin_country, original_language, original_title, overview,
    runtime, tagline, title, genres, top_actors

`genres` are MovieLens genres (pipe-separated, e.g. "Action|Comedy"),
joined in via links.csv. Everything else comes from TMDB's
/movie/{id}?append_to_response=credits endpoint. `top_actors` is the
first 5 cast members ordered by `order` (TMDB's billing order).

Auth: set one of these env vars before running:
    TMDB_BEARER_TOKEN   v4 read access token (preferred)
    TMDB_API_KEY        v3 API key

The script is resumable: if OUTPUT_PATH already exists, tmdbIds present
in it are skipped and new rows are appended.
"""

import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ML_DATA_DIR = Path("../datasets/ml-32m/")
OUTPUT_PATH = Path("./data/metadata.csv")
TMDB_BASE = "https://api.themoviedb.org/3"
N_TOP_ACTORS = 5
MAX_WORKERS = 16
REQUEST_TIMEOUT = 15
MAX_RETRIES = 5

COLUMNS = [
    "tmdbId",
    "origin_country",
    "original_language",
    "original_title",
    "overview",
    "runtime",
    "tagline",
    "title",
    "genres",
    "top_actors",
]


def _build_session() -> requests.Session:
    bearer = os.environ.get("TMDB_BEARER_TOKEN")
    api_key = os.environ.get("TMDB_API_KEY")
    if not bearer and not api_key:
        raise RuntimeError(
            "Set TMDB_BEARER_TOKEN (v4) or TMDB_API_KEY (v3) in the env"
        )
    s = requests.Session()
    if bearer:
        s.headers["Authorization"] = f"Bearer {bearer}"
    else:
        s.params = {"api_key": api_key}
    return s


def _fetch_one(session: requests.Session, tmdb_id: int) -> dict | None:
    url = f"{TMDB_BASE}/movie/{tmdb_id}"
    params = {"append_to_response": "credits", "language": "en-US"}
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 2 ** attempt))
            time.sleep(wait)
            continue
        time.sleep(2 ** attempt)
    return None


def _row_from_payload(tmdb_id: int, payload: dict | None,
                      genre_str: str) -> dict | None:
    if payload is None:
        return None
    cast = payload.get("credits", {}).get("cast", []) or []
    cast_sorted = sorted(cast, key=lambda c: c.get("order", 10**9))
    top_actors = [c["name"] for c in cast_sorted[:N_TOP_ACTORS]]
    return {
        "tmdbId": tmdb_id,
        "origin_country": str(payload.get("origin_country") or []),
        "original_language": payload.get("original_language") or "",
        "original_title": payload.get("original_title") or "",
        "overview": payload.get("overview") or "",
        "runtime": payload.get("runtime"),
        "tagline": payload.get("tagline") or "",
        "title": payload.get("title") or "",
        "genres": genre_str,
        "top_actors": str(top_actors),
    }


def main() -> None:
    links = pd.read_csv(ML_DATA_DIR / "links.csv")
    movies = pd.read_csv(ML_DATA_DIR / "movies.csv")
    df = links.merge(movies[["movieId", "genres"]], on="movieId", how="left")
    df = df.dropna(subset=["tmdbId"]).copy()
    df["tmdbId"] = df["tmdbId"].astype(int)
    df["genres"] = df["genres"].fillna("")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done: set[int] = set()
    if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0:
        existing = pd.read_csv(OUTPUT_PATH, usecols=["tmdbId"])
        done = set(existing["tmdbId"].astype(int).tolist())
        print(f"Resuming: {len(done)} rows already present")

    todo = df[~df["tmdbId"].isin(done)]
    print(f"Fetching {len(todo)} of {len(df)} movies")
    if todo.empty:
        return

    genre_map = dict(zip(todo["tmdbId"], todo["genres"]))
    write_header = not OUTPUT_PATH.exists() or OUTPUT_PATH.stat().st_size == 0
    session = _build_session()

    with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {
                ex.submit(_fetch_one, session, int(tid)): int(tid)
                for tid in todo["tmdbId"]
            }
            pbar = tqdm(total=len(futures), desc="TMDB")
            n_missing = 0
            for fut in as_completed(futures):
                tid = futures[fut]
                payload = fut.result()
                row = _row_from_payload(tid, payload, genre_map.get(tid, ""))
                if row is None:
                    n_missing += 1
                else:
                    writer.writerow(row)
                    f.flush()
                pbar.update(1)
            pbar.close()
            print(f"Done. {n_missing} movies returned no payload (404 / errors)")


if __name__ == "__main__":
    main()
