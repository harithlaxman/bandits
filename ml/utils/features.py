"""Movie feature store + per-user interaction featurizer for the contextual bandit.

Each movie gets a dense "cos vector" used for semantic-similarity features:
  - with embeddings: a sentence-transformer embedding of title/tagline/overview
    (cached to data/movie_embeddings_<model>.npy, computed once over metadata.csv);
  - without embeddings: an L2-normalised multi-hot genre vector.

The bandit does NOT consume that vector directly (a per-user linear model over a
~400-dim vector is hopelessly under-determined with ~60 samples). Instead each
candidate is mapped to a compact, well-conditioned context vector phi(state, movie)
built from the candidate's relationship to the user's running taste profile
(centroids of liked/disliked movies, liked genres/actors/languages). See UserState.
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("./data/")
METADATA_CSV = DATA_DIR / "metadata.csv"

# phi(state, movie) layout — every entry is arm-dependent so it affects argmax.
PHI_FEATURES = (
    "bias",
    "cos_liked",          # cos(movie, liked centroid)
    "cos_disliked",       # cos(movie, disliked centroid)
    "max_cos_liked",      # nearest liked movie
    "genre_overlap_liked",
    "genre_overlap_disliked",
    "lang_match_liked",   # fraction of liked movies sharing this language
    "actor_overlap_liked",
    "runtime_z",
)
PHI_DIM = len(PHI_FEATURES)


def _parse_genres(raw) -> set[str]:
    if not isinstance(raw, str) or not raw or raw == "(no genres listed)":
        return set()
    return {g for g in raw.split("|") if g}


def _parse_actors(raw) -> set[str]:
    if not isinstance(raw, str) or not raw:
        return set()
    try:
        return {a for a in ast.literal_eval(raw) if a}
    except (ValueError, SyntaxError):
        return set()


def _clean_str(raw) -> str:
    return raw if isinstance(raw, str) else ""


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b) / (na * nb)


class MovieStore:
    """Per-movie features keyed by tmdbId, built once over the full metadata."""

    def __init__(self, df: pd.DataFrame, use_embeddings: bool,
                 embed_model: str, cache_dir: Path = DATA_DIR):
        df = df.copy()
        df["tmdbId"] = df["tmdbId"].astype(int)
        ids = df["tmdbId"].tolist()

        self.genres = {mid: _parse_genres(g)
                       for mid, g in zip(ids, df["genres"])}
        self.actors = {mid: _parse_actors(a)
                       for mid, a in zip(ids, df["top_actors"])}
        self.language = {mid: _clean_str(l)
                         for mid, l in zip(ids, df["original_language"])}

        runtime = pd.to_numeric(df["runtime"], errors="coerce")
        rt_mean = float(runtime.mean())
        rt_std = float(runtime.std()) or 1.0
        rt_z = ((runtime.fillna(rt_mean) - rt_mean) / rt_std).tolist()
        self.runtime_z = dict(zip(ids, rt_z))

        # structured multi-hot genre vectors (also the cos-vector fallback)
        vocab = sorted({g for gs in self.genres.values() for g in gs})
        self._genre_idx = {g: i for i, g in enumerate(vocab)}
        self._structured = {mid: self._structured_vec(self.genres[mid])
                            for mid in ids}

        if use_embeddings:
            emb = self._load_or_compute_embeddings(df, embed_model, cache_dir)
            self._cos_vec = emb
        else:
            self._cos_vec = self._structured
        self.cos_dim = next(iter(self._cos_vec.values())).shape[0]

    def _structured_vec(self, genres: set[str]) -> np.ndarray:
        v = np.zeros(len(self._genre_idx), dtype=np.float32)
        for g in genres:
            v[self._genre_idx[g]] = 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def _load_or_compute_embeddings(self, df: pd.DataFrame, embed_model: str,
                                    cache_dir: Path) -> dict[int, np.ndarray]:
        slug = embed_model.replace("/", "_")
        emb_path = cache_dir / f"movie_embeddings_{slug}.npy"
        ids_path = cache_dir / f"movie_embeddings_{slug}_ids.npy"
        ids = df["tmdbId"].tolist()

        if emb_path.exists() and ids_path.exists():
            cached_ids = np.load(ids_path).tolist()
            cached_emb = np.load(emb_path)
            if set(ids).issubset(set(cached_ids)):
                lookup = {int(mid): cached_emb[i]
                          for i, mid in enumerate(cached_ids)}
                return {mid: lookup[mid] for mid in ids}

        from sentence_transformers import SentenceTransformer

        def _text(row) -> str:
            parts = [_clean_str(row.title), _clean_str(row.tagline),
                     _clean_str(row.overview)]
            return ". ".join(p for p in parts if p) or "movie"

        texts = [_text(row) for row in df.itertuples(index=False)]
        print(f"[features] embedding {len(texts)} movies with {embed_model} "
              f"(one-time; cached to {emb_path})")
        model = SentenceTransformer(embed_model)
        emb = model.encode(texts, batch_size=256, show_progress_bar=True,
                           normalize_embeddings=True).astype(np.float32)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, emb)
        np.save(ids_path, np.array(ids, dtype=np.int64))
        return {mid: emb[i] for i, mid in enumerate(ids)}

    def cos_vec(self, mid: int) -> np.ndarray:
        return self._cos_vec[mid]

    @classmethod
    def from_metadata(cls, use_embeddings: bool, embed_model: str,
                      csv_path: Path = METADATA_CSV,
                      cache_dir: Path = DATA_DIR) -> "MovieStore":
        return cls(pd.read_csv(csv_path), use_embeddings, embed_model, cache_dir)


class UserState:
    """A single user's running taste profile, updated from revealed rewards.

    Labels: 1 = liked, -1 = disliked, 0 = neutral (informs neither centroid).
    phi() reads the profile *as of now*; observe() folds a revealed movie in
    afterwards — so a candidate is never scored against its own membership.
    """

    def __init__(self, store: MovieStore):
        self.store = store
        d = store.cos_dim
        self._liked_vecs: list[np.ndarray] = []
        self._sum_liked = np.zeros(d, dtype=np.float32)
        self._sum_disliked = np.zeros(d, dtype=np.float32)
        self.n_liked = 0
        self.n_disliked = 0
        self.liked_genres: set[str] = set()
        self.disliked_genres: set[str] = set()
        self.liked_actors: set[str] = set()
        self._liked_lang: dict[str, int] = {}

    def phi(self, mid: int) -> np.ndarray:
        s = self.store
        v = s.cos_vec(mid)

        cos_liked = (_cos(v, self._sum_liked / self.n_liked)
                     if self.n_liked else 0.0)
        cos_disliked = (_cos(v, self._sum_disliked / self.n_disliked)
                        if self.n_disliked else 0.0)
        max_cos_liked = (max(_cos(v, lv) for lv in self._liked_vecs)
                         if self._liked_vecs else 0.0)

        g = s.genres[mid]
        gn = max(1, len(g))
        genre_ov_liked = len(g & self.liked_genres) / gn
        genre_ov_disliked = len(g & self.disliked_genres) / gn

        lang = s.language[mid]
        lang_match = (self._liked_lang.get(lang, 0) / self.n_liked
                      if self.n_liked else 0.0)

        a = s.actors[mid]
        actor_ov_liked = len(a & self.liked_actors) / max(1, len(a))

        return np.array([
            1.0, cos_liked, cos_disliked, max_cos_liked,
            genre_ov_liked, genre_ov_disliked, lang_match,
            actor_ov_liked, s.runtime_z[mid],
        ], dtype=np.float64)

    def observe(self, mid: int, label: int) -> None:
        s = self.store
        v = s.cos_vec(mid)
        if label == 1:
            self._liked_vecs.append(v)
            self._sum_liked += v
            self.n_liked += 1
            self.liked_genres |= s.genres[mid]
            self.liked_actors |= s.actors[mid]
            lang = s.language[mid]
            self._liked_lang[lang] = self._liked_lang.get(lang, 0) + 1
        elif label == -1:
            self._sum_disliked += v
            self.n_disliked += 1
            self.disliked_genres |= s.genres[mid]
