"""
LinUCB Baselines — MovieLens Edition
=====================================

Loads preprocessed MovieLens data (movies.pkl, users.pkl) and runs
LinUCB variants as baselines for comparison with LLM-as-bandit.

Usage:
    python hybridlinucb.py
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ─── Config ───────────────────────────────────────────────────────────

MOVIES_PKL = "data/movies.pkl"
USERS_PKL = "data/users.pkl"
RESULTS_DIR = "results"

N_IMPRESSIONS = 50
N_RUNS = 20
ALPHA_VALUES = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 1.5]


# ─── Data Loading ────────────────────────────────────────────────────

def load_data(users_pkl_path: str, movies_pkl_path: str):
    """
    Load and parse preprocessed MovieLens data.

    Returns:
        users_data: list of dicts per user with impressions parsed
        movies_df: the movies DataFrame
    """
    print("Loading data...")
    users_df = pd.read_pickle(users_pkl_path)
    movies_df = pd.read_pickle(movies_pkl_path)

    print(f"  Users: {len(users_df.index.unique())}")
    print(f"  Movies: {len(movies_df)}")

    # Build movie lookup
    movie_lookup = {}
    for movie_id, row in movies_df.iterrows():
        movie_lookup[movie_id] = {
            'movie_id': movie_id,
            'title': row.get('title', ''),
            'genres_list': row.get('genres_list', []),
            'tags': row.get('tags', []),
        }

    # Parse each user's data
    users_data = []
    user_ids = users_df.index.unique()

    for user_id in user_ids:
        user_rows = users_df.loc[user_id]
        if isinstance(user_rows, pd.Series):
            user_rows = pd.DataFrame([user_rows])

        # Get user history
        first_row = user_rows.iloc[0]
        history_str = first_row.get('history', '')
        history_ids = [int(x) for x in str(history_str).split()] if isinstance(history_str, str) and history_str else []

        # Parse impressions
        impressions = []
        clicks = []

        for _, row in user_rows.iterrows():
            imp_str = row['impressions']
            if not isinstance(imp_str, str):
                continue

            round_movies = []
            round_clicks = []

            for item in imp_str.split():
                parts = item.rsplit('-', 1)
                if len(parts) != 2:
                    continue

                movie_id = int(parts[0])
                clicked = parts[1] == '1'

                if movie_id in movie_lookup:
                    round_movies.append(movie_lookup[movie_id])
                else:
                    round_movies.append({
                        'movie_id': movie_id,
                        'title': '',
                        'genres_list': [],
                        'tags': [],
                    })
                round_clicks.append(clicked)

            if len(round_movies) > 0:
                impressions.append(round_movies)
                clicks.append(round_clicks)

        impressions = impressions[:N_IMPRESSIONS]
        clicks = clicks[:N_IMPRESSIONS]

        # Build user profile from history (last 5 liked movies)
        user_profile = []
        for mid in history_ids[-5:]:
            if mid in movie_lookup:
                user_profile.append(movie_lookup[mid])

        users_data.append({
            'user_id': user_id,
            'user_profile': user_profile,
            'history_ids': history_ids,
            'impressions': impressions,
            'clicks': clicks,
            'n_impressions': len(impressions),
        })

    print(f"\nParsed {len(users_data)} users")
    for u in users_data[:5]:
        print(f"  {u['user_id']}: {u['n_impressions']} impressions, "
              f"{len(u['user_profile'])} profile movies")
    if len(users_data) > 5:
        print(f"  ... and {len(users_data) - 5} more")

    return users_data, movies_df


# ─── Feature Encoder ─────────────────────────────────────────────────

class MovieFeatureEncoder:
    """
    Genre-based feature encoder for MovieLens movies.
    One-hot encodes genres as the feature vector.
    """

    def __init__(self, movies_df: pd.DataFrame):
        all_genres = set()
        for gl in movies_df['genres_list']:
            if isinstance(gl, list):
                all_genres.update(gl)
        self.all_genres = sorted(all_genres)
        self.genre_to_idx = {g: i for i, g in enumerate(self.all_genres)}
        self.n_genres = len(self.all_genres)

        print(f"\nFeatureEncoder initialized")
        print(f"  Genres: {self.n_genres}")
        print(f"  Context dim (shared model): {self.get_context_dim()}")

    def movie_features(self, movie: dict) -> np.ndarray:
        vec = np.zeros(self.n_genres)
        for g in movie.get('genres_list', []):
            if g in self.genre_to_idx:
                vec[self.genre_to_idx[g]] = 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def user_features(self, user_profile: List[dict]) -> np.ndarray:
        if len(user_profile) == 0:
            return np.zeros(self.n_genres)
        vecs = [self.movie_features(m) for m in user_profile]
        avg = np.mean(vecs, axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
        return avg

    def context_vector(self, user_feat: np.ndarray,
                       movie_feat: np.ndarray) -> np.ndarray:
        interaction = user_feat * movie_feat
        bias = np.array([1.0])
        return np.concatenate([user_feat, movie_feat, interaction, bias])

    def get_context_dim(self) -> int:
        d = self.n_genres
        return d + d + d + 1  # user + movie + interaction + bias

    def get_movie_dim(self) -> int:
        return self.n_genres


# ─── LinUCB Algorithms ───────────────────────────────────────────────

class LinUCBShared:
    def __init__(self, d: int, alpha: float = 0.3):
        self.d = d
        self.alpha = alpha
        self.A = np.eye(d)
        self.b = np.zeros(d)
        self.A_inv = np.eye(d)

    def select_arm(self, contexts: List[np.ndarray]) -> int:
        theta_hat = self.A_inv @ self.b
        ucbs = []
        for x in contexts:
            exploit = x @ theta_hat
            explore = self.alpha * np.sqrt(x @ self.A_inv @ x)
            ucbs.append(exploit + explore)
        return int(np.argmax(ucbs))

    def update(self, context: np.ndarray, reward: float):
        self.A += np.outer(context, context)
        self.b += reward * context
        self.A_inv = np.linalg.inv(self.A)


class LinUCBDisjointByGenre:
    """Per-primary-genre parameters."""
    def __init__(self, d: int, alpha: float = 0.3):
        self.d = d
        self.alpha = alpha
        self.A = defaultdict(lambda: np.eye(d))
        self.b = defaultdict(lambda: np.zeros(d))
        self.A_inv = defaultdict(lambda: np.eye(d))

    def select_arm(self, features: List[np.ndarray],
                   genres: List[str]) -> int:
        ucbs = []
        for x, g in zip(features, genres):
            Ai = self.A_inv[g]
            theta = Ai @ self.b[g]
            ucbs.append(x @ theta + self.alpha * np.sqrt(x @ Ai @ x))
        return int(np.argmax(ucbs))

    def update(self, feature: np.ndarray, genre: str, reward: float):
        self.A[genre] += np.outer(feature, feature)
        self.b[genre] += reward * feature
        self.A_inv[genre] = np.linalg.inv(self.A[genre])


class LinUCBHybrid:
    def __init__(self, k: int, d: int, alpha: float = 0.3):
        self.k = k
        self.d = d
        self.alpha = alpha
        self.A0 = np.eye(k)
        self.b0 = np.zeros(k)
        self.A0_inv = np.eye(k)
        self.Aa = defaultdict(lambda: np.eye(d))
        self.Ba = defaultdict(lambda: np.zeros((d, k)))
        self.ba = defaultdict(lambda: np.zeros(d))
        self.Aa_inv = defaultdict(lambda: np.eye(d))

    def select_arm(self, shared_feats: List[np.ndarray],
                   arm_feats: List[np.ndarray],
                   arm_ids: List[str]) -> int:
        beta_hat = self.A0_inv @ self.b0
        ucbs = []
        for z, x, aid in zip(shared_feats, arm_feats, arm_ids):
            Ai = self.Aa_inv[aid]
            Ba = self.Ba[aid]
            theta = Ai @ (self.ba[aid] - Ba @ beta_hat)
            s = (z @ self.A0_inv @ z
                 - 2 * z @ self.A0_inv @ Ba.T @ Ai @ x
                 + x @ Ai @ x
                 + x @ Ai @ Ba @ self.A0_inv @ Ba.T @ Ai @ x)
            ucbs.append(z @ beta_hat + x @ theta + self.alpha * np.sqrt(max(s, 0)))
        return int(np.argmax(ucbs))

    def update(self, z: np.ndarray, x: np.ndarray,
               arm_id: str, reward: float):
        Ai = self.Aa_inv[arm_id]
        Ba = self.Ba[arm_id]
        ba = self.ba[arm_id]

        self.A0 += Ba.T @ Ai @ Ba
        self.b0 += Ba.T @ Ai @ ba

        self.Aa[arm_id] += np.outer(x, x)
        self.Ba[arm_id] += np.outer(x, z)
        self.ba[arm_id] += reward * x

        Ai_new = np.linalg.inv(self.Aa[arm_id])
        self.Aa_inv[arm_id] = Ai_new

        Ba_new = self.Ba[arm_id]
        ba_new = self.ba[arm_id]
        self.A0 += np.outer(z, z) - Ba_new.T @ Ai_new @ Ba_new
        self.b0 += reward * z - Ba_new.T @ Ai_new @ ba_new
        self.A0_inv = np.linalg.inv(self.A0)


# ─── Experiment Runner ───────────────────────────────────────────────

def get_primary_genre(movie: dict) -> str:
    gl = movie.get('genres_list', [])
    return gl[0] if gl else 'unknown'


def run_single_user(user_data: dict, encoder: MovieFeatureEncoder,
                    algorithm: str, alpha: float, seed: int = None) -> dict:
    if seed is not None:
        np.random.seed(seed)

    user_feat = encoder.user_features(user_data['user_profile'])
    d_ctx = encoder.get_context_dim()
    d_mov = encoder.get_movie_dim()

    if algorithm == 'random':
        agent = None
    elif algorithm == 'shared':
        agent = LinUCBShared(d=d_ctx, alpha=alpha)
    elif algorithm == 'disjoint_by_genre':
        agent = LinUCBDisjointByGenre(d=d_ctx, alpha=alpha)
    elif algorithm == 'hybrid':
        agent = LinUCBHybrid(k=d_ctx, d=d_mov + 1, alpha=alpha)

    rewards = []

    for t in range(len(user_data['impressions'])):
        candidates = user_data['impressions'][t]
        click_labels = user_data['clicks'][t]
        n = len(candidates)

        if algorithm == 'random':
            chosen = np.random.randint(n)

        elif algorithm == 'shared':
            contexts = []
            for mov in candidates:
                mf = encoder.movie_features(mov)
                ctx = encoder.context_vector(user_feat, mf)
                contexts.append(ctx)
            chosen = agent.select_arm(contexts)
            agent.update(contexts[chosen], float(click_labels[chosen]))

        elif algorithm == 'disjoint_by_genre':
            features, genres = [], []
            for mov in candidates:
                mf = encoder.movie_features(mov)
                ctx = encoder.context_vector(user_feat, mf)
                features.append(ctx)
                genres.append(get_primary_genre(mov))
            chosen = agent.select_arm(features, genres)
            agent.update(features[chosen], genres[chosen], float(click_labels[chosen]))

        elif algorithm == 'hybrid':
            shared_f, arm_f, arm_ids = [], [], []
            for mov in candidates:
                mf = encoder.movie_features(mov)
                z = encoder.context_vector(user_feat, mf)
                x = np.concatenate([mf, [1.0]])
                shared_f.append(z)
                arm_f.append(x)
                arm_ids.append(get_primary_genre(mov))
            chosen = agent.select_arm(shared_f, arm_f, arm_ids)
            agent.update(shared_f[chosen], arm_f[chosen],
                        arm_ids[chosen], float(click_labels[chosen]))

        rewards.append(float(click_labels[chosen]))

    return {
        'rewards': rewards,
        'cumulative_reward': np.cumsum(rewards).tolist(),
        'total_reward': sum(rewards),
    }


def run_experiment(users_data: List[dict], encoder: MovieFeatureEncoder,
                   algorithm: str, alpha: float, n_runs: int = 10) -> dict:
    all_run_totals = []
    # Accumulate rewards per user across runs: user_index -> list of reward lists
    rewards_by_user = defaultdict(list)

    for run in range(n_runs):
        run_totals = []

        for i, user_data in enumerate(users_data):
            result = run_single_user(user_data, encoder, algorithm, alpha,
                                     seed=run * 1000 + hash(user_data['user_id']) % 1000)
            run_totals.append(result['total_reward'])
            rewards_by_user[i].append(result['rewards'])

        all_run_totals.append(np.mean(run_totals))

    # Average rewards across runs per user
    per_user_results = []
    for i, user_data in enumerate(users_data):
        run_rewards = rewards_by_user[i]
        max_len = max(len(r) for r in run_rewards)
        reward_matrix = np.zeros((len(run_rewards), max_len))
        for j, r in enumerate(run_rewards):
            reward_matrix[j, :len(r)] = r
        avg_rewards = np.mean(reward_matrix, axis=0).tolist()
        total = sum(avg_rewards)

        per_user_results.append({
            'user_id': user_data['user_id'],
            'model': algorithm,
            'rewards': avg_rewards,
            'cumulative_reward': total,
            'total_rounds': max_len,
            'ctr': total / max_len if max_len > 0 else 0,
        })

    return {
        'algorithm': algorithm,
        'alpha': alpha,
        'mean_total': np.mean(all_run_totals),
        'std_total': np.std(all_run_totals),
        'all_results': per_user_results,
    }


def tune_alpha(users_data, encoder, algorithm, alpha_values, n_runs=10):
    best_alpha = None
    best_reward = -1

    for alpha in alpha_values:
        r = run_experiment(users_data, encoder, algorithm, alpha, n_runs)
        print(f"  {algorithm} alpha={alpha:.2f}: reward={r['mean_total']:.2f} +/- {r['std_total']:.2f}")
        if r['mean_total'] > best_reward:
            best_reward = r['mean_total']
            best_alpha = alpha

    print(f"  ** Best alpha for {algorithm}: {best_alpha} (reward={best_reward:.2f})")
    return best_alpha


# ─── Plotting ─────────────────────────────────────────────────────────

def plot_cumulative_rewards(results_dict: dict, save_path: str = None):
    colors = {
        'random':            '#AAAAAA',
        'shared':            '#1f77b4',
        'disjoint_by_genre': '#ff7f0e',
        'hybrid':            '#d62728',
    }
    llm_color_cycle = ['#2ca02c', '#9467bd', '#e377c2', '#17becf', '#bcbd22']

    fig, ax = plt.subplots(figsize=(8, 5))
    num_users = None
    llm_idx = 0

    for name, result in results_dict.items():
        all_results = result['all_results']
        num_users = len(all_results)

        if name in colors:
            color = colors[name]
        else:
            color = llm_color_cycle[llm_idx % len(llm_color_cycle)]
            llm_idx += 1

        max_rounds = max(len(r['rewards']) for r in all_results)
        reward_matrix = np.full((len(all_results), max_rounds), np.nan)

        for i, r in enumerate(all_results):
            rewards = r['rewards']
            reward_matrix[i, :len(rewards)] = rewards

        num_users_per_step = np.sum(~np.isnan(reward_matrix), axis=0)
        cum_reward_matrix = np.nancumsum(reward_matrix, axis=1)
        timesteps = np.arange(1, max_rounds + 1)
        cum_avg_matrix = cum_reward_matrix / timesteps[np.newaxis, :]
        avg_cum_reward = np.nanmean(cum_avg_matrix, axis=0)
        se_cum = np.nanstd(cum_avg_matrix, axis=0) / np.sqrt(num_users_per_step)

        label = f"{name} (a={result['alpha']:.2f})" if result.get('alpha', 0) > 0 else name
        ax.plot(timesteps, avg_cum_reward, color=color, linewidth=2, label=label)
        ax.fill_between(timesteps, avg_cum_reward - se_cum, avg_cum_reward + se_cum,
                        alpha=0.15, color=color)

    ax.set_xlabel('Round')
    ax.set_ylabel('Cumulative Average Reward (CTR)')
    ax.set_title(f'MovieLens — Cumulative Average Reward | {num_users} users')
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_combined_with_llm(linucb_results: dict, llm_json_paths: dict, save_path: str = None):
    """Plot LinUCB baselines alongside LLM results on the same graph."""
    combined = dict(linucb_results)

    for model_name, json_path in llm_json_paths.items():
        with open(json_path, 'r') as f:
            llm_results = json.load(f)

        combined[model_name] = {
            'algorithm': model_name,
            'alpha': 0.0,
            'mean_total': np.mean([r['cumulative_reward'] for r in llm_results]),
            'std_total': np.std([r['cumulative_reward'] for r in llm_results]),
            'all_results': llm_results,
        }

    plot_cumulative_rewards(combined, save_path=save_path)


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    # 1. Load data
    users_data, movies_df = load_data(USERS_PKL, MOVIES_PKL)

    # 2. Initialize encoder (genre-based features)
    encoder = MovieFeatureEncoder(movies_df)

    # 3. Random baseline
    print("\n" + "=" * 60)
    print("RANDOM BASELINE")
    print("=" * 60)
    random_result = run_experiment(users_data, encoder, 'random', alpha=0.0, n_runs=N_RUNS)
    print(f"Random: {random_result['mean_total']:.2f} +/- {random_result['std_total']:.2f}")

    # 4. Tune alpha for each algorithm
    print("\n" + "=" * 60)
    print("TUNING ALPHA")
    print("=" * 60)

    best_results = {'random': random_result}

    for algo in ['shared', 'disjoint_by_genre', 'hybrid']:
        print(f"\n--- {algo} ---")
        best_alpha = tune_alpha(users_data, encoder, algo, ALPHA_VALUES, n_runs=N_RUNS)
        best_result = run_experiment(users_data, encoder, algo, best_alpha, n_runs=N_RUNS)
        best_results[algo] = best_result

    # 5. Summary table
    print("\n" + "=" * 60)
    print("FINAL RESULTS (best alpha per algorithm)")
    print("=" * 60)
    print(f"{'Algorithm':<25} {'Alpha':<8} {'Mean Reward':<15} {'Std':<10}")
    print("-" * 58)
    for algo, r in best_results.items():
        print(f"{algo:<25} {r['alpha']:<8.2f} {r['mean_total']:<15.2f} {r['std_total']:<10.2f}")

    # 6. Save JSON results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for algo, r in best_results.items():
        json_path = os.path.join(RESULTS_DIR, f"linucb_{algo}_{timestamp}.json")
        save_data = {
            'algorithm': algo,
            'alpha': r['alpha'],
            'mean_total': r['mean_total'],
            'std_total': r['std_total'],
            'all_results': r['all_results'],
        }
        with open(json_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"Saved {algo} results to {json_path}")

    # 7. Plot
    print("\nGenerating plot...")
    plot_results = {
        'random': best_results['random'],
        'hybrid': best_results['hybrid'],
    }
    plot_cumulative_rewards(plot_results, save_path='linucb_cumulative_rewards.png')

    return best_results


if __name__ == '__main__':
    results = main()
