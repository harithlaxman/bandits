"""
LinUCB Baselines — Integrated with your MIND preprocessing
============================================================

This script loads your two pickle files:
  1. users_with_embeddings_50.pkl  (filtered_df)
  2. news_with_embeddings_50.pkl   (filtered_news_df)

and runs LinUCB variants as baselines for comparison with your LLM-as-bandit.

Usage:
    python linucb_mind.py

Make sure the two pkl files are in the same directory (or update the paths below).
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIG — update these paths to match your setup
# =============================================================================

USERS_PKL = "data/users_with_embeddings_50.pkl"
NEWS_PKL = "data/news_with_embeddings_50.pkl"
RESULTS_DIR = "results"

N_IMPRESSIONS = 50       # how many impressions per user to use
N_CANDIDATES = 5         # candidates per impression (should be 5 from your preprocessing)
N_RUNS = 20              # repeated runs for variance estimation (random has variance, UCB is deterministic)

# Alpha values to sweep for each algorithm
ALPHA_VALUES = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 1.5]


# =============================================================================
# DATA LOADING — parses your exact pickle format
# =============================================================================

def load_data(users_pkl_path: str, news_pkl_path: str):
    """
    Load and parse your preprocessed MIND data.
    
    Returns:
        users_data: list of dicts, one per user, each with:
            - 'user_id': str
            - 'user_embedding': np.ndarray (sentence transformer embedding)
            - 'history_news_ids': list of str
            - 'impressions': list of lists of article dicts
            - 'clicks': list of lists of booleans
        news_df: the filtered news DataFrame
    """
    print("Loading data...")
    users_df = pd.read_pickle(users_pkl_path)
    news_df = pd.read_pickle(news_pkl_path)
    
    print(f"  Users: {len(users_df.index.unique())}")
    print(f"  News articles: {len(news_df)}")
    print(f"  News columns: {list(news_df.columns)}")
    print(f"  Categories: {news_df['category'].nunique()}")
    print(f"  Subcategories: {news_df['sub_category'].nunique()}")
    
    # Build lookup for news metadata
    # news_df is indexed by news_id
    news_lookup = {}
    for news_id, row in news_df.iterrows():
        news_lookup[news_id] = {
            'news_id': news_id,
            'category': row.get('category', ''),
            'sub_category': row.get('sub_category', ''),
            'title': row.get('title', ''),
            'abstract': row.get('abstract', ''),
        }
        # Add embeddings if available
        if 'embedding' in row.index and row['embedding'] is not None:
            news_lookup[news_id]['embedding'] = np.array(row['embedding'])
        if 'pca_embedding' in row.index and row['pca_embedding'] is not None:
            news_lookup[news_id]['pca_embedding'] = np.array(row['pca_embedding'])
    
    # Parse each user's data
    users_data = []
    user_ids = users_df.index.unique()
    
    for user_id in user_ids:
        user_rows = users_df.loc[user_id]
        if isinstance(user_rows, pd.Series):
            user_rows = pd.DataFrame([user_rows])
        
        # Get user history (from first row — same across rows for this user)
        first_row = user_rows.iloc[0]
        history_str = first_row.get('history', '')
        history_ids = history_str.split() if isinstance(history_str, str) else []
        
        # Parse impressions
        impressions = []
        clicks = []
        
        for _, row in user_rows.iterrows():
            imp_str = row['impressions']
            if not isinstance(imp_str, str):
                continue
            
            articles_raw = imp_str.split()
            round_articles = []
            round_clicks = []
            
            for item in articles_raw:
                # Format: "NEWSID-0" or "NEWSID-1"
                parts = item.rsplit('-', 1)
                if len(parts) != 2:
                    continue
                
                news_id = parts[0]
                clicked = parts[1] == '1'
                
                if news_id in news_lookup:
                    round_articles.append(news_lookup[news_id])
                    round_clicks.append(clicked)
                else:
                    # Article not in news_df — skip or use placeholder
                    round_articles.append({
                        'news_id': news_id,
                        'category': 'unknown',
                        'sub_category': 'unknown',
                        'title': '',
                        'abstract': '',
                    })
                    round_clicks.append(clicked)
            
            if len(round_articles) > 0:
                impressions.append(round_articles)
                clicks.append(round_clicks)
        
        # Limit to N_IMPRESSIONS
        impressions = impressions[:N_IMPRESSIONS]
        clicks = clicks[:N_IMPRESSIONS]
        
        # Build user profile from history
        user_profile = []
        for nid in history_ids[:5]:  # first 5 history articles (matches your LLM setup)
            if nid in news_lookup:
                user_profile.append(news_lookup[nid])
        
        users_data.append({
            'user_id': user_id,
            'user_profile': user_profile,
            'history_ids': history_ids,
            'impressions': impressions,
            'clicks': clicks,
            'n_impressions': len(impressions),
        })
    
    print(f"\nParsed {len(users_data)} users")
    for u in users_data:
        print(f"  {u['user_id']}: {u['n_impressions']} impressions, "
              f"{len(u['user_profile'])} profile articles")
    
    return users_data, news_df


# =============================================================================
# FEATURE ENCODER
# =============================================================================

class MINDFeatureEncoder:
    """
    Feature encoder tailored to your MIND data.
    
    Two modes:
    1. Categorical features: one-hot category + subcategory (sparse but interpretable)
    2. Embedding features: use your precomputed sentence transformer embeddings (dense)
    
    You can choose which to use. Embeddings are richer but categorical features
    make the bandit more comparable to the LLM (which sees category/subcategory text).
    """
    
    def __init__(self, news_df: pd.DataFrame, feature_mode: str = 'embedding'):
        """
        Args:
            news_df: your filtered_news_df (indexed by news_id)
            feature_mode: 'categorical', 'embedding', or 'both'
        """
        self.feature_mode = feature_mode
        
        # Build category/subcategory mappings
        all_cats = sorted(news_df['category'].dropna().unique())
        all_subcats = sorted(news_df['sub_category'].dropna().unique())
        
        self.cat_to_idx = {c: i for i, c in enumerate(all_cats)}
        self.subcat_to_idx = {s: i for i, s in enumerate(all_subcats)}
        self.n_cats = len(all_cats)
        self.n_subcats = len(all_subcats)
        
        # Embedding dim (from PCA)
        if 'pca_embedding' in news_df.columns:
            sample = news_df['pca_embedding'].dropna().iloc[0]
            self.embedding_dim = len(sample)
        else:
            self.embedding_dim = 0
        
        print(f"\nFeatureEncoder initialized (mode={feature_mode})")
        print(f"  Categories: {self.n_cats}, Subcategories: {self.n_subcats}")
        print(f"  Categorical feature dim: {self.n_cats + self.n_subcats}")
        if self.embedding_dim:
            print(f"  Embedding dim (PCA): {self.embedding_dim}")
        print(f"  Context dim (shared model): {self.get_context_dim()}")
    
    def _cat_features(self, category: str, sub_category: str) -> np.ndarray:
        """One-hot encode category + subcategory."""
        cat_vec = np.zeros(self.n_cats)
        subcat_vec = np.zeros(self.n_subcats)
        if category in self.cat_to_idx:
            cat_vec[self.cat_to_idx[category]] = 1.0
        if sub_category in self.subcat_to_idx:
            subcat_vec[self.subcat_to_idx[sub_category]] = 1.0
        return np.concatenate([cat_vec, subcat_vec])
    
    def article_features(self, article: dict) -> np.ndarray:
        """Get feature vector for a single article."""
        if self.feature_mode == 'categorical':
            return self._cat_features(article.get('category', ''),
                                       article.get('sub_category', ''))
        elif self.feature_mode == 'embedding':
            if 'pca_embedding' in article:
                return np.array(article['pca_embedding'])
            return np.zeros(self.embedding_dim)
        elif self.feature_mode == 'both':
            cat_feat = self._cat_features(article.get('category', ''),
                                           article.get('sub_category', ''))
            if 'pca_embedding' in article:
                emb = np.array(article['pca_embedding'])
            else:
                emb = np.zeros(self.embedding_dim)
            return np.concatenate([cat_feat, emb])
    
    def user_features(self, user_profile: List[dict]) -> np.ndarray:
        """
        Build user preference vector from their profile articles.
        Same dim as article_features — normalized distribution.
        """
        if len(user_profile) == 0:
            return np.zeros(self._base_dim())
        
        vecs = [self.article_features(a) for a in user_profile]
        # Average to get preference distribution
        avg = np.mean(vecs, axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
        return avg
    
    def context_vector(self, user_feat: np.ndarray, 
                       article_feat: np.ndarray) -> np.ndarray:
        """
        Build full context vector for shared/hybrid models.
        [user_features, article_features, interaction, bias]
        """
        interaction = user_feat * article_feat
        bias = np.array([1.0])
        return np.concatenate([user_feat, article_feat, interaction, bias])
    
    def _base_dim(self) -> int:
        if self.feature_mode == 'categorical':
            return self.n_cats + self.n_subcats
        elif self.feature_mode == 'embedding':
            return self.embedding_dim
        elif self.feature_mode == 'both':
            return self.n_cats + self.n_subcats + self.embedding_dim
    
    def get_context_dim(self) -> int:
        d = self._base_dim()
        return d + d + d + 1  # user + article + interaction + bias
    
    def get_article_dim(self) -> int:
        return self._base_dim()


# =============================================================================
# LinUCB ALGORITHMS
# =============================================================================

class LinUCBShared:
    """Single shared parameter vector. Best when arms never repeat."""
    
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


class LinUCBDisjointByCategory:
    """Per-category parameters. Middle ground for non-repeating arms."""
    
    def __init__(self, d: int, alpha: float = 0.3):
        self.d = d
        self.alpha = alpha
        self.A = defaultdict(lambda: np.eye(d))
        self.b = defaultdict(lambda: np.zeros(d))
        self.A_inv = defaultdict(lambda: np.eye(d))
    
    def select_arm(self, features: List[np.ndarray], 
                   categories: List[str]) -> int:
        ucbs = []
        for x, cat in zip(features, categories):
            Ai = self.A_inv[cat]
            theta = Ai @ self.b[cat]
            ucbs.append(x @ theta + self.alpha * np.sqrt(x @ Ai @ x))
        return int(np.argmax(ucbs))
    
    def update(self, feature: np.ndarray, category: str, reward: float):
        self.A[category] += np.outer(feature, feature)
        self.b[category] += reward * feature
        self.A_inv[category] = np.linalg.inv(self.A[category])


class LinUCBHybrid:
    """
    Full Algorithm 2 from Li et al. 2010.
    Arm identity grouped by category (since articles don't repeat).
    """
    
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


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_single_user(user_data: dict, encoder: MINDFeatureEncoder,
                    algorithm: str, alpha: float, seed: int = None) -> dict:
    """
    Run one algorithm on one user.
    
    Returns dict with rewards list and cumulative reward.
    """
    if seed is not None:
        np.random.seed(seed)
    
    user_feat = encoder.user_features(user_data['user_profile'])
    d_ctx = encoder.get_context_dim()
    d_art = encoder.get_article_dim()
    
    # Initialize agent
    if algorithm == 'random':
        agent = None
    elif algorithm == 'shared':
        agent = LinUCBShared(d=d_ctx, alpha=alpha)
    elif algorithm == 'disjoint_by_cat':
        agent = LinUCBDisjointByCategory(d=d_ctx, alpha=alpha)
    elif algorithm == 'hybrid':
        agent = LinUCBHybrid(k=d_ctx, d=d_art + 1, alpha=alpha)
    
    rewards = []
    
    for t in range(len(user_data['impressions'])):
        candidates = user_data['impressions'][t]
        click_labels = user_data['clicks'][t]
        n = len(candidates)
        
        if algorithm == 'random':
            chosen = np.random.randint(n)
        
        elif algorithm == 'shared':
            contexts = []
            for art in candidates:
                af = encoder.article_features(art)
                ctx = encoder.context_vector(user_feat, af)
                contexts.append(ctx)
            chosen = agent.select_arm(contexts)
            agent.update(contexts[chosen], float(click_labels[chosen]))
        
        elif algorithm == 'disjoint_by_cat':
            features, cats = [], []
            for art in candidates:
                af = encoder.article_features(art)
                ctx = encoder.context_vector(user_feat, af)
                features.append(ctx)
                cats.append(art.get('category', 'unknown'))
            chosen = agent.select_arm(features, cats)
            agent.update(features[chosen], cats[chosen], float(click_labels[chosen]))
        
        elif algorithm == 'hybrid':
            shared_f, arm_f, arm_ids = [], [], []
            for art in candidates:
                af = encoder.article_features(art)
                z = encoder.context_vector(user_feat, af)
                x = np.concatenate([af, [1.0]])
                shared_f.append(z)
                arm_f.append(x)
                arm_ids.append(art.get('category', 'unknown'))
            chosen = agent.select_arm(shared_f, arm_f, arm_ids)
            agent.update(shared_f[chosen], arm_f[chosen], 
                        arm_ids[chosen], float(click_labels[chosen]))
        
        rewards.append(float(click_labels[chosen]))
    
    return {
        'rewards': rewards,
        'cumulative_reward': np.cumsum(rewards).tolist(),
        'total_reward': sum(rewards),
    }


def run_experiment(users_data: List[dict], encoder: MINDFeatureEncoder,
                   algorithm: str, alpha: float, n_runs: int = 10) -> dict:
    """
    Run algorithm across all users, averaged over multiple runs.
    
    Returns results in TWO formats:
    1. 'all_results': list of per-user result dicts (same format as your LLM code)
       so you can pass it directly to plot_learning_curves()
    2. Summary stats: mean_total, std_total, etc.
    """
    # For deterministic algorithms (LinUCB), n_runs > 1 only matters for 'random'
    # but we keep the loop for consistency
    
    all_run_totals = []
    
    # We'll store the last run's per-user results for the LLM-compatible format
    per_user_results = []
    
    for run in range(n_runs):
        run_totals = []
        run_per_user = []
        
        for user_data in users_data:
            result = run_single_user(user_data, encoder, algorithm, alpha, 
                                     seed=run * 1000 + hash(user_data['user_id']) % 1000)
            run_totals.append(result['total_reward'])
            run_per_user.append({
                'user_id': user_data['user_id'],
                'model': algorithm,
                'rewards': result['rewards'],
                'cumulative_reward': int(result['total_reward']),
                'total_rounds': len(result['rewards']),
                'ctr': result['total_reward'] / len(result['rewards']) if result['rewards'] else 0,
            })
        
        all_run_totals.append(np.mean(run_totals))
        per_user_results = run_per_user  # keep last run
    
    return {
        'algorithm': algorithm,
        'alpha': alpha,
        'mean_total': np.mean(all_run_totals),
        'std_total': np.std(all_run_totals),
        'all_results': per_user_results,  # LLM-compatible format
    }


# =============================================================================
# ALPHA TUNING
# =============================================================================

def tune_alpha(users_data, encoder, algorithm, alpha_values, n_runs=10):
    """Find best alpha for a given algorithm."""
    best_alpha = None
    best_reward = -1
    results = []
    
    for alpha in alpha_values:
        r = run_experiment(users_data, encoder, algorithm, alpha, n_runs)
        results.append(r)
        print(f"  {algorithm} alpha={alpha:.2f}: reward={r['mean_total']:.2f} ± {r['std_total']:.2f}")
        if r['mean_total'] > best_reward:
            best_reward = r['mean_total']
            best_alpha = alpha
    
    print(f"  ** Best alpha for {algorithm}: {best_alpha} (reward={best_reward:.2f})")
    return best_alpha, results


# =============================================================================
# PLOTTING
# =============================================================================

def plot_cumulative_rewards(results_dict: dict, save_path: str = None):
    """
    Plot cumulative average reward (CTR) over time — SAME metric as your LLM code.
    """
    # High-contrast, colorblind-friendly palette
    colors = {
        'random':          '#AAAAAA',   # grey
        'shared':          '#1f77b4',   # blue
        'disjoint_by_cat': '#ff7f0e',   # orange
        'hybrid':          '#d62728',   # red
    }
    # LLM models get distinct colors that don't clash with the above
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
        
        label = f"{name} (α={result['alpha']:.2f})" if result.get('alpha', 0) > 0 else name
        ax.plot(timesteps, avg_cum_reward, color=color, linewidth=2, label=label)
        ax.fill_between(timesteps, avg_cum_reward - se_cum, avg_cum_reward + se_cum,
                        alpha=0.15, color=color)
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Cumulative Average Reward (CTR)')
    ax.set_title(f'Cumulative Average Reward | {num_users} users')
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_combined_with_llm(linucb_results: dict, llm_json_paths: dict, save_path: str = None):
    """
    Plot LinUCB baselines alongside your LLM results on the SAME graph.
    
    Args:
        linucb_results: {algo_name: result_dict} from run_experiment
        llm_json_paths: {model_name: path_to_json} from your slm.py output
                        e.g. {'gpt-4.1': 'results/gpt-4.1_20250101.json',
                               'llama3.1': 'results/llama3.1_latest_20250101.json'}
    """
    import json
    
    # Merge LLM results into the same format
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


# =============================================================================
# MAIN
# =============================================================================

def main():
    # 1. Load data
    users_data, news_df = load_data(USERS_PKL, NEWS_PKL)
    
    # 2. Initialize encoder
    #    Using 'embedding' — your PCA'd sentence-transformer vectors (100d)
    #    Alternatives: 'categorical' (one-hot cat+subcat) or 'both'
    encoder = MINDFeatureEncoder(news_df, feature_mode='embedding')
    
    # 3. Run random baseline
    print("\n" + "="*60)
    print("RANDOM BASELINE")
    print("="*60)
    random_result = run_experiment(users_data, encoder, 'random', alpha=0.0, n_runs=N_RUNS)
    print(f"Random: {random_result['mean_total']:.2f} ± {random_result['std_total']:.2f}")
    
    # 4. Tune alpha for each algorithm
    print("\n" + "="*60)
    print("TUNING ALPHA")
    print("="*60)
    
    best_results = {'random': random_result}
    
    for algo in ['shared', 'disjoint_by_cat', 'hybrid']:
        print(f"\n--- {algo} ---")
        best_alpha, _ = tune_alpha(users_data, encoder, algo, ALPHA_VALUES, n_runs=N_RUNS)
        
        # Run with best alpha
        best_result = run_experiment(users_data, encoder, algo, best_alpha, n_runs=N_RUNS)
        best_results[algo] = best_result
    
    # 5. Summary table
    print("\n" + "="*60)
    print("FINAL RESULTS (best alpha per algorithm)")
    print("="*60)
    print(f"{'Algorithm':<25} {'Alpha':<8} {'Mean Reward':<15} {'Std':<10}")
    print("-"*58)
    for algo, r in best_results.items():
        print(f"{algo:<25} {r['alpha']:<8.2f} {r['mean_total']:<15.2f} {r['std_total']:<10.2f}")
    
    # 6. Save JSON results for ALL variants (so you can re-plot anytime)
    import json
    from datetime import datetime
    
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
    
    # 7. Plot only random + hybrid
    print("\nGenerating plot (random + hybrid)...")
    plot_results = {
        'random': best_results['random'],
        'hybrid': best_results['hybrid'],
    }
    plot_cumulative_rewards(plot_results, save_path='linucb_cumulative_rewards.png')
    
    # =================================================================
    # TO PLOT ALONGSIDE YOUR LLM RESULTS:
    # =================================================================
    # After running slm.py, you'll have JSON files in results/.
    # Use plot_combined_with_llm() to put everything on one graph:
    #
    plot_combined_with_llm(
        linucb_results={'hybrid': best_results['hybrid']},
        llm_json_paths={
            'gpt-4.1': 'results/gpt-4.1_0.json',
            'llama3.1': 'results/llama3.1_latest_0.json',
            'ministral-3:8b': 'results/ministral-3-8b.json',
            'granite4:tiny-h': 'results/granite4tiny-h.json',
        },
        save_path='all_methods_comparison.png'
    )
    #
    # Or load saved JSONs later:
    #
    #   with open('results/linucb_hybrid_XXXXXXXX.json') as f:
    #       hybrid_data = json.load(f)
    #   plot_cumulative_rewards({'hybrid': hybrid_data}, save_path='...')
    # =================================================================
    
    return best_results


if __name__ == '__main__':
    results = main()