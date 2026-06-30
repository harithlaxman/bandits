"""Per-user linear contextual bandits over the phi(state, movie) features.

One model instance per user (warm-started from cold-start ratings), mirroring how
the LLM harness only ever conditions on a single user at a time. Reward target is
the raw label in {1, 0, -1}; "click" (for CTR/regret) is reward == 1.

Both models maintain A^{-1} directly via Sherman-Morrison rank-1 updates; the
feature dim is tiny (PHI_DIM ~ 9) so this is essentially free.
"""
import numpy as np


class _LinModel:
    def __init__(self, d: int, lam: float):
        self.d = d
        self.Ainv = np.eye(d) / lam     # A = lam*I  =>  A^{-1} = I/lam
        self.b = np.zeros(d)

    def theta(self) -> np.ndarray:
        return self.Ainv @ self.b

    def update(self, phi: np.ndarray, reward: float) -> None:
        Ap = self.Ainv @ phi
        denom = 1.0 + float(phi @ Ap)
        self.Ainv -= np.outer(Ap, Ap) / denom
        self.b += reward * phi

    def scores(self, phis: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError


class LinUCB(_LinModel):
    def __init__(self, d: int, lam: float, alpha: float):
        super().__init__(d, lam)
        self.alpha = alpha

    def scores(self, phis: list[np.ndarray]) -> np.ndarray:
        th = self.theta()
        out = np.empty(len(phis))
        for i, p in enumerate(phis):
            mean = float(th @ p)
            var = float(p @ self.Ainv @ p)
            out[i] = mean + self.alpha * np.sqrt(max(var, 0.0))
        return out


class LinTS(_LinModel):
    """Linear Thompson Sampling: one theta sampled per decision, shared across arms."""

    def __init__(self, d: int, lam: float, v: float, rng: np.random.Generator):
        super().__init__(d, lam)
        self.v = v
        self.rng = rng

    def scores(self, phis: list[np.ndarray]) -> np.ndarray:
        cov = self.v ** 2 * self.Ainv
        cov = 0.5 * (cov + cov.T)  # symmetrize against float drift
        theta_tilde = self.rng.multivariate_normal(self.theta(), cov)
        return np.array([float(theta_tilde @ p) for p in phis])


def make_model(algo: str, d: int, alpha: float, lam: float, v: float,
               rng: np.random.Generator) -> _LinModel:
    if algo == "linucb":
        return LinUCB(d, lam, alpha)
    if algo == "lints":
        return LinTS(d, lam, v, rng)
    raise ValueError(f"unknown algo: {algo!r} (valid: 'linucb', 'lints')")
