import numpy as np

D = 10
alpha = 1


class LinUCBDisjointArm:
    def __init__(self) -> None:
        self.A = np.identity(D)
        self.b = np.zeros([D, 1])

    def calc_ucb(self, xa):
        xa = xa.reshape([-1, 1])  # assuming xa's shape is (1, d)
        Ainv = np.linalg.inv(self.A)
        theta = Ainv @ self.b
        ucb = theta.T @ xa + alpha * np.sqrt(xa.T @ Ainv @ xa)
        return float(ucb)

    def update(self, xa, reward):
        xa = xa.reshape([-1, 1])
        self.A += xa @ xa.T
        self.b += reward * xa


class LinUCBDisjointPolicy:
    def __init__(self, K) -> None:
        self.arms = [LinUCBDisjointArm() for _ in range(K)]

    def choose_best_arm(self, x):
        Q = [arm.calc_ucb(x) for arm in self.arms]
        max_score = max(Q)
        idxs = np.argwhere(Q == max_score)
        idxs = idxs.flatten()
        best_arm = np.random.choice(idxs)

        return best_arm

class LinUCBHybridArm:
    def __init__(self) -> None:
        self.A = np.identity(D)
        self.b = np.zeros([D, 1])
