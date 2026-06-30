import numpy as np
import matplotlib.pyplot as plt

NUM_RUNS = 2000
NUM_ARMS = 10
TIMESTEPS = 1000


def _update_value(q, qstar, n):
    r = np.random.normal(qstar, 1)
    if n == 0:
        n = 1
    q += (1 / n) * (r - q)
    return q, r


def UCB(Q, N, Q_OPT):
    rewards = np.zeros(TIMESTEPS)
    # Initialization
    for i in range(NUM_ARMS):
        Q[i], r = _update_value(Q[i].item(), Q_OPT[i].item(), 1)
        N[i] += 1

    for ts in range(TIMESTEPS):
        ucb_vec = Q + np.sqrt(2*np.log(ts) / N)
        chosen_i = np.argmax(ucb_vec)

        Q[chosen_i], r = _update_value(
            Q[chosen_i].item(), Q_OPT[chosen_i].item(), N[chosen_i].item()
        )
        N[chosen_i] += 1
        rewards[ts] += r
    return rewards


def egreedy(Q, N, Q_OPT, e):
    rewards = np.zeros(TIMESTEPS)
    for ts in range(TIMESTEPS):
        if np.random.rand() < e:
            i = np.random.randint(NUM_ARMS)
        else:
            max_value = max(Q)
            idxs = np.argwhere(Q == max_value)
            idxs = idxs.flatten()
            i = np.random.choice(idxs)

        Q[i], r = _update_value(Q[i].item(), Q_OPT[i].item(), N[i].item())
        N[i] += 1
        rewards[ts] += r
    return rewards


if __name__ == "__main__":
    timesteps = range(1, TIMESTEPS + 1)

    eg_rewards = np.zeros(TIMESTEPS)
    ucb_rewards = np.zeros(TIMESTEPS)
    for run in range(1, NUM_RUNS + 1):
        Q = np.zeros(NUM_ARMS)
        Q_OPT = np.random.uniform(-5, 5, NUM_ARMS)
        noise = np.random.normal(0, 1, NUM_ARMS)
        Q_OPT += noise
        N = np.zeros(NUM_ARMS)
        eg_rewards += egreedy(Q.copy(), N.copy(), Q_OPT.copy(), 0.01)
        ucb_rewards += UCB(Q.copy(), N.copy(), Q_OPT.copy())

    eg_rewards /= NUM_RUNS
    ucb_rewards /= NUM_RUNS

    plt.plot(timesteps, ucb_rewards, label="UCB")
    plt.plot(timesteps, eg_rewards, label="e-greedy")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.show()
