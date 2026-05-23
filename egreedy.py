import random

import numpy as np
import matplotlib.pyplot as plt

NUM_RUNS = 2000
NUM_ARMS = 10
TIMESTEPS = 1000

EPS = [0, 0.01, 0.1]


class arm:
    def __init__(self) -> None:
        self.curr_value = 0
        self.chosen_times = 0
        # randomly fixing an optimal value
        self.opt_value = np.random.normal(0, 1)

    def update_value(self, reward):
        self.chosen_times += 1
        self.curr_value += (1 / self.chosen_times) * (reward - self.curr_value)

    def get_true_reward(self):
        return np.random.normal(self.opt_value, 1)

def greedy_choose(values):
    max_value = np.max(values)
    idx = np.argwhere(values == max_value)
    return idx.flatten()


if __name__ == "__main__":
    for eps in EPS:
        run_details = np.zeros(TIMESTEPS)
        print(f"e-greedy: e = {eps}")
        for run in range(1, NUM_RUNS + 1):
            arms = []
            values = np.array([])
            for _ in range(NUM_ARMS):
                a = arm()
                arms.append(a)
                values = np.append(values, a.curr_value)

            for ts in range(TIMESTEPS):
                idx = greedy_choose(values)
                i = None
                if np.random.rand() > eps:
                    if len(idx) > 1:
                        # randomly choose between actions with same max value
                        i = idx[random.randint(0, len(idx) - 1)]
                    else:
                        i = idx[0]
                else:
                    i = np.random.randint(NUM_ARMS)

                # update value for that action
                rt = arms[i].get_true_reward()
                arms[i].update_value(rt)
                values[i] = arms[i].curr_value

                # store reward to plot later
                run_details[ts] += rt

        run_details /= NUM_RUNS
        timesteps = range(1, TIMESTEPS + 1)
        plt.plot(timesteps, run_details, label=f"ε = {eps}")

    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.show()
