import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ----------------------------
# Environment: 5x5 Gridworld with slip
# ----------------------------
class Gridworld5x5:
    """
    5x5 gridworld:
      - start = (0,0)
      - goal  = (4,4)
      - reward = -1 per step, +10 upon reaching goal (and episode ends)
      - slip probability: with prob slip, executed action becomes a random action
      - episode ends at goal OR when max_steps is reached
    """
    def __init__(self, slip=0.2, max_steps=200, seed=0):
        self.n = 5
        self.n_states = self.n * self.n
        self.n_actions = 4  # 0=up,1=right,2=down,3=left
        self.slip = float(slip)
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.t = 0
        self.s = 0  # (0,0)
        return self.s

    def _to_rc(self, s):
        return divmod(s, self.n)

    def _to_s(self, r, c):
        return r * self.n + c

    def step(self, a):
        self.t += 1

        # Slip: execute a random action with probability slip
        if self.rng.random() < self.slip:
            a_exec = int(self.rng.integers(0, self.n_actions))
        else:
            a_exec = int(a)

        r, c = self._to_rc(self.s)

        if a_exec == 0:   # up
            r = max(0, r - 1)
        elif a_exec == 1: # right
            c = min(self.n - 1, c + 1)
        elif a_exec == 2: # down
            r = min(self.n - 1, r + 1)
        elif a_exec == 3: # left
            c = max(0, c - 1)

        s_next = self._to_s(r, c)

        done = (s_next == self.n_states - 1) or (self.t >= self.max_steps)
        reward = 10.0 if (s_next == self.n_states - 1) else -1.0

        self.s = s_next
        return s_next, reward, done


# ----------------------------
# Different algorithms
# ----------------------------
def epsilon_greedy(Q, s, eps, rng):
    """Choose action epsilon-greedily w.r.t. Q[s]. Break ties randomly."""
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    row = Q[s]
    maxv = np.max(row)
    candidates = np.flatnonzero(row == maxv)
    return int(rng.choice(candidates))

def run_sarsa(env_seed, episodes, alpha, gamma, eps, slip, max_steps):
    env = Gridworld5x5(slip=slip, max_steps=max_steps, seed=env_seed)
    rng = np.random.default_rng(env_seed + 1000)
    Q = np.zeros((env.n_states, env.n_actions), dtype=np.float64)

    returns = np.zeros(episodes, dtype=np.float64)

    for ep in range(episodes):
        s = env.reset()
        a = epsilon_greedy(Q, s, eps, rng)
        G = 0.0

        while True:
            s2, r, done = env.step(a)
            G += r

            if done:
                td_target = r
                Q[s, a] += alpha * (td_target - Q[s, a])
                break
            else:
                a2 = epsilon_greedy(Q, s2, eps, rng)
                td_target = r + gamma * Q[s2, a2]
                Q[s, a] += alpha * (td_target - Q[s, a])
                s, a = s2, a2

        returns[ep] = G

    return returns

def run_q_learning(env_seed, episodes, alpha, gamma, eps, slip, max_steps):
    env = Gridworld5x5(slip=slip, max_steps=max_steps, seed=env_seed)
    rng = np.random.default_rng(env_seed + 2000)
    Q = np.zeros((env.n_states, env.n_actions), dtype=np.float64)

    returns = np.zeros(episodes, dtype=np.float64)

    for ep in range(episodes):
        s = env.reset()
        G = 0.0

        while True:
            a = epsilon_greedy(Q, s, eps, rng)
            s2, r, done = env.step(a)
            G += r

            if done:
                td_target = r
            else:
                td_target = r + gamma * np.max(Q[s2])

            Q[s, a] += alpha * (td_target - Q[s, a])

            if done:
                break
            s = s2

        returns[ep] = G

    return returns

def run_dyna_q(env_seed, episodes, alpha, gamma, eps, slip, max_steps, k_plan):
    """
    Dyna-Q (simple):
      - real step: Q-learning update
      - model: store last observed (r, s') for each (s, a)
      - planning: sample seen (s,a) pairs uniformly from model keys, do k_plan model backups
    """
    env = Gridworld5x5(slip=slip, max_steps=max_steps, seed=env_seed)
    rng = np.random.default_rng(env_seed + 3000)
    Q = np.zeros((env.n_states, env.n_actions), dtype=np.float64)

    model = {}

    returns = np.zeros(episodes, dtype=np.float64)

    for ep in range(episodes):
        s = env.reset()
        G = 0.0

        while True:
            a = epsilon_greedy(Q, s, eps, rng)
            s2, r, done = env.step(a)
            G += r

            # 1) Real experience update (Q-learning)
            if done:
                td_target = r
            else:
                td_target = r + gamma * np.max(Q[s2])
            Q[s, a] += alpha * (td_target - Q[s, a])

            # 2) Update model
            model[(s, a)] = (r, s2)

            # 3) Planning updates
            if k_plan > 0 and len(model) > 0:
                keys = list(model.keys())
                for _ in range(k_plan):
                    sm, am = keys[int(rng.integers(0, len(keys)))]
                    rm, s2m = model[(sm, am)]
                    # model backup uses same Q-learning target
                    if s2m == env.n_states - 1:
                        td_m = rm
                    else:
                        td_m = rm + gamma * np.max(Q[s2m])
                    Q[sm, am] += alpha * (td_m - Q[sm, am])

            if done:
                break
            s = s2

        returns[ep] = G

    return returns


# ----------------------------
# Ploter (Figures 3 & 4)
# ----------------------------
def main():
    slip = 0.2
    episodes = 250
    seeds = [0, 1, 2, 3, 4, 5]  # 6 seeds
    alpha = 0.25
    gamma = 0.95
    eps = 0.10
    max_steps = 200

    os.makedirs("figures", exist_ok=True)

    # ---- Figure 3: SARSA vs Q-learning vs Dyna-Q(k=10)
    sarsa_runs = []
    ql_runs = []
    dyna10_runs = []

    for sd in seeds:
        sarsa_runs.append(run_sarsa(sd, episodes, alpha, gamma, eps, slip, max_steps))
        ql_runs.append(run_q_learning(sd, episodes, alpha, gamma, eps, slip, max_steps))
        dyna10_runs.append(run_dyna_q(sd, episodes, alpha, gamma, eps, slip, max_steps, k_plan=10))

    sarsa_mean = np.mean(np.stack(sarsa_runs), axis=0)
    ql_mean = np.mean(np.stack(ql_runs), axis=0)
    dyna10_mean = np.mean(np.stack(dyna10_runs), axis=0)

    plt.figure(figsize=(8.0, 3.6))
    plt.plot(sarsa_mean, label="SARSA")
    plt.plot(ql_mean, label="Q-learning")
    plt.plot(dyna10_mean, label="Dyna-Q (k=10)")
    plt.xlabel("Episode")
    plt.ylabel("Return per episode")
    plt.title(f"Gridworld (5x5, slip={slip}): average learning curves ({len(seeds)} seeds)")
    plt.legend()
    ax = plt.gca()
    ax.set_ylim(-25, 5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    plt.tight_layout()
    plt.savefig("figures/fig_learning_gridworld.pdf", bbox_inches="tight", pad_inches=0.12)
    plt.close()

    # ---- Figure 4: Dyna-Q planning budget sensitivity (k=0,5,10)
    ks = [0, 5, 10]
    curves = {}

    for k in ks:
        runs = []
        for sd in seeds:
            runs.append(run_dyna_q(sd, episodes, alpha, gamma, eps, slip, max_steps, k_plan=k))
        curves[k] = np.mean(np.stack(runs), axis=0)

    plt.figure(figsize=(8.0, 3.6))
    for k in ks:
        plt.plot(curves[k], label=f"k={k}")
    plt.xlabel("Episode")
    plt.ylabel("Return per episode")
    plt.title(f"Dyna-Q planning budget sensitivity (Gridworld 5x5, slip={slip})")
    plt.legend()
    ax = plt.gca()
    ax.set_ylim(-25, 5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    plt.tight_layout()
    plt.savefig("figures/fig_dyna_budget.pdf", bbox_inches="tight", pad_inches=0.12)
    plt.close()

    print("Saved:")
    print("  figures/fig_learning_gridworld.pdf")
    print("  figures/fig_dyna_budget.pdf")


if __name__ == "__main__":
    main()
