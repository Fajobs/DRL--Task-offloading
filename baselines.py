"""
baselines.py — Three benchmark algorithms for comparison with DQN.

The paper (Section VI-B) evaluates DQN against:
  1. Random  — pick offloading ratios and channel uniformly at random
  2. Q-Learning — tabular RL with discretized state/action spaces
  3. GA (Genetic Algorithm) — evolutionary heuristic optimization

Each function takes a pre-built MECEnvironment, runs its strategy,
and returns the *average per-device latency* (seconds).
"""

import random
import numpy as np

from config import ALPHA_LEVELS, BETA_LEVELS, NUM_CHANNELS


# ═══════════════════════════════════════════════════════════════════════
# 1. RANDOM BASELINE
# ═══════════════════════════════════════════════════════════════════════

def evaluate_random(env) -> float:
    """
    Random offloading: for every device, draw alpha ∈ [0,1] uniformly,
    then beta ∈ [0, 1-alpha], compute gamma = 1-alpha-beta, and pick
    a random channel.

    This is the weakest baseline — no intelligence at all.

    Returns
    -------
    float — average task completion latency across all devices (seconds).
    """
    env.reset()
    total_latency = 0.0

    for n in range(env.num_devices):
        alpha = random.random()                       # [0, 1]
        beta  = random.random() * (1.0 - alpha)       # [0, 1-alpha]
        gamma = 1.0 - alpha - beta                     # remainder → cloud
        ch    = random.randint(0, NUM_CHANNELS - 1)

        latency = env.compute_latency_for_action(n, alpha, beta, gamma, ch)
        total_latency += latency

    return total_latency / env.num_devices


# ═══════════════════════════════════════════════════════════════════════
# 2. TABULAR Q-LEARNING BASELINE
# ═══════════════════════════════════════════════════════════════════════

def evaluate_qlearning(env, num_episodes: int = 200) -> float:
    """
    Classic tabular Q-learning with discretized states and actions.

    State space:
        We bucket each device's task into one of 16 bins:
        4 task types × 4 data-size buckets = 16 discrete states.

    Action space:
        Same discrete (alpha, beta, channel) triples as the DQN,
        using ALPHA_LEVELS and BETA_LEVELS from config.

    The Q-table stores expected *rewards* (negative latency).
    During evaluation, we pick the action with the *highest* Q-value
    (= least negative = lowest latency).

    Parameters
    ----------
    env : MECEnvironment
    num_episodes : int — how many training episodes to run

    Returns
    -------
    float — average latency after training (seconds).
    """
    # ── Build the discrete action list ───────────────────────────────
    q_actions = []
    for a in ALPHA_LEVELS:
        for b in BETA_LEVELS:
            if a + b <= 1.0 + 1e-9:
                for ch in range(NUM_CHANNELS):
                    q_actions.append((a, b, ch))
    num_q_actions = len(q_actions)

    # ── Q-table: 16 states × num_q_actions ───────────────────────────
    num_states = 16   # 4 task types × 4 size buckets
    Q_table = np.zeros((num_states, num_q_actions))

    lr  = 0.1    # learning rate for Q-update
    gam = 0.9    # discount factor
    eps = 1.0    # exploration rate (decays over episodes)

    def state_to_idx(task_type: int, data_bits: float) -> int:
        """Map (task_type, data_size) to one of 16 discrete state indices."""
        size_kb = data_bits / (1024 * 8)
        if size_kb < 150:
            bucket = 0
        elif size_kb < 500:
            bucket = 1
        elif size_kb < 1500:
            bucket = 2
        else:
            bucket = 3
        return task_type * 4 + bucket

    # ── Training loop ────────────────────────────────────────────────
    for ep in range(num_episodes):
        env.reset()

        for n in range(env.num_devices):
            task  = env.tasks[n]
            s_idx = state_to_idx(task["type"], task["data_bits"])

            # ε-greedy action selection
            if random.random() < eps:
                a_idx = random.randrange(num_q_actions)
            else:
                # Pick the action with the HIGHEST Q-value (max reward)
                a_idx = int(np.argmax(Q_table[s_idx]))

            alpha, beta, ch = q_actions[a_idx]
            gamma_val = max(0.0, 1.0 - alpha - beta)
            latency   = env.compute_latency_for_action(
                n, alpha, beta, gamma_val, ch
            )
            reward = -latency   # reward = negative latency

            # Q-learning update: Q(s,a) ← Q(s,a) + lr·[r + γ·max Q(s') − Q(s,a)]
            if n < env.num_devices - 1:
                next_task = env.tasks[n + 1]
                s_next    = state_to_idx(next_task["type"], next_task["data_bits"])
                Q_table[s_idx, a_idx] += lr * (
                    reward + gam * np.max(Q_table[s_next])
                    - Q_table[s_idx, a_idx]
                )
            else:
                Q_table[s_idx, a_idx] += lr * (
                    reward - Q_table[s_idx, a_idx]
                )

        eps = max(0.01, eps * 0.99)   # decay exploration

    # ── Evaluate the learned policy (greedy) ─────────────────────────
    env.reset()
    total_latency = 0.0

    for n in range(env.num_devices):
        task  = env.tasks[n]
        s_idx = state_to_idx(task["type"], task["data_bits"])
        a_idx = int(np.argmax(Q_table[s_idx]))   # greedy: best Q-value

        alpha, beta, ch = q_actions[a_idx]
        gamma_val = max(0.0, 1.0 - alpha - beta)
        latency   = env.compute_latency_for_action(
            n, alpha, beta, gamma_val, ch
        )
        total_latency += latency

    return total_latency / env.num_devices


# ═══════════════════════════════════════════════════════════════════════
# 3. GENETIC ALGORITHM (GA) BASELINE
# ═══════════════════════════════════════════════════════════════════════

def evaluate_ga(env, pop_size: int = 50, generations: int = 100,
                gene_len: int = 128) -> float:
    """
    Heuristic search using a binary-coded Genetic Algorithm.

    Each *individual* in the population is a long binary string that
    encodes (alpha, beta) for every device.  The GA evolves the
    population through selection, crossover, and mutation to minimise
    total latency.

    Gene encoding per device (gene_len bits total):
        First half  → alpha ∈ [0, 1]
        Second half → beta_raw ∈ [0, 1], then beta = beta_raw × (1 − alpha)
        gamma = 1 − alpha − beta

    Channel assignment is simplified: device n uses channel (n mod Nc).

    Parameters
    ----------
    env : MECEnvironment
    pop_size : int    — number of individuals in the population
    generations : int — number of evolutionary generations
    gene_len : int    — bits per device (paper uses 128)

    Returns
    -------
    float — average latency of the best individual found (seconds).
    """
    env.reset()
    num_d = env.num_devices
    bits_per_var = gene_len // 2          # 64 bits for alpha, 64 for beta
    total_bits   = num_d * gene_len       # total genome length

    # ── Decode a binary genome into per-device decisions ─────────────
    def decode(individual: list) -> list:
        """Convert binary string → list of (alpha, beta, gamma) per device."""
        decisions = []
        for n in range(num_d):
            start = n * gene_len
            # Decode alpha from the first half of this device's gene
            alpha_bits = individual[start : start + bits_per_var]
            alpha = int("".join(map(str, alpha_bits)), 2) / (2**bits_per_var - 1)

            # Decode beta from the second half, scaled to [0, 1-alpha]
            beta_bits = individual[start + bits_per_var : start + gene_len]
            beta_raw  = int("".join(map(str, beta_bits)), 2) / (2**bits_per_var - 1)
            beta  = beta_raw * (1.0 - alpha)
            gamma = 1.0 - alpha - beta

            decisions.append((alpha, beta, gamma))
        return decisions

    # ── Fitness = negative total latency (higher is better) ──────────
    def fitness(individual: list) -> float:
        decisions = decode(individual)
        total = 0.0
        for n in range(num_d):
            a, b, g = decisions[n]
            ch = n % NUM_CHANNELS
            total += env.compute_latency_for_action(n, a, b, g, ch)
        return -total

    # ── Initialize a random population ───────────────────────────────
    population = [
        np.random.randint(0, 2, total_bits).tolist()
        for _ in range(pop_size)
    ]

    # ── Evolutionary loop ────────────────────────────────────────────
    for gen in range(generations):
        # Score and rank every individual
        scored = [(fitness(ind), ind) for ind in population]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Elitism: keep the top 10 % unchanged
        elite_count = max(2, pop_size // 10)
        new_pop = [s[1] for s in scored[:elite_count]]

        # Fill the rest via crossover + mutation
        while len(new_pop) < pop_size:
            # Pick two elite parents at random
            p1 = random.choice(new_pop[:elite_count])
            p2 = random.choice(new_pop[:elite_count])

            # Single-point crossover
            cx = random.randint(1, total_bits - 1)
            child = p1[:cx] + p2[cx:]

            # Bit-flip mutation (1 % chance per bit)
            for i in range(len(child)):
                if random.random() < 0.01:
                    child[i] = 1 - child[i]

            new_pop.append(child)

        population = new_pop

    # ── Extract the best solution ────────────────────────────────────
    best = max(population, key=fitness)
    decisions = decode(best)
    total_latency = 0.0
    for n in range(num_d):
        a, b, g = decisions[n]
        ch = n % NUM_CHANNELS
        total_latency += env.compute_latency_for_action(n, a, b, g, ch)

    return total_latency / num_d
