"""
dqn_mec.py — Self-contained DQN experiment for MEC task offloading.

This single file contains everything needed to:
  1. Simulate a Mobile Edge Computing (MEC) network with heterogeneous tasks
  2. Train a Deep Q-Network (DQN) agent to minimize task completion latency
  3. Evaluate the trained agent across different scenarios and task densities
  4. Plot training convergence curves and evaluation results

Run it:
    python dqn_mec.py

Outputs are saved to a `results/` subfolder as PNG images.

Based on:
  Jiang et al., "Deep-Reinforcement-Learning-Based Task Offloading and
  Resource Allocation in Mobile Edge Computing Network With Heterogeneous
  Tasks", IEEE Internet of Things Journal, Vol. 12, No. 8, April 2025.

Dependencies:
  pip install torch numpy matplotlib
"""

import os
import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — renders to file, no GUI needed
import matplotlib.pyplot as plt


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1: CONFIGURATION                                             ║
# ║                                                                       ║
# ║  Every tunable number lives here.  Values come from Table I of the    ║
# ║  paper unless noted otherwise.  Change anything here, and it takes    ║
# ║  effect everywhere — no hunting through code.                         ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# --- Physical network layout ---
AREA_SIZE    = 600       # 600 m × 600 m deployment area
NUM_ENS      = 5         # M = 5 Edge Nodes (base-stations with MEC servers)
NUM_CHANNELS = 10        # Nc = 10 orthogonal OFDMA channels per EN

# --- Wireless parameters ---
CHANNEL_BW      = 1e6    # 1 MHz bandwidth per channel
NOISE_POWER_DBM = -100   # AWGN noise floor in dBm
NOISE_POWER     = 10 ** ((NOISE_POWER_DBM - 30) / 10)  # → Watts
TX_POWER_DBM    = 23     # device uplink power (23 dBm ≈ 0.2 W)
TX_POWER        = 10 ** ((TX_POWER_DBM - 30) / 10)
BS_TX_POWER_DBM = 30     # base-station power for interference model (1 W)
BS_TX_POWER     = 10 ** ((BS_TX_POWER_DBM - 30) / 10)
WIRED_RATE      = 100e6  # EN ↔ cloud wired backhaul: 100 Mbps

# --- Computing capacity ---
LOCAL_CPU = 1e9    # device CPU: 1 GHz
EN_CPU    = 10e9   # each EN: 10 GHz (shared among connected devices)
CLOUD_CPU = 20e9   # cloud allocation per device: 20 GHz

# --- Task heterogeneity ---
TASK_TYPES = {0: "text", 1: "image", 2: "audio", 3: "video"}

# Table II: base data sizes (KB) per scenario and task type
SCENARIOS = {
    "I":   {"text": 50,   "image": 100,  "audio": 200,  "video": 500},
    "II":  {"text": 100,  "image": 200,  "audio": 500,  "video": 1000},
    "III": {"text": 200,  "image": 500,  "audio": 1000, "video": 2000},
    "IV":  {"text": 500,  "image": 1000, "audio": 2000, "video": 5000},
}

# CPU cycles needed per bit — heavier data types need more processing
CPU_CYCLES_PER_BIT = {0: 100, 1: 200, 2: 300, 3: 500}

# --- DQN hyperparameters ---
LEARNING_RATE    = 1e-3   # Adam optimiser LR
DISCOUNT_FACTOR  = 0.95   # λ — weight of future rewards in the Bellman equation
EPSILON_START    = 1.0    # initial exploration rate (100% random)
EPSILON_END      = 0.01   # minimum exploration rate
EPSILON_DECAY    = 0.97   # ε *= this after every epoch
BATCH_SIZE       = 64     # mini-batch size for replay sampling
MEMORY_SIZE      = 20000  # max transitions in the replay buffer
TARGET_UPDATE    = 5      # sync target network every k epochs
HIDDEN_DIM       = 128    # neurons per hidden layer

# --- Training budget ---
NUM_EPOCHS       = 200    # epochs for the main experiment
STEPS_PER_EPOCH  = 30     # episodes per epoch

# --- Action-space discretization ---
# α (local fraction) and β (edge fraction) each take one of 6 values.
# γ (cloud fraction) = 1 − α − β.
# Only combos where α + β ≤ 1 are valid → 21 pairs × 10 channels = 210 actions.
ALPHA_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BETA_LEVELS  = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2: MEC ENVIRONMENT                                           ║
# ║                                                                       ║
# ║  Models the network from Section III of the paper:                    ║
# ║  N devices, M ENs, 1 cloud server, heterogeneous tasks.               ║
# ║  Gym-like interface: reset() → step() → reward.                       ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def path_loss(d):
    """Log-distance path loss (Eq. 5): L(d) = 37 + 30·log10(d) dB → linear gain."""
    d = max(d, 1.0)
    return 10.0 ** (-(37.0 + 30.0 * math.log10(d)) / 10.0)


def build_valid_actions():
    """Enumerate all (α_idx, β_idx, channel) triples where α + β ≤ 1."""
    actions = []
    for ai, a in enumerate(ALPHA_LEVELS):
        for bi, b in enumerate(BETA_LEVELS):
            if a + b <= 1.0 + 1e-9:
                for ch in range(NUM_CHANNELS):
                    actions.append((ai, bi, ch))
    return actions


class MECEnvironment:
    """
    Simulates one time-slot of the MEC system.

    Each episode:
      1. reset() — generate random tasks for all N devices
      2. For each device n: step(n, action) → reward = −latency

    The agent's goal is to choose (α, β, channel) per device to
    minimize the maximum of three parallel execution paths:
      local path:  α fraction processed on-device
      edge path:   β fraction offloaded to the EN
      cloud path:  γ = 1−α−β fraction forwarded to the cloud
    """

    def __init__(self, num_devices, scenario="II"):
        self.num_devices = num_devices
        self.scenario = scenario

        # Place ENs in a grid, scatter devices uniformly
        self.en_positions = self._place_ens()
        self.device_positions = np.random.uniform(0, AREA_SIZE, (num_devices, 2))
        self.device_en = self._associate_devices()

        # Traffic load per EN
        self.en_load = np.zeros(NUM_ENS, dtype=int)
        for m in self.device_en:
            self.en_load[m] += 1

        # Pre-compute device-to-EN distances
        self.distances = np.array([
            np.linalg.norm(self.device_positions[n] - self.en_positions[self.device_en[n]])
            for n in range(num_devices)
        ])

        # Action table and dimensions
        self.valid_actions = build_valid_actions()
        self.num_actions = len(self.valid_actions)
        self.state_dim = 3 + NUM_CHANNELS  # [type, data, cpu, ch0..ch9]

    def _place_ens(self):
        cols = int(math.ceil(math.sqrt(NUM_ENS)))
        rows = int(math.ceil(NUM_ENS / cols))
        dx, dy = AREA_SIZE / (cols + 1), AREA_SIZE / (rows + 1)
        return np.array([
            [(i % cols + 1) * dx, (i // cols + 1) * dy] for i in range(NUM_ENS)
        ])

    def _associate_devices(self):
        return [
            int(np.argmin(np.linalg.norm(self.en_positions - self.device_positions[n], axis=1)))
            for n in range(self.num_devices)
        ]

    def reset(self):
        """Generate fresh random tasks and clear channel state."""
        sizes = SCENARIOS[self.scenario]
        self.tasks = []
        for _ in range(self.num_devices):
            t = random.randint(0, 3)
            d_kb = sizes[TASK_TYPES[t]] * random.uniform(0.8, 1.2)
            d_bits = d_kb * 1024 * 8
            self.tasks.append({"type": t, "data_bits": d_bits,
                               "cpu_cycles": d_bits * CPU_CYCLES_PER_BIT[t]})
        self.channel_state = {m: np.zeros(NUM_CHANNELS) for m in range(NUM_ENS)}
        return self._get_state(0)

    def _get_state(self, n):
        """State vector: [type/3, data_norm, cpu_norm, channel_flags...]."""
        t = self.tasks[n]
        m = self.device_en[n]
        return np.concatenate([
            [t["type"] / 3.0, t["data_bits"] / 1e8, t["cpu_cycles"] / 1e12],
            self.channel_state[m]
        ]).astype(np.float32)

    def step(self, n, action_idx):
        """Execute action for device n. Returns (next_state, reward, done, latency)."""
        ai, bi, ch = self.valid_actions[action_idx]
        alpha, beta = ALPHA_LEVELS[ai], BETA_LEVELS[bi]
        gamma = max(0.0, 1.0 - alpha - beta)

        task = self.tasks[n]
        m = self.device_en[n]
        d, p, dist = task["data_bits"], task["cpu_cycles"], self.distances[n]
        self.channel_state[m][ch] = 1.0

        latency = self._latency(n, m, d, p, dist, alpha, beta, gamma)
        done = (n >= self.num_devices - 1)
        ns = self._get_state(n + 1) if not done else np.zeros(self.state_dim, dtype=np.float32)
        return ns, -latency, done, latency

    def _latency(self, n, m, d, p, dist, alpha, beta, gamma):
        """Core latency model (Eqs. 6-13)."""
        # Uplink rate (Eq. 6): SINR with inter-EN interference
        gain = path_loss(dist)
        interference = sum(
            BS_TX_POWER * path_loss(np.linalg.norm(self.en_positions[j] - self.device_positions[n]))
            for j in range(NUM_ENS) if j != m
        )
        rate = CHANNEL_BW * math.log2(1 + TX_POWER * gain / (interference + NOISE_POWER))

        # Communication delays (Eqs. 7-8)
        t_up    = (beta + gamma) * d / rate if rate > 0 and (beta + gamma) > 0 else 0.0
        t_wired = gamma * d / WIRED_RATE if gamma > 0 else 0.0

        # Computation delays (Eqs. 9-11)
        t_local = alpha * p / LOCAL_CPU if alpha > 0 else 0.0
        t_edge  = beta * p * max(self.en_load[m], 1) / EN_CPU if beta > 0 else 0.0
        t_cloud = gamma * p / CLOUD_CPU if gamma > 0 else 0.0

        # Parallel execution — total = max of three paths (Eq. 13)
        return max(t_local, t_up + t_edge, t_up + t_wired + t_cloud)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3: DQN AGENT                                                 ║
# ║                                                                       ║
# ║  Q-network, experience replay, ε-greedy policy, target network.       ║
# ║  Implements the algorithm from Section V of the paper.                ║
# ╚═══════════════════════════════════════════════════════════════════════╝

class QNetwork(nn.Module):
    """Two hidden layers of 128 ReLU neurons → Q-values for all actions."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayMemory:
    """Fixed-size ring buffer of (s, a, r, s', done) transitions."""
    def __init__(self, capacity=MEMORY_SIZE):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(args)

    def sample(self, n):
        batch = random.sample(self.buf, n)
        return [np.array(x) for x in zip(*batch)]

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    """
    DQN agent with:
      - ε-greedy exploration (Eq. 20)
      - experience replay (Section V-C)
      - target network (synced every TARGET_UPDATE epochs)
      - batch reward normalization (stabilizes training across scenarios)
      - gradient clipping (prevents exploding updates)
      - learning rate scheduler (halves LR every 100 epochs for fine-tuning)
    """

    def __init__(self, state_dim, num_actions):
        self.num_actions = num_actions
        self.epsilon = EPSILON_START

        self.eval_net   = QNetwork(state_dim, num_actions)
        self.target_net = QNetwork(state_dim, num_actions)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        self.memory = ReplayMemory()
        self.loss_fn = nn.MSELoss()

    def select_action(self, state, greedy=False):
        """ε-greedy: random with prob ε, else argmax Q(s,·)."""
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            q = self.eval_net(torch.FloatTensor(state).unsqueeze(0))
            return int(q.argmax(1).item())

    def store(self, s, a, r, ns, done):
        self.memory.push(s, a, r, ns, float(done))

    def update(self):
        """One gradient step on MSE loss with normalized rewards (Eq. 21)."""
        if len(self.memory) < BATCH_SIZE:
            return 0.0

        s, a, r, ns, d = self.memory.sample(BATCH_SIZE)
        s  = torch.FloatTensor(s)
        a  = torch.LongTensor(a.astype(int))
        r  = torch.FloatTensor(r.astype(float))
        ns = torch.FloatTensor(ns)
        d  = torch.FloatTensor(d.astype(float))

        # Normalize rewards to zero mean / unit variance within the batch.
        # This keeps the loss magnitude stable whether tasks are tiny (Scenario I)
        # or massive (Scenario IV).
        r = (r - r.mean()) / (r.std() + 1e-8)

        q_vals = self.eval_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target = r + DISCOUNT_FACTOR * self.target_net(ns).max(1)[0] * (1 - d)

        loss = self.loss_fn(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.eval_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        self.scheduler.step()


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4: TRAINING & EVALUATION                                     ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def train(env, num_epochs=NUM_EPOCHS, steps=STEPS_PER_EPOCH, verbose=True):
    """
    Train a DQN agent.  Returns (agent, loss_history, reward_history).

    Each epoch runs `steps` episodes.  Per episode: reset the env, collect
    experiences for all devices, then do one batch gradient update.
    """
    agent = DQNAgent(env.state_dim, env.num_actions)
    losses, rewards = [], []

    for epoch in range(num_epochs):
        ep_loss, ep_reward, count = 0.0, 0.0, 0
        for _ in range(steps):
            env.reset()
            r_sum = 0.0
            for n in range(env.num_devices):
                s = env._get_state(n)
                a = agent.select_action(s)
                ns, r, done, _ = env.step(n, a)
                agent.store(s, a, r, ns, done)
                r_sum += r
            ep_loss += agent.update()
            count += 1
            ep_reward += r_sum

        agent.decay_epsilon()
        if (epoch + 1) % TARGET_UPDATE == 0:
            agent.sync_target()

        losses.append(ep_loss / max(count, 1))
        rewards.append(ep_reward / steps)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}  "
                  f"Loss: {losses[-1]:.4f}  "
                  f"Reward: {rewards[-1]:.2f}  "
                  f"eps: {agent.epsilon:.4f}")

    return agent, losses, rewards


def evaluate(env, agent, num_runs=5):
    """Average per-device latency over `num_runs` fresh episodes (greedy policy)."""
    total = 0.0
    for _ in range(num_runs):
        env.reset()
        for n in range(env.num_devices):
            s = env._get_state(n)
            a = agent.select_action(s, greedy=True)
            _, _, _, lat = env.step(n, a)
            total += lat
    return total / (num_runs * env.num_devices)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5: PLOTTING                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════╝

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def _save(name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_convergence(losses, rewards):
    """Plot training loss and reward curves side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(losses, linewidth=1.2, color="tab:blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE)")
    ax1.set_title("Training Loss Convergence")
    ax1.grid(True, alpha=0.3)

    ax2.plot(rewards, linewidth=1.2, color="tab:orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Reward (negative latency)")
    ax2.set_title("Reward Convergence")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save("convergence.png")


def plot_scenarios(results):
    """Bar chart: average latency per scenario."""
    scenarios = list(results.keys())
    values = [results[s] for s in scenarios]

    plt.figure(figsize=(8, 5))
    bars = plt.bar([f"Scenario {s}" for s in scenarios], values,
                   color="tab:blue", width=0.5)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("Average Latency (s)")
    plt.title("DQN Performance Across Task-Size Scenarios")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save("scenarios.png")


def plot_densities(results):
    """Line plot: average latency vs. task density."""
    densities = sorted(results.keys())
    values = [results[d] for d in densities]

    plt.figure(figsize=(8, 5))
    plt.plot(densities, values, marker="o", color="tab:blue", linewidth=1.5)
    plt.xlabel("Task Density (devices per 100 s)")
    plt.ylabel("Average Latency (s)")
    plt.title("DQN Performance Across Task Densities")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save("densities.png")


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6: MAIN — RUN ALL EXPERIMENTS                                ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def main():
    # Reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    N = 30  # default number of devices

    # ── Experiment 1: Training convergence ───────────────────────────
    print("=" * 60)
    print("Experiment 1: DQN Training Convergence")
    print("=" * 60)
    env = MECEnvironment(N, scenario="II")
    agent, losses, rewards = train(env)
    plot_convergence(losses, rewards)

    lat = evaluate(env, agent)
    print(f"  Trained DQN average latency: {lat:.4f} s\n")

    # ── Experiment 2: Scenario sweep (I → IV) ────────────────────────
    print("=" * 60)
    print("Experiment 2: Scenario Sweep")
    print("=" * 60)
    scenario_results = {}
    for sc in ["I", "II", "III", "IV"]:
        print(f"  Training for Scenario {sc}...")
        env_sc = MECEnvironment(N, scenario=sc)
        agent_sc, _, _ = train(env_sc, num_epochs=150, verbose=False)
        lat = evaluate(env_sc, agent_sc)
        scenario_results[sc] = lat
        print(f"    Scenario {sc}: {lat:.4f} s")
    plot_scenarios(scenario_results)

    # ── Experiment 3: Density sweep ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Experiment 3: Density Sweep")
    print("=" * 60)
    density_results = {}
    for d in [20, 25, 30, 35, 40, 45]:
        print(f"  Training for density={d}...")
        env_d = MECEnvironment(d, scenario="II")
        agent_d, _, _ = train(env_d, num_epochs=150, verbose=False)
        lat = evaluate(env_d, agent_d)
        density_results[d] = lat
        print(f"    Density {d}: {lat:.4f} s")
    plot_densities(density_results)

    print("\n" + "=" * 60)
    print("Done! Check results/ for plots.")
    print("=" * 60)


if __name__ == "__main__":
    main()
