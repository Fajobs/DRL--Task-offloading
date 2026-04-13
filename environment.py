"""
environment.py — MEC system simulation environment.

This module models the Mobile Edge Computing network described in Section III
of the paper.  It simulates:
  • N mobile devices scattered in a 600 m × 600 m area
  • M = 5 Edge Nodes (ENs / base-stations with MEC servers)
  • 1 remote cloud server connected to ENs via wired links
  • Heterogeneous tasks (text, image, audio, video)

The environment follows an OpenAI-Gym-like interface:
  state  = env.reset()          — generate fresh tasks, return first state
  state, reward, done, latency = env.step(device, action)

Terminology refresher (from the paper):
  α (alpha) — fraction of the task processed locally on the device
  β (beta)  — fraction offloaded to the connected Edge Node
  γ (gamma) — fraction forwarded to the remote cloud server
  α + β + γ = 1
"""

import math
import random

import numpy as np

from config import (
    AREA_SIZE, NUM_ENS, NUM_CHANNELS, CHANNEL_BW,
    NOISE_POWER, TX_POWER, BS_TX_POWER, WIRED_RATE,
    LOCAL_CPU, EN_CPU, CLOUD_CPU,
    TASK_TYPES, SCENARIOS, CPU_CYCLES_PER_BIT,
    ALPHA_LEVELS, BETA_LEVELS,
)


# ─────────────────────────────────────────────────────────────────────
# Helper: wireless path-loss model  (Eq. 5)
# ─────────────────────────────────────────────────────────────────────

def path_loss(distance_m: float) -> float:
    """
    Compute the *linear-scale* channel gain between a device and a base
    station separated by `distance_m` metres.

    The paper uses a simple log-distance model (Eq. 5):
        L(d) = 37 + 30 · log10(d)   [dB]

    We return the gain (inverse of loss) in linear scale so it can be
    plugged straight into the SINR formula.
    """
    distance_m = max(distance_m, 1.0)          # avoid log(0)
    loss_db = 37.0 + 30.0 * math.log10(distance_m)
    return 10.0 ** (-loss_db / 10.0)           # dB → linear gain


# ─────────────────────────────────────────────────────────────────────
# Helper: build the discrete action table
# ─────────────────────────────────────────────────────────────────────

def build_valid_actions(num_channels: int) -> list:
    """
    Enumerate every valid (alpha_index, beta_index, channel) triple.

    A combination is valid when alpha + beta ≤ 1 (so gamma ≥ 0).
    The DQN agent picks an integer index into this list; we then look
    up the corresponding offloading ratios and channel.

    Returns
    -------
    list of (int, int, int)
        Each entry is (alpha_level_index, beta_level_index, channel_id).
    """
    actions = []
    for ai, a in enumerate(ALPHA_LEVELS):
        for bi, b in enumerate(BETA_LEVELS):
            if a + b <= 1.0 + 1e-9:            # allow tiny float tolerance
                for ch in range(num_channels):
                    actions.append((ai, bi, ch))
    return actions


# ─────────────────────────────────────────────────────────────────────
# Main environment class
# ─────────────────────────────────────────────────────────────────────

class MECEnvironment:
    """
    Simulates one time-slot of the MEC system.

    Workflow per episode
    --------------------
    1. reset()  — randomly generate one task per device.
    2. For each device n = 0 … N-1:
       a. Read state  = _get_state(n)
       b. Agent picks action_index
       c. step(n, action_index) → next_state, reward, done, latency

    The reward is the *negative* task-completion latency (Eq. 18),
    so maximizing reward = minimizing delay.
    """

    def __init__(self, num_devices: int, scenario: str = "II",
                 task_density: int = 30):
        """
        Parameters
        ----------
        num_devices : int
            N — number of IoT devices (also equals the number of tasks
            generated per time-slot).
        scenario : str
            One of "I", "II", "III", "IV" — selects task sizes from Table II.
        task_density : int
            Conceptual density (tasks per 100 s).  In this simulation it
            simply equals num_devices; kept for labelling purposes.
        """
        self.num_devices = num_devices
        self.scenario = scenario
        self.task_density = task_density

        # --- Place Edge Nodes in a grid pattern inside the area ----------
        self.en_positions = self._place_ens()

        # --- Scatter devices uniformly at random -------------------------
        self.device_positions = np.random.uniform(
            0, AREA_SIZE, (num_devices, 2)
        )

        # --- Associate each device to its nearest EN (Section III-A) -----
        self.device_en = self._associate_devices()

        # --- Traffic load: how many devices are connected to each EN -----
        self.en_load = np.zeros(NUM_ENS, dtype=int)
        for m in self.device_en:
            self.en_load[m] += 1

        # --- Pre-compute device-to-EN distances (used in SINR calc) ------
        self.distances = np.zeros(num_devices)
        for n in range(num_devices):
            m = self.device_en[n]
            self.distances[n] = np.linalg.norm(
                self.device_positions[n] - self.en_positions[m]
            )

        # --- Discrete action table ---------------------------------------
        self.valid_actions = build_valid_actions(NUM_CHANNELS)
        self.num_actions = len(self.valid_actions)

        # --- State vector size per device --------------------------------
        # [task_type_normalized, data_size_normalized, cpu_cycles_normalized,
        #  channel_0_occupied?, channel_1_occupied?, …, channel_Nc-1_occupied?]
        self.state_dim = 3 + NUM_CHANNELS

    # ─── Internal helpers ────────────────────────────────────────────

    def _place_ens(self) -> np.ndarray:
        """Distribute M ENs roughly evenly across the area in a grid."""
        positions = []
        cols = int(math.ceil(math.sqrt(NUM_ENS)))
        rows = int(math.ceil(NUM_ENS / cols))
        dx = AREA_SIZE / (cols + 1)
        dy = AREA_SIZE / (rows + 1)
        for i in range(NUM_ENS):
            r, c = divmod(i, cols)
            positions.append([(c + 1) * dx, (r + 1) * dy])
        return np.array(positions)

    def _associate_devices(self) -> list:
        """Connect every device to the closest EN (Euclidean distance)."""
        associations = []
        for n in range(self.num_devices):
            dists = np.linalg.norm(
                self.en_positions - self.device_positions[n], axis=1
            )
            associations.append(int(np.argmin(dists)))
        return associations

    # ─── Gym-like interface ──────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """
        Start a new episode: generate one random task per device and
        clear all channel occupancy.

        Returns the state vector for device 0.
        """
        self.tasks = []
        scenario_sizes = SCENARIOS[self.scenario]

        for n in range(self.num_devices):
            # Pick a random task type (0=text, 1=image, 2=audio, 3=video)
            task_type = random.randint(0, 3)
            type_name = TASK_TYPES[task_type]

            # Base data size from the scenario table, with ±20 % jitter
            data_size_kb = scenario_sizes[type_name] * random.uniform(0.8, 1.2)
            data_size_bits = data_size_kb * 1024 * 8       # KB → bits

            # Total CPU cycles = data_bits × cycles_per_bit
            cpu_cycles = data_size_bits * CPU_CYCLES_PER_BIT[task_type]

            self.tasks.append({
                "type": task_type,
                "data_bits": data_size_bits,
                "cpu_cycles": cpu_cycles,
            })

        # Channel occupancy tracker: one array per EN, all free at start
        self.channel_state = {
            m: np.zeros(NUM_CHANNELS) for m in range(NUM_ENS)
        }
        return self._get_state(0)

    def _get_state(self, device_idx: int) -> np.ndarray:
        """
        Build the observation vector for device `device_idx`.

        Layout (Eq. 15 in the paper):
          [ task_type / 3,              — normalized to [0, 1]
            data_bits / 1e8,            — rough normalization
            cpu_cycles / 1e12,          — rough normalization
            ch_0_busy, ch_1_busy, …  ]  — binary flags (0 or 1)
        """
        task = self.tasks[device_idx]
        m = self.device_en[device_idx]

        data_norm = task["data_bits"] / 1e8
        cpu_norm  = task["cpu_cycles"] / 1e12
        ch_state  = self.channel_state[m].copy()

        state = np.concatenate([
            [task["type"] / 3.0, data_norm, cpu_norm],
            ch_state,
        ])
        return state.astype(np.float32)

    def step(self, device_idx: int, action_idx: int):
        """
        Execute the chosen action for one device and return results.

        Parameters
        ----------
        device_idx : int
            Which device is acting (0 … N-1).
        action_idx : int
            Index into self.valid_actions.

        Returns
        -------
        next_state : np.ndarray   — state of the next device (or zeros if done)
        reward     : float        — negative latency (Eq. 18)
        done       : bool         — True after the last device has acted
        latency    : float        — task completion time T_n in seconds (Eq. 13)
        """
        # ── Decode the discrete action into (alpha, beta, channel) ───
        ai, bi, ch = self.valid_actions[action_idx]
        alpha = ALPHA_LEVELS[ai]
        beta  = BETA_LEVELS[bi]
        gamma = max(0.0, 1.0 - alpha - beta)

        # ── Retrieve task info and device location ───────────────────
        task   = self.tasks[device_idx]
        m      = self.device_en[device_idx]       # connected EN index
        d_bits = task["data_bits"]
        p_cpu  = task["cpu_cycles"]
        dist   = self.distances[device_idx]

        # Mark the chosen channel as occupied at this EN
        self.channel_state[m][ch] = 1.0

        # ── Compute the latency using the paper's equations ──────────
        latency = self._compute_latency(
            device_idx, m, d_bits, p_cpu, dist, alpha, beta, gamma
        )

        # ── Reward = negative latency (Eq. 18) ──────────────────────
        reward = -latency

        # ── Advance to the next device ───────────────────────────────
        done = (device_idx >= self.num_devices - 1)
        if not done:
            next_state = self._get_state(device_idx + 1)
        else:
            next_state = np.zeros(self.state_dim, dtype=np.float32)

        return next_state, reward, done, latency

    # ─── Latency calculation (shared by step() and baselines) ────────

    def compute_latency_for_action(self, device_idx: int,
                                   alpha: float, beta: float,
                                   gamma: float, channel: int) -> float:
        """
        Public helper used by baseline algorithms (Random, Q-learning, GA)
        that supply continuous alpha/beta/gamma directly.
        """
        task   = self.tasks[device_idx]
        m      = self.device_en[device_idx]
        d_bits = task["data_bits"]
        p_cpu  = task["cpu_cycles"]
        dist   = self.distances[device_idx]
        return self._compute_latency(
            device_idx, m, d_bits, p_cpu, dist, alpha, beta, gamma
        )

    def _compute_latency(self, device_idx, m, d_bits, p_cpu, dist,
                         alpha, beta, gamma) -> float:
        """
        Core latency model (Eqs. 6-13).

        The task is split three ways:
          • α fraction → processed locally on the device
          • β fraction → offloaded to the connected EN
          • γ fraction → forwarded from the EN to the cloud

        Because the three portions execute *in parallel*, the total
        completion time is the maximum across the three paths (Eq. 13).
        """
        # ── Uplink transmission rate: device → EN  (Eq. 6) ──────────
        gain = path_loss(dist)

        # Inter-EN interference: signals from all *other* base-stations
        interference = 0.0
        for j in range(NUM_ENS):
            if j != m:
                d_j = np.linalg.norm(
                    self.en_positions[j] - self.device_positions[device_idx]
                )
                interference += BS_TX_POWER * path_loss(d_j)

        sinr    = TX_POWER * gain / (interference + NOISE_POWER)
        rate_up = CHANNEL_BW * math.log2(1 + sinr)   # Shannon capacity

        # ── Communication delays ─────────────────────────────────────
        # Device → EN uplink time (Eq. 7)
        if rate_up > 0 and (beta + gamma) > 0:
            t_comm_en = (beta + gamma) * d_bits / rate_up
        else:
            t_comm_en = 0.0

        # EN → Cloud wired transfer time (Eq. 8)
        t_comm_cloud = (gamma * d_bits / WIRED_RATE) if gamma > 0 else 0.0

        # ── Computation delays ───────────────────────────────────────
        # Local device (Eq. 9)
        t_local = (alpha * p_cpu / LOCAL_CPU) if alpha > 0 else 0.0

        # Edge Node (Eq. 10) — CPU shared among all connected devices
        load_m = max(self.en_load[m], 1)
        t_en = (beta * p_cpu * load_m / EN_CPU) if beta > 0 else 0.0

        # Cloud server (Eq. 11)
        t_cloud = (gamma * p_cpu / CLOUD_CPU) if gamma > 0 else 0.0

        # ── Per-path totals (Eq. 12) ────────────────────────────────
        t_L = t_local                              # local path
        t_E = t_comm_en + t_en                     # edge path
        t_C = t_comm_en + t_comm_cloud + t_cloud   # cloud path

        # ── Overall completion time (Eq. 13) — parallel execution ───
        return max(t_L, t_E, t_C)
