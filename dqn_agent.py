"""
dqn_agent.py — Deep Q-Network agent for the MEC offloading problem.

This module implements the DQN algorithm described in Section V of the paper:
  • A fully-connected neural network approximates Q(s, a; θ)  (Eq. 19)
  • An experience replay buffer stores past transitions to break correlation
  • A separate *target* network stabilizes training
  • ε-greedy exploration gradually shifts from random to greedy (Eq. 20)

Key classes
-----------
QNetwork      — the neural network itself (two hidden layers)
ReplayMemory  — fixed-size FIFO buffer of (s, a, r, s', done) tuples
DQNAgent      — ties everything together: action selection, training, etc.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    LEARNING_RATE, DISCOUNT_FACTOR,
    EPSILON_START, EPSILON_END, EPSILON_DECAY,
    BATCH_SIZE, MEMORY_SIZE, HIDDEN_DIM,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. Q-Network  (Section V-D)
# ═══════════════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    """
    A simple fully-connected network that maps a state vector to
    Q-values for every possible discrete action.

    Architecture:
        Input(state_dim) → Linear(hidden) → ReLU
                         → Linear(hidden) → ReLU
                         → Linear(action_dim)   ← one output per action

    The paper mentions K fully-connected layers; we use K = 3 (two hidden
    + one output) with 128 neurons per hidden layer by default.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values for all actions given a batch of states."""
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════
# 2. Experience Replay Memory  (Section V-C, step 2)
# ═══════════════════════════════════════════════════════════════════════

class ReplayMemory:
    """
    A fixed-capacity ring buffer that stores experience tuples
    (state, action, reward, next_state, done).

    During training, random mini-batches are sampled from this buffer.
    Random sampling breaks the temporal correlation between consecutive
    experiences, which stabilises and accelerates learning.
    """

    def __init__(self, capacity: int = MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store one transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        Draw a random mini-batch and return five NumPy arrays:
        states, actions, rewards, next_states, dones.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════════
# 3. DQN Agent  (Section V-B & V-C)
# ═══════════════════════════════════════════════════════════════════════

class DQNAgent:
    """
    Wraps the evaluate network, target network, replay memory, and
    training loop into a single convenient object.

    Lifecycle
    ---------
    1. Create:   agent = DQNAgent(state_dim, num_actions)
    2. Act:      action = agent.select_action(state)
    3. Store:    agent.store(s, a, r, s', done)
    4. Learn:    loss = agent.update()          — one gradient step
    5. Repeat 2-4 for many steps.
    6. Periodically call agent.update_target()  — sync target net
    7. After each epoch call agent.decay_epsilon()
    """

    def __init__(self, state_dim: int, num_actions: int,
                 device: str = "cpu"):
        self.num_actions = num_actions
        self.device = device

        # Exploration rate — starts high (random) and decays over time
        self.epsilon = EPSILON_START

        # Two identical networks: "evaluate" (trained) and "target" (frozen)
        self.eval_net   = QNetwork(state_dim, num_actions).to(device)
        self.target_net = QNetwork(state_dim, num_actions).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()   # target net is never trained directly

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=LEARNING_RATE)
        self.memory    = ReplayMemory(MEMORY_SIZE)
        self.loss_fn   = nn.MSELoss()

    # ─── Action selection (Eq. 20) ───────────────────────────────────

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """
        Choose an action using the ε-greedy policy.

        With probability ε → pick a random action  (exploration)
        With probability 1-ε → pick argmax Q(s, a)  (exploitation)

        Set `greedy=True` to force pure exploitation (used at evaluation).
        """
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.eval_net(state_t)
            return int(q_values.argmax(dim=1).item())

    # ─── Experience storage ──────────────────────────────────────────

    def store(self, state, action, reward, next_state, done):
        """Push a transition into the replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    # ─── Training step (Eq. 21) ─────────────────────────────────────

    def update(self) -> float:
        """
        Sample a mini-batch from replay memory and perform one gradient
        descent step on the MSE loss between predicted and target Q-values.

        Loss (Eq. 21):
            L(θ) = E[ (r + γ · max_a' Q_target(s', a'; θ') − Q(s, a; θ))² ]

        Returns the scalar loss value (useful for plotting convergence).
        """
        if len(self.memory) < BATCH_SIZE:
            return 0.0   # not enough data yet

        # Sample a random mini-batch
        states, actions, rewards, next_states, dones = \
            self.memory.sample(BATCH_SIZE)

        # Convert to PyTorch tensors
        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Q(s, a; θ) for the actions that were actually taken
        q_values = self.eval_net(states_t) \
            .gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target: r + γ · max_a' Q_target(s', a'; θ')
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            target = rewards_t + DISCOUNT_FACTOR * next_q * (1 - dones_t)

        # Gradient descent on MSE loss
        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # ─── Target network sync (Section V-C, step 4) ──────────────────

    def update_target(self):
        """
        Copy all weights from the evaluate network to the target network.
        Called every TARGET_UPDATE_FREQ epochs to keep the target stable.
        """
        self.target_net.load_state_dict(self.eval_net.state_dict())

    # ─── Exploration decay ───────────────────────────────────────────

    def decay_epsilon(self):
        """
        Shrink ε after each epoch so the agent gradually shifts from
        exploration (random actions) to exploitation (learned policy).
        """
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
