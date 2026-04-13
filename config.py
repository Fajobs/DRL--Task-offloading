"""
config.py — Central configuration for the MEC simulation.

This file holds every tunable parameter used across the project.
All values are drawn from Table I of the paper unless noted otherwise.

Paper reference:
  Jiang et al., "Deep-Reinforcement-Learning-Based Task Offloading and
  Resource Allocation in Mobile Edge Computing Network With Heterogeneous
  Tasks", IEEE Internet of Things Journal, Vol. 12, No. 8, April 2025.
"""

import math

# ═══════════════════════════════════════════════════════════════════════
# 1. PHYSICAL NETWORK LAYOUT
# ═══════════════════════════════════════════════════════════════════════

# Side length (metres) of the square area where devices are deployed.
# The paper uses a 600 m × 600 m region.
AREA_SIZE = 600

# Number of Edge Nodes (ENs), i.e. base-stations with MEC servers.
# Paper: M = 5.
NUM_ENS = 5

# ═══════════════════════════════════════════════════════════════════════
# 2. WIRELESS COMMUNICATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════

# Number of orthogonal channels available at each EN (OFDMA).
NUM_CHANNELS = 10

# Bandwidth of a single channel in Hz (1 MHz).
CHANNEL_BW = 1e6

# Additive White Gaussian Noise (AWGN) power.
# Specified in dBm, then converted to Watts for calculations.
NOISE_POWER_DBM = -100
NOISE_POWER = 10 ** ((NOISE_POWER_DBM - 30) / 10)

# Mobile device uplink transmit power (23 dBm ≈ 0.2 W).
TX_POWER_DBM = 23
TX_POWER = 10 ** ((TX_POWER_DBM - 30) / 10)

# Base-station transmit power, used to model inter-EN interference (30 dBm = 1 W).
BS_TX_POWER_DBM = 30
BS_TX_POWER = 10 ** ((BS_TX_POWER_DBM - 30) / 10)

# Wired backhaul link rate between ENs and the cloud server (bits/s).
# 100 Mbps.
WIRED_RATE = 100e6

# ═══════════════════════════════════════════════════════════════════════
# 3. COMPUTING CAPACITY
# ═══════════════════════════════════════════════════════════════════════

# Local device CPU speed in cycles/second (1 GHz).
LOCAL_CPU = 1e9

# Each EN's total CPU speed in cycles/second (10 GHz).
# This is shared equally among all devices connected to that EN.
EN_CPU = 10e9

# Cloud server CPU speed allocated per device (20 GHz).
CLOUD_CPU = 20e9

# ═══════════════════════════════════════════════════════════════════════
# 4. TASK HETEROGENEITY
# ═══════════════════════════════════════════════════════════════════════

# Four task types reflecting different IoT application data.
TASK_TYPES = {
    0: "text",
    1: "image",
    2: "audio",
    3: "video",
}

# Scenarios from Table II of the paper.
# Each scenario defines the *base* data size (in KB) for every task type.
# Scenario I has the smallest tasks; Scenario IV has the largest.
SCENARIOS = {
    "I":   {"text": 50,   "image": 100,  "audio": 200,  "video": 500},
    "II":  {"text": 100,  "image": 200,  "audio": 500,  "video": 1000},
    "III": {"text": 200,  "image": 500,  "audio": 1000, "video": 2000},
    "IV":  {"text": 500,  "image": 1000, "audio": 2000, "video": 5000},
}

# CPU cycles required to process one bit of data, per task type.
# More complex data (video) needs more computation per bit.
CPU_CYCLES_PER_BIT = {
    0: 100,   # text   — lightweight processing
    1: 200,   # image  — moderate processing
    2: 300,   # audio  — heavier processing
    3: 500,   # video  — heaviest processing
}

# ═══════════════════════════════════════════════════════════════════════
# 5. DQN HYPER-PARAMETERS  (Section V of the paper)
# ═══════════════════════════════════════════════════════════════════════

LEARNING_RATE = 5e-4           # Adam optimizer learning rate
DISCOUNT_FACTOR = 0.95         # γ (lambda in the paper) — how much future rewards matter
EPSILON_START = 1.0            # Initial exploration rate (100 % random)
EPSILON_END = 0.01             # Minimum exploration rate
EPSILON_DECAY = 0.97           # Multiply epsilon by this after every epoch — aggressive decay
BATCH_SIZE = 128               # Mini-batch size sampled from replay memory
MEMORY_SIZE = 20000            # Maximum transitions stored in replay buffer
TARGET_UPDATE_FREQ = 5         # Copy eval-net weights to target-net every k epochs
HIDDEN_DIM = 256               # Neurons per hidden layer in the Q-network

# Training budget
NUM_EPOCHS = 300               # Number of training epochs
STEPS_PER_EPOCH = 30           # Episodes (full environment resets) per epoch

# ═══════════════════════════════════════════════════════════════════════
# 6. ACTION-SPACE DISCRETISATION
# ═══════════════════════════════════════════════════════════════════════

# The DQN outputs a discrete action that maps to:
#   (alpha index, beta index, channel index)
# where alpha = fraction processed locally, beta = fraction at the EN,
# and gamma = 1 - alpha - beta = fraction at the cloud.
# We discretize alpha and beta into 6 levels each (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).
ALPHA_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BETA_LEVELS  = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
