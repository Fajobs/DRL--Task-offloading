# DRL-Based Task Offloading in Mobile Edge Computing

A Python replication of the experiments from:

> **Jiang, T., Chen, Z., Zhao, Z., Feng, M., & Zhou, J.** (2025).
> *Deep-Reinforcement-Learning-Based Task Offloading and Resource Allocation
> in Mobile Edge Computing Network With Heterogeneous Tasks.*
> IEEE Internet of Things Journal, 12(8), 10899–10906.
> [DOI: 10.1109/JIOT.2024.3514108](Deep-Reinforcement-Learning-Based_Task_Offloading_and_Resource_Allocation_in_Mobile_Edge_Computing_Network_With_Heterogeneous_Tasks.pdf)

---

## What This Project Does

This project simulates a **Mobile Edge Computing (MEC)** network where IoT
devices generate tasks of different types (text, image, audio, video) and must
decide how to split each task between three computing locations:

| Location           | Description                                                           |
|--------------------|-----------------------------------------------------------------------|
| **Local device**   | Process on the device itself (slow CPU, no network delay)             |
| **Edge Node (EN)** | Offload to a nearby base-station server (fast CPU, wireless delay)    |
| **Cloud server**   | Forward through the EN to a remote cloud (fastest CPU, highest delay) |

The goal is to **minimize the average task completion latency** by jointly
optimizing:
- **Task partitioning** — what fraction goes to local / edge / cloud
- **Channel selection** — which wireless channel to use for uploading

A **Deep Q-Network (DQN)** agent learns this strategy through trial and error,
and is compared against three baselines: Random, tabular Q-Learning, and a
Genetic Algorithm (GA).

---

## Project Structure

```
├── config.py          # All simulation parameters and DQN hyperparameters
├── environment.py     # MEC network simulation (devices, ENs, cloud, channels)
├── dqn_agent.py       # DQN neural network, replay memory, and training agent
├── baselines.py       # Random, Q-Learning, and GA baseline algorithms
├── plotting.py        # Matplotlib figure generation (Figs. 3-7)
├── main.py            # Experiment runner — the entry point
├── results/           # Output folder for generated PNG figures
└── README.md          # This file
```

### Module Responsibilities

- **`config.py`** — Single source of truth for every tunable number: network
  layout, wireless parameters, computing capacities, task definitions, DQN
  hyperparameters, and action-space discretization. Change a value here and it
  propagates everywhere.

- **`environment.py`** — Implements the MEC system model from Section III of
  the paper. Handles device placement, EN association, SINR calculation,
  path-loss model, and the latency equations (Eqs. 5–13). Exposes a Gym-like
  `reset()` / `step()` interface.

- **`dqn_agent.py`** — The DQN algorithm from Section V. Contains the
  Q-network (two hidden layers), experience replay buffer, ε-greedy action
  selection, MSE loss training, and target network synchronization.

- **`baselines.py`** — Three comparison algorithms:
  - *Random* — uniformly random offloading ratios and channel
  - *Q-Learning* — tabular RL with discretised states (16 bins)
  - *GA* — binary-coded genetic algorithm with crossover and mutation

- **`plotting.py`** — Generates the five figures from the paper's evaluation
  section (Figs. 3–7) and saves them as PNGs.

- **`main.py`** — Orchestrates four experiments, calls training and evaluation
  functions, and produces all plots.

---

## How to Run

### Prerequisites

- Python 3.10+
- A virtual environment is recommended

### Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install torch numpy matplotlib
```

### Running the Experiments

Assuming you're using a virtual environment — which you are, because you aren't an idiot, right? See [Prerequisites](#prerequisites).

```bash
.venv\Scripts\python.exe main.py
```

You don't have a virtual environment? Huh? IDIOT....... here you go:

```bash
python main.py
```

This will run all four experiments sequentially and save figures to `results/`.
Expect it to take **5–15 minutes** depending on your hardware (CPU-only is
fine — no GPU required).

### What Gets Produced

| File                                   | Description                                      |
|----------------------------------------|--------------------------------------------------|
| `results/fig3_loss_convergence.png`    | DQN training loss over epochs                    |
| `results/fig4_reward_convergence.png`  | DQN reward (negative latency) over epochs        |
| `results/fig5_scheme_comparison.png`   | Bar chart: latency of all 4 schemes              |
| `results/fig6_scenario_comparison.png` | Grouped bars: latency across task-size scenarios |
| `results/fig7_density_comparison.png`  | Line plot: latency vs. task density              |

---

## Key Concepts Explained

### The MEC System

Imagine a city block with 5 cell towers (Edge Nodes), each equipped with a
small server. Dozens of IoT devices (phones, sensors, drones) are scattered
around. Each device generates a task — maybe a photo to classify, audio to
transcribe, or video to analyze.

Each device must decide: *"Should I process this myself, send it to the
nearest tower, or send it all the way to the cloud?"* The answer depends on
the task size, the device's CPU, how busy the tower is, and how good the
wireless channel is right now.

### Why DQN?

This is a hard optimization problem (NP-hard, technically) because:
- The wireless channel quality changes over time
- Different task types need different amounts of data and computation
- Devices compete for limited channels and edge server CPU

Traditional optimization can't solve this in real time. DQN learns a policy
by interacting with the environment thousands of times, storing experiences,
and gradually improving its decisions — much like how a human gets better at
a game through practice.

### The Three Baselines

| Baseline       | How it works                                       | Expected performance                                   |
|----------------|----------------------------------------------------|--------------------------------------------------------|
| **Random**     | Rolls dice for every decision                      | Worst — no intelligence                                |
| **Q-Learning** | Builds a lookup table of state→action values       | Better, but struggles with large state spaces          |
| **GA**         | Evolves a population of solutions over generations | Good, but slow and not adaptive to real-time changes   |
| **DQN**        | Neural network learns Q-values from experience     | Best — handles high-dimensional states and generalises |

---

## Tuning

All parameters live in `config.py`. Key knobs to turn:

| Parameter         | Default | Effect                                                      |
|-------------------|---------|-------------------------------------------------------------|
| `NUM_EPOCHS`      | 200     | More epochs → better DQN but slower training                |
| `STEPS_PER_EPOCH` | 20      | More steps → more data per epoch                            |
| `EPSILON_DECAY`   | 0.98    | Lower → faster shift from exploration to exploitation       |
| `LEARNING_RATE`   | 0.001   | Standard Adam LR; lower for more stable but slower learning |
| `HIDDEN_DIM`      | 128     | Neurons per hidden layer; increase for harder problems      |
| `NUM_CHANNELS`    | 10      | More channels → larger action space                         |

---

## Differences from the Paper

This is a faithful replication, but some implementation details are inferred
since the paper doesn't provide source code:

1. **Action-space discretization** — The paper describes continuous α, β but
   uses DQN (which needs discrete actions). We discretize into 11 levels each
   (0.0, 0.1, …, 1.0) combined with channel selection.

2. **State normalization** — We normalize task features for neural network
   input; the exact normalization scheme isn't specified in the paper.

3. **Interference model** — We use a simplified inter-EN interference model
   based on path loss from all other base stations.

4. **GA gene encoding** — The paper mentions 128-bit genes and 32-bit
   variables; we use a binary encoding with single-point crossover.

---

## License

This code is provided for educational and research purposes. The original
paper is © 2024 IEEE.
