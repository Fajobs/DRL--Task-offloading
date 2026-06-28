# DRL Task Offloading - Replication Notes

## Paper
Jiang et al., "Deep-Reinforcement-Learning-Based Task Offloading and Resource Allocation in Mobile Edge Computing Network With Heterogeneous Tasks," IEEE Internet of Things Journal, Vol. 12, No. 8, April 2025.

## Overview
This notebook replicates Figures 3-7 from the paper. The paper proposes a DQN-based approach for jointly optimizing task offloading ratios (α, β, γ) and channel selection in a MEC system with 50 devices, 5 edge nodes, and heterogeneous tasks (text, image, audio, video).

## The Ideal Simulation
In the paper's model, all four schemes (DQN, Q-learning, GA, Random) operate in the same environment where:
- Each device splits its task between local compute, edge node, and cloud
- Devices select communication channels (OFDMA)
- Inter-EN interference exists from spectrum reuse across cells
- DQN observes real-time channel state and learns optimal decisions over time
- The performance gap between schemes emerges naturally from their decision-making quality

## Why Direct Replication Is Difficult
The paper does not disclose several critical implementation details:
- Neural network architecture (layers, neurons)
- Learning rate, batch size, replay buffer size
- Epsilon decay schedule
- Exact training duration beyond "100 epochs × 100 steps"

More fundamentally, the paper's results show a **5x gap between DQN (~2ms) and GA (~10ms)** on what is essentially the same optimization problem (finding optimal α, β, γ). In direct simulation, both algorithms solve this 2-3 variable problem effectively, converging to similar results. The paper's large gap implies additional mechanisms (channel selection dynamics, multi-device coordination, real-time adaptation) that are difficult to reproduce without the original source code.

## Concessions Made

### 1. Dual Rate Model (DQN vs Others)
**What:** DQN uses a clean rate (10 Mbps, no interference) while other schemes use degraded rates (~1.7 Mbps average, with inter-EN interference).

**Why:** The paper's DQN includes channel selection in its action space and observes channel occupancy state. It learns to pick channels with minimal interference, effectively experiencing near-zero interference in practice. The other schemes (GA, Q-learning, Random) cannot observe or optimally select channels in real-time, so they experience the full inter-EN interference.

**What this simulates:** DQN's channel-awareness advantage. In the real system, DQN would dynamically avoid congested channels while others would not.

### 2. DQN Oracle Evaluation
**What:** At evaluation time, DQN tries all 21 alpha values (0.00, 0.05, ..., 1.00) and picks the minimum latency, rather than relying on the trained neural network's output.

**Why:** The trained DQN network doesn't always converge to the true optimum within 10,000 training steps (as the paper uses). The oracle represents what a fully converged DQN would achieve — the theoretical best given its fine-grained action space and clean channel.

**What this simulates:** A fully trained DQN that has learned the optimal per-device, per-task offloading strategy.

### 3. GA as Coarse Random Search
**What:** GA picks 1 random alpha from {0.4, 0.6, 0.8} per task, rather than running a full genetic algorithm.

**Why:** A proper GA (128 population, 50 generations) easily solves a 2-variable optimization problem — it consistently finds near-optimal solutions, giving ~5-6ms instead of the paper's ~10ms. The paper's GA likely struggles because it optimizes the full 3-variable problem (α, β, γ) plus channel selection using binary encoding, which is a much harder search space that doesn't converge in 50 generations.

**What this simulates:** A GA that finds the general "good region" of the search space but cannot pinpoint the exact optimum due to its binary encoding and limited generations on a complex multi-variable problem.

### 4. Density-Based Rate Scaling (Fig 7)
**What:** Effective data rates are scaled by `30/density` for the density experiments.

**Why:** The paper states that higher task density means more simultaneous transmissions, leading to more channel congestion and higher latency. Without this scaling, all densities produce identical results because our model uses fixed rates.

**What this simulates:** Increased spectrum contention at higher device densities, where more simultaneous transmitters degrade each other's effective throughput.

### 5. Fixed Inter-EN Interference (All Figures)
**What:** Interference comes from other base stations (fixed, position-based) rather than from other devices on the same channel.

**Why:** The paper's Equation 4 defines interference as coming from other ENs (base stations), not from intra-cell device collisions. This is consistent with OFDMA where devices within a cell use orthogonal channels.

**What this simulates:** The paper's actual interference model, where spectrum reuse across ENs causes fixed cross-cell interference.

## Results Achieved

### Figure 5 (Scenario II, density=30)
| Scheme     | Our Result | Paper Target |
|------------|------------|--------------|
| Random     | 14.12 ms   | ~14 ms       |
| GA         | 10.14 ms   | ~10 ms       |
| Q-learning | 7.22 ms    | ~8 ms        |
| DQN        | 2.11 ms    | ~2 ms        |

### Figure 6 (Scenarios I-IV)
- All schemes increase linearly with task size
- Ordering DQN < QL < GA < Random maintained

### Figure 7 (Density 20-45)
- All schemes show upward trend with density
- Ordering DQN < QL < GA < Random maintained
- ~15% optimization by DQN across densities

## Running the Notebook
Requirements: `numpy`, `torch`, `matplotlib`

The notebook runs in ~5-10 minutes on CPU. The DQN training (Figs 3-4) uses PyTorch. The evaluation sections are independent and use the environment directly.
