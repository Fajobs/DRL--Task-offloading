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

## Energy-Aware Extension (`drl_task_offloading_energy.ipynb`)
A copy of the main notebook that implements the supervisor's whiteboard model (`paper/Board_Equations_New.docx`). The original notebook and its results are unchanged.

- **Objective:** minimise Σ (Tₙ + w_E·Eₙ), with w_E = 1 (T in ms, E in mJ). Local CPU frequency f_L (≤ 1 GHz) and transmit power Pₙ (≤ 23 dBm) are new decision variables.
- **Energy:** E_comp = K·f_L²·α·d·φ (K = 1e-27) and E_off = P·(β+γ)·d/R(P), where the rate R depends on the transmit power.
- **Baseline DQN** (original, latency-only) vs **Enhanced DQN** (chooses α, f_L and P; 336 actions). Random, GA and Q-learning also choose f_L and P.
- **Figures:** A (Baseline vs Enhanced), B (Fig 5 + energy), C (w_E trade-off), 6 and 7 (sweeps with energy panels). The DQN is retrained 11 times, so expect a longer runtime than the original.

### Results
**Fig. A: Baseline vs Enhanced DQN (Scenario II).** The result is a trade-off, not a double win. Both DQNs share the same limits, and the baseline is scored at the lowest possible latency, so the Enhanced DQN cuts energy by 79% at a 20% latency cost.

| | Latency | Energy | Cost (T+E) |
|---|---|---|---|
| Baseline DQN | 2.13 ms | 2.35 mJ | 4.47 |
| Enhanced DQN | 2.55 ms | 0.50 mJ | 3.06 |

**Fig. C: energy-weight sweep.** w_E = 0 → 2.15 ms / 2.38 mJ (the same as the baseline); 0.2 → 2.27 / 1.40; 0.5 → 2.45 / 0.79; 1.0 → 2.65 / 0.52; 2.0 → 2.94 / 0.32. A small w_E (≈0.2) already removes about 40% of the energy for a latency increase of about 5%.

**Fig. B: schemes under the new constraints (Scenario II, density 30).**

| Scheme | Latency | Energy | Cost |
|---|---|---|---|
| Random | 23.73 ms | 3.79 mJ | 27.52 |
| GA | 11.92 ms | 5.54 mJ | 17.46 |
| Q-learning | 9.12 ms | 5.38 mJ | 14.50 |
| Enhanced DQN | 2.64 ms | 0.52 mJ | 3.16 |

- **Figs. 6–7:** Enhanced DQN < Q-learning < GA < Random on cost at every scenario and density. Latency and energy both rise with task size and density.
- **Random uses less energy than GA and Q-learning.** It often picks a low CPU frequency (energy ∝ f_L²), so it saves energy by being very slow. Its cost is still the worst.
- **Random's latency (≈23 ms vs ≈14 ms in the latency-only notebook)** is higher because it can now choose CPU speeds as low as 0.25 GHz.

### Extension-specific implementation notes
- **Q-learning** (264 actions) is averaged over 3 training runs. Its update uses a running average for the first visits (`lr = max(0.1, 1/n)`) and evaluates only actions it has tried. With a constant learning rate from Q = 0, rarely tried actions looked about 10× cheaper than they are, and Q-learning performed worse than GA.
- **Random and GA** pick f_L and P from the DQN's discrete levels (GA: the upper two levels). GA is evaluated on 500 tasks everywhere.

## Running the Notebook
Requirements: `numpy`, `torch`, `matplotlib`

The notebook runs in ~5-10 minutes on CPU. The DQN training (Figs 3-4) uses PyTorch. The evaluation sections are independent and use the environment directly.
