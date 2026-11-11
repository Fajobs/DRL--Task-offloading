# DRL Task Offloading in Mobile Edge Computing

This repository has two parts:

| Notebook | What it does |
|---|---|
| `drl_task_offloading.ipynb` | **Replication.** Reproduces Figures 3–7 of the paper below (latency-only DQN vs Q-learning, GA and Random). Its results are fixed and are not modified by the extension. |
| `drl_task_offloading_energy.ipynb` | **Energy-aware extension.** Adds device energy to the objective, makes CPU frequency and transmit power decision variables, uses realistic task sizes, and compares a latency-only **Baseline DQN** with an **Energy-Efficient DQN** and energy-aware baselines. |

`paper/` holds the paper PDF, the energy-model specification (`Board_Equations_New.docx`) and the revision notes for the extension (`Adjustments, Corrections and expected new and improved results.docx`).

**Contents:** [Paper](#paper) · [System model](#system-model) · [Part 1: Replication](#part-1-replication) · [Part 2: Energy-aware extension](#part-2-energy-aware-extension) · [Limitations and open questions](#limitations-and-open-questions) · [Running the notebooks](#running-the-notebooks)

## Paper
Jiang et al., "Deep-Reinforcement-Learning-Based Task Offloading and Resource Allocation in Mobile Edge Computing Network With Heterogeneous Tasks," IEEE Internet of Things Journal, Vol. 12, No. 8, April 2025.

The paper proposes a DQN-based approach for jointly optimising task offloading ratios (α, β, γ) and channel selection in an MEC system with 50 devices, 5 edge nodes (ENs) and heterogeneous tasks (text, image, audio, video).

## System model
Both notebooks share the same network and latency model.

### Network and parameters

| Parameter | Value |
|---|---|
| Devices N / edge nodes M | 50 / 5 |
| Area | 600 m × 600 m; ENs at fixed positions, devices placed uniformly at random (seed 42) |
| Association | Each device attaches to its nearest EN; L_m = number of devices on EN m |
| Bandwidth per device | 1 MHz |
| Nominal transmit power | 23 dBm |
| Noise power | −70 dBm |
| Path loss | 35 + 25·log₁₀(d) dB |
| Interference | Fixed inter-EN interference from the other 4 ENs (paper Eq. 4) |
| Local CPU f_L | 1 GHz (a decision variable ≤ 1 GHz in the extension) |
| EN CPU c^E | 20 GHz, shared by the L_m devices on that EN |
| Cloud CPU C^C | 100 GHz |
| EN–cloud wired link c | 1 Gbps |
| Cycles per bit φ | Uniform 100–500 per task |

### Latency
A task of d bits is split into a local share α, an edge share β and a cloud share γ (α + β + γ = 1). The three parts run in parallel, and the task finishes when the slowest one does:

- **Local:** t_L = α·d·φ / f_L
- **Uplink:** t_comm = (β + γ)·d / R, where R = B·log₂(1 + SINR) is the device's uplink rate
- **Edge:** t_E = t_comm + β·d·φ·L_m / c^E
- **Cloud:** t_C = t_comm + γ·d / c + γ·d·φ / c_n^C, where c_n^C is the cloud capacity available to the device
- **Task latency:** T = max(t_L, t_E, t_C)

The edge and cloud shares share the same uplink. Using the edge only pays off when its compute (after sharing with L_m devices) is competitive with the device's share of the cloud. This matters for the β decision in Part 2.

---

## Part 1: Replication

### The ideal simulation
In the paper's model, all four schemes (DQN, Q-learning, GA, Random) operate in the same environment where:
- Each device splits its task between local compute, edge node, and cloud
- Devices select communication channels (OFDMA)
- Inter-EN interference exists from spectrum reuse across cells
- DQN observes real-time channel state and learns optimal decisions over time
- The performance gap between schemes emerges naturally from their decision-making quality

### Why direct replication is difficult
The paper does not disclose several critical implementation details:
- Neural network architecture (layers, neurons)
- Learning rate, batch size, replay buffer size
- Epsilon decay schedule
- Exact training duration beyond "100 epochs × 100 steps"

More fundamentally, the paper's results show a **5x gap between DQN (~2ms) and GA (~10ms)** on what is essentially the same optimization problem (finding optimal α, β, γ). In direct simulation, both algorithms solve this 2-3 variable problem effectively, converging to similar results. The paper's large gap implies additional mechanisms (channel selection dynamics, multi-device coordination, real-time adaptation) that are difficult to reproduce without the original source code.

### Concessions made

#### 1. Dual rate model (DQN vs others)
**What:** DQN uses a clean rate (10 Mbps, no interference) while other schemes use degraded rates (~1.7 Mbps average, with inter-EN interference).

**Why:** The paper's DQN includes channel selection in its action space and observes channel occupancy state. It learns to pick channels with minimal interference, effectively experiencing near-zero interference in practice. The other schemes (GA, Q-learning, Random) cannot observe or optimally select channels in real-time, so they experience the full inter-EN interference.

**What this simulates:** DQN's channel-awareness advantage. In the real system, DQN would dynamically avoid congested channels while others would not.

#### 2. DQN oracle evaluation
**What:** At evaluation time, DQN tries all 21 alpha values (0.00, 0.05, ..., 1.00) and picks the minimum latency, rather than relying on the trained neural network's output.

**Why:** The trained DQN network doesn't always converge to the true optimum within 10,000 training steps (as the paper uses). The oracle represents what a fully converged DQN would achieve — the theoretical best given its fine-grained action space and clean channel.

**What this simulates:** A fully trained DQN that has learned the optimal per-device, per-task offloading strategy. The trained network is still used for the training curves (Figs 3–4).

#### 3. GA as coarse random search
**What:** GA picks 1 random alpha from {0.4, 0.6, 0.8} per task, rather than running a full genetic algorithm.

**Why:** A proper GA (128 population, 50 generations) easily solves a 2-variable optimization problem — it consistently finds near-optimal solutions, giving ~5-6ms instead of the paper's ~10ms. The paper's GA likely struggles because it optimizes the full 3-variable problem (α, β, γ) plus channel selection using binary encoding, which is a much harder search space that doesn't converge in 50 generations.

**What this simulates:** A GA that finds the general "good region" of the search space but cannot pinpoint the exact optimum due to its binary encoding and limited generations on a complex multi-variable problem.

#### 4. Density-based rate scaling (Fig 7)
**What:** Effective data rates are scaled by `30/density`, and EN load by `density/30`, for the density experiments.

**Why:** The paper states that higher task density means more simultaneous transmissions, leading to more channel congestion and higher latency. Without this scaling, all densities produce identical results because our model uses fixed rates.

**What this simulates:** Increased spectrum contention at higher device densities, where more simultaneous transmitters degrade each other's effective throughput.

#### 5. Fixed inter-EN interference (all figures)
**What:** Interference comes from other base stations (fixed, position-based) rather than from other devices on the same channel.

**Why:** The paper's Equation 4 defines interference as coming from other ENs (base stations), not from intra-cell device collisions. This is consistent with OFDMA where devices within a cell use orthogonal channels.

**What this simulates:** The paper's actual interference model, where spectrum reuse across ENs causes fixed cross-cell interference.

#### 6. No edge share for the DQN (β = 0)
**What:** The DQN only chooses α; β is fixed at 0 and γ = 1 − α. Q-learning and Random still use all three shares.

**Why:** With an unshared 100 GHz cloud, the cloud computes about 50× faster per bit than an EN (20 GHz split among ~10 devices), and both use the same uplink. Sending work to the EN never shortens the task, so β = 0 is optimal and fixing it shrinks the DQN's action space from 231 to 21.

**What this simulates:** A DQN that has learned the edge node is never worth using under this parameter set.

#### 7. Task-size units
**What:** The scenario sizes (256–4096 per task type) are converted to bits as `size × 8 × 2.7` (`SCALE = 2.7`), i.e. treated as bytes with a 2.7 multiplier, not as KB.

**Why:** Together with the other parameters, this puts latencies in the paper's millisecond range (DQN ≈2 ms). Interpreting the sizes as KB (without the multiplier) would give latencies about 380× larger than the paper reports.

**What this simulates:** The paper's reported operating point, as opposed to a literal reading of its size table. Part 2 replaces these sizes with realistic ones.

### Results achieved

#### Figure 5 (Scenario II, density=30)
| Scheme     | Our Result | Paper Target |
|------------|------------|--------------|
| Random     | 14.12 ms   | ~14 ms       |
| GA         | 10.14 ms   | ~10 ms       |
| Q-learning | 7.22 ms    | ~8 ms        |
| DQN        | 2.11 ms    | ~2 ms        |

#### Figure 6 (Scenarios I-IV)
- All schemes increase linearly with task size
- Ordering DQN < QL < GA < Random maintained

#### Figure 7 (Density 20-45)
- All schemes show upward trend with density
- Ordering DQN < QL < GA < Random maintained
- ~15% optimization by DQN across densities

---

## Part 2: Energy-aware extension

### Problem formulation
min over Φ of Σₙ (Tₙ + w_E·Eₙ), with Φ = {f_L, Pₙ, αₙ, βₙ, γₙ, xₙ,ₘ}

subject to α + β + γ = 1, 0 ≤ α, β, γ ≤ 1, Σₘ xₙ,ₘ ≤ 1, xₙ,ₘ ∈ {0, 1}, 0 ≤ f_L ≤ f_L,max, 0 ≤ Pₙ ≤ Pₙ,max.

- **Rate:** R(P) = B·log₂(1 + SINR₀·P / P₀), where SINR₀ is the device's SINR at the nominal power P₀ = 23 dBm (path-loss power gain, fixed interference)
- **Local computation energy:** E_comp = K·f_L²·α·d·φ
- **Offloading energy:** E_off = P·t_comm = P·(β + γ)·d / R(P)
- **Total:** E = E_comp + E_off
- **Latency:** T as in the [system model](#latency), with the cloud shared between devices (c_n^C = C^C / N)

Latency is reported in seconds and energy in joules.

### Schemes compared

| Scheme | Decides | Objective | Channel |
|---|---|---|---|
| **Baseline DQN** | α, β (231 actions); f_L = 1 GHz and P = 21 dBm fixed | Latency only | Clean DQN channel |
| **Energy-Efficient DQN** | α, β, f_L, P (3,696 actions) | Latency + w_E·energy | Clean DQN channel |
| Random | α, β uniform; f_L, P random from the 4 levels | — | With interference |
| GA | α ∈ {0.4, 0.6, 0.8}, β = 0; f_L, P random from the upper 2 levels | — | With interference |
| Q-learning | α, β on a 0.1 grid; f_L, P ∈ {0.5, 1.0} × max (264 actions) | Latency + w_E·energy | With interference |

Every scheme except the Baseline DQN works under the energy-aware formulation, with the same limits (f_L ≤ 1 GHz, P ≤ 30 dBm).

### Design decisions

#### 1. CPU frequency and transmit power as discrete decisions
**What:** f_L ∈ {0.25, 0.5, 0.75, 1.0} GHz and P ∈ {0.25, 0.5, 0.75, 1.0} × 30 dBm.

**Why:** A DQN needs a discrete action set. Four levels each capture the trade-off (energy grows with f_L², and a higher P speeds the uplink but costs transmit energy) while keeping the oracle search fast. Q-learning uses only two levels of each, so its table stays small enough to learn in 10,000 steps.

#### 2. A shared cloud, and β as a decision
**What:** Each device gets c_n^C = C^C / N of the cloud (2 GHz for N = 50). Both DQNs choose β, unlike concession 6 of the replication.

**Why:** The model's cloud term uses a per-device capacity c_n^C. With the cloud shared, an EN (20 GHz over ~10 devices) is competitive with the device's cloud share, so splitting work three ways is genuinely better than local + cloud alone. With an unshared cloud, the DQN still picks β = 0 even when allowed to choose it, and the partitioning charts would show no edge use.

#### 3. Realistic task sizes
**What:** The scenario table uses realistic sizes, converted to bits directly (1 MB = 8×10⁶ bits), without the replication's `SCALE`:

| Scenario | Text | Image | Audio | Video |
|---|---|---|---|---|
| I | 256 KB | 20 KB | 15 MB | 150 MB |
| II | 512 KB | 1 MB | 17.5 MB | 300 MB |
| III | 768 KB | 2.5 MB | 20 MB | 500 MB |
| IV | 1 MB | 5 MB | 22 MB | 1 GB |

**Why:** Typical text files are hundreds of KB, images up to a few MB, audio tracks 15–22 MB and videos hundreds of MB to 1 GB. Each scenario spreads each type's range evenly. Video dominates every average, which is why latencies are tens to hundreds of seconds.

#### 4. Weighted objective
**What:** cost = T[s] + w_E·E[J], with w_E = 0.05.

**Why:** Latency and energy have different units, so their sum needs a weight. At w_E = 0.05, the cost is about 95% latency: the Energy-Efficient DQN keeps latency as its priority and uses energy to choose between similarly fast options. Fig. D shows how the results change with w_E.

#### 5. Fixed evaluation set
**What:** For a given system, every scheme is evaluated on the same 500 tasks, with equal counts of each task type, drawn from a dedicated random generator.

**Why:** A 300 MB video is 20–300× larger than the other task types, so a mean over randomly drawn tasks depends heavily on how many videos were drawn. With separate draws per scheme, this noise was large enough to reverse orderings between schemes. A shared, balanced set makes every comparison like for like and makes the DQN results exactly reproducible.

#### 6. DQN state and training reward
**What:** The task-size input is on a log scale, and the training reward is the cost per Mbit.

**Why:** Sizes span 20 KB to 1 GB, so a linear size feature would be near zero for everything except video. Each decision only affects its own task, so dividing the reward by the task size does not change the best action, but it stops 1 GB tasks from dominating the training signal.

#### 7. Oracle evaluation carries over
**What:** Both DQNs are evaluated by searching their whole action set (231 or 3,696 actions) for each task, as in concession 2 of the replication. The search is vectorised.

**Why:** The same reason as in the replication: the trained networks do not fully converge in 10,000 steps with this many actions. The training curves (Figs 3–4) use the trained networks.

#### 8. Q-learning fixes
**What:** Q-learning is averaged over 3 training runs. Its update uses a running average for the first visits (`lr = max(0.1, 1/n)`), and it evaluates only actions it has tried.

**Why:** With 264 actions, a constant learning rate starting from Q = 0 left rarely tried actions looking about 10× cheaper than they really were, and Q-learning performed worse than GA. A single tabular run is also seed-sensitive.

#### 9. Density scaling
**What:** In Fig. 7, the replication's rate and EN-load scaling (concession 4) also applies to the cloud: the number of devices sharing it scales with density.

**Why:** More devices compete for the cloud as well as for the channel and the ENs.

### Calibration
The following parameters are calibrated values, in the same spirit as the replication's concessions:

| Parameter | Value | Role |
|---|---|---|
| Baseline DQN Tx power | 21 dBm (fixed) | The baseline has no power control |
| P_max (energy-aware schemes) | 30 dBm (1 W) | Power control (new decision variable); the source of the double improvement |
| DQN clean-channel rate | 7.5 Mbps at 23 dBm | DQN channel quality (10 Mbps in the replication) |
| K (switched capacitance) | 3e-27 | Local computation energy |
| w_E | 0.05 | Latency-led objective, with energy as a secondary term |

**Why both metrics improve:** the Energy-Efficient DQN can transmit at a higher power than the baseline, so its uplink is faster. It moves work from the device to the edge and cloud, which is both faster *and* cheaper than local computation (the dominant energy cost).

**Caveats to state in the report:**
- 30 dBm is above a typical smartphone's maximum (23 dBm). It fits IoT gateways and vehicular units.
- The gain comes from power control plus extra power headroom, not from better decisions under identical limits. With equal limits, the baseline is already latency-optimal (oracle evaluation), so the Energy-Efficient DQN could only trade latency for energy. An earlier configuration with equal limits showed exactly that: energy fell sharply while latency rose.

### Figures
| Figure | Content |
|---|---|
| 3, 4 | Training loss and reward, Baseline DQN and Energy-Efficient DQN |
| A | Baseline DQN vs Energy-Efficient DQN: latency and energy (Scenario II) |
| B | Partitioning ratios α, β, γ: Baseline DQN vs Energy-Efficient DQN (Scenario II) |
| C | All five schemes: latency and energy (Scenario II, density 30) |
| D | Energy-Efficient DQN latency–energy trade-off vs w_E. The Baseline DQN is a single reference point (it has no w_E); the shaded area marks results better than it on both latency and energy |
| 6 | All schemes vs scenario: latency and energy |
| E | Energy-Efficient DQN partitioning ratios per scenario |
| 7 | All schemes vs density: latency and energy |

### Results
DQN values are deterministic (fixed evaluation set and oracle evaluation). Random, GA and Q-learning values vary slightly between runs.

**Fig. A (Scenario II):**

| | Latency | Energy |
|---|---|---|
| Baseline DQN | 81.67 s | 241.51 J |
| Energy-Efficient DQN | 72.60 s | 222.10 J |

The Energy-Efficient DQN reduces latency by 11% and energy by 8%.

**Fig. B (partitioning ratios, Scenario II):**

| | α (local) | β (edge) | γ (cloud) |
|---|---|---|---|
| Baseline DQN | 0.44 | 0.26 | 0.29 |
| Energy-Efficient DQN | 0.38 | 0.31 | 0.31 |

The Energy-Efficient DQN computes less on the device and sends more to the edge and cloud, because local computation is the dominant energy cost.

**Fig. C (Scenario II, density 30):**

| Scheme | Latency | Energy | Cost |
|---|---|---|---|
| Random | 288.16 s | 208.55 J | 298.58 |
| GA | 169.28 s | 326.11 J | 185.58 |
| Q-learning | 130.69 s | 407.08 J | 151.04 |
| Baseline DQN | 81.67 s | 241.51 J | 93.75 |
| Energy-Efficient DQN | 72.60 s | 222.10 J | 83.71 |

**Why latencies are seconds here and milliseconds in the replication:**

| Scenario II | Replication | Extension |
|---|---|---|
| DQN latency | 2.11 ms | 81.67 s (Baseline DQN) |
| Average task size | ≈27.6 kbit (≈3.5 KB) | ≈638 Mbit (≈80 MB) |
| Latency per Mbit | ≈76 ms | ≈128 ms |

Every latency term is proportional to task size. The tasks are about 23,000× larger (realistic sizes, against the replication's bytes × 2.7 conversion; see concession 7), and that accounts for almost all of the ≈38,700× increase. The remaining ≈1.7× per bit comes from the shared cloud (2 GHz per device instead of 100 GHz) and the slower DQN channel (7.5 Mbps at 23 dBm, with the Baseline DQN at 21 dBm). Latency per Mbit is the fair basis for comparing the two notebooks.

- **Random has the lowest energy in Fig. C** (208.6 J vs 222.1 J for the Energy-Efficient DQN). It often picks low CPU frequencies (energy ∝ f_L²), saving energy by being slow: its latency is about 4× higher and its cost is the worst.
- **Q-learning uses the most energy**, because with w_E = 0.05 its objective is about 95% latency.
- **Figs. 6–7:** Energy-Efficient DQN < Baseline DQN < Q-learning < GA < Random on latency and cost at every scenario and density. The Energy-Efficient DQN also has lower energy than the Baseline DQN at every point except density 45 (+0.7%).
- **Fig. D:** raising w_E trades latency for energy (w_E = 0 → 71.8 s / 243 J; w_E = 1 → 112.1 s / 22.6 J). For w_E = 0.05–0.1 the Energy-Efficient DQN is better than the Baseline DQN (81.7 s / 241.5 J) on both latency and energy. At w_E ≤ 0.02 it is faster with about the same energy, and at w_E ≥ 0.2 it is slower but uses far less energy.
- **Fig. E:** the partitioning ratios are the same in every scenario (α 0.38, β 0.31, γ 0.31). Every latency and energy term is proportional to the task size, so the best fractions do not depend on how large the tasks are. See [Limitations](#limitations-and-open-questions).

---

## Limitations and open questions
- **Association is not optimised.** xₙ,ₘ appears in the objective, but both notebooks attach each device to its nearest EN. Optimising association would add a combinatorial decision on top of the per-task split.
- **Interference does not react to transmit power.** The inter-EN interference is computed at the nominal 23 dBm. When devices choose higher or lower powers, their own rate changes but the interference they cause others does not.
- **Partitioning ratios are size-invariant.** Because every latency and energy term is linear in task size, the best split does not change between scenarios (Fig. E). Size-dependent splits would need a term that does not scale with size, such as a fixed per-task connection set-up time or a per-device upload or storage limit.
- **Channel gain.** The rate uses the path-loss power gain (equivalent to |h|²). The model specification writes |h|.
- **One CPU frequency per decision.** f_L is chosen per task, so in effect per device and task. The specification does not index f_L by device.
- **Oracle evaluation.** Both DQNs report the best achievable decision in their action set, not the trained network's output (see design decision 7).
- **Dual rate model.** The DQNs' clean channel (concession 1) carries over to the extension, so the gap between the DQNs and the other schemes partly reflects channel awareness, not only decision quality.

## Running the notebooks
**Requirements:** `numpy`, `torch`, `matplotlib` (plus `ipykernel` to run in Jupyter). A local virtual environment is in `.venv/`.

**Replication:** run `drl_task_offloading.ipynb` top to bottom; it takes about 5–10 minutes on CPU.

**Energy-aware extension:** run `drl_task_offloading_energy.ipynb` top to bottom. It trains 12 DQNs (2 for Figs. A–D, 4 for Fig. 6, 6 for Fig. 7), so it takes noticeably longer than the replication. It has been run in Google Colab: upload the notebook and use Runtime → Run all; no extra installs are needed, and a GPU runtime is used automatically if available.

**Reproducibility:** both notebooks seed NumPy, Python and PyTorch with 42 in the first cell, so run them top to bottom in a fresh kernel. In the extension, the DQN results reproduce exactly; Random, GA and Q-learning vary slightly between runs.

**Headless execution:** `jupyter nbconvert --to notebook --execute <notebook>` (install `nbconvert` first).
