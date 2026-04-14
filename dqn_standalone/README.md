# DQN for MEC Task Offloading — Standalone Experiment

A single self-contained Python file that trains a Deep Q-Network to solve the
task offloading problem in a Mobile Edge Computing network with heterogeneous
tasks.

Based on: [Jiang et al., IEEE IoT Journal, 2025](../paper/Deep-Reinforcement-Learning-Based_Task_Offloading_and_Resource_Allocation_in_Mobile_Edge_Computing_Network_With_Heterogeneous_Tasks.pdf)

---

## Quick Start

```bash
pip install torch numpy matplotlib
python dqn_mec.py
```

Runs in ~2 minutes on CPU. Outputs go to `results/`.

---

## What It Does

Trains a DQN agent to decide, for each IoT device in a simulated MEC network:
- What fraction of a task to process locally (α)
- What fraction to offload to the nearest edge server (β)
- What fraction to send to the cloud (γ = 1 − α − β)
- Which wireless channel to use

Then evaluates the trained agent across four task-size scenarios (I–IV) and
six task density levels (20–45 devices).

---

## Design Decisions and Justifications

### Why DQN over other RL algorithms?

The paper specifically proposes DQN as the solution method. DQN is a natural
fit here because:
- The state space is continuous (task sizes, channel states) but
  low-dimensional (13 features), so a small neural network can approximate
  Q-values effectively.
- The action space is discrete (choose a ratio + channel combo), which is
  exactly what DQN is designed for.
- Experience replay breaks the temporal correlation between consecutive
  device decisions within an episode, which is important because devices
  connected to the same EN share channel state.

More advanced algorithms like PPO or SAC could work but would be overkill for
this problem size and would obscure the paper's contribution.

### Why discretize α and β into 6 levels instead of 11?

The paper uses DQN (discrete actions) but describes continuous ratios. We need
to discretize. With 11 levels (0.0, 0.1, …, 1.0):
- 66 valid (α, β) pairs × 10 channels = 660 actions
- The agent needs to explore each action many times to learn its Q-value
- With limited training budget, most actions are rarely visited

With 6 levels (0.0, 0.2, …, 1.0):
- 21 valid pairs × 10 channels = 210 actions
- Each action gets ~3× more training exposure
- The 0.2 granularity barely affects solution quality — the optimal split
  rarely needs finer precision than 20% increments

### Why batch updates per episode instead of per device?

The original approach updated the neural network after every single device
action (30 updates per episode × 30 episodes per epoch = 900 gradient steps
per epoch). This was:
- Extremely slow (each step requires a forward + backward pass)
- Unnecessary — the replay buffer already decorrelates experiences

The current approach: collect all 30 device experiences, then do one batch
update. Same learning signal, 30× fewer gradient steps, runs in minutes
instead of hours.

### Why normalize rewards within each mini-batch?

Raw rewards (negative latency) vary wildly across scenarios:
- Scenario I: rewards around −0.5 to −2
- Scenario IV: rewards around −10 to −30

Without normalization, the loss magnitude in Scenario IV is ~100× larger than
Scenario I, causing unstable gradients and poor convergence. Batch
normalization (zero mean, unit variance) keeps the loss scale consistent
regardless of the absolute latency values.

### Why a learning rate scheduler?

Early training needs a higher LR to explore the Q-value landscape quickly.
Later training benefits from a lower LR to fine-tune the policy without
overshooting. We halve the LR every 100 epochs — a simple but effective
schedule.

### Why gradient clipping?

Large tasks (Scenario IV) can produce outlier rewards that cause gradient
spikes. Clipping the gradient norm to 1.0 prevents any single batch from
destabilizing the network weights. This is standard practice in DRL.

### Why two hidden layers of 128 neurons?

The state vector is only 13-dimensional. A larger network (e.g., 3×256 = 189K
params) is harder to train with limited data and much slower on CPU. The
2×128 architecture (45K params) is sufficient to capture the mapping from
(task type, data size, channel state) → optimal action, and trains ~4× faster.

### Why ε-greedy with decay factor 0.97?

With decay 0.97, epsilon reaches ~0.05 by epoch 100 and ~0.01 by epoch 150.
This means:
- First ~50 epochs: heavy exploration (ε > 0.2), filling the replay buffer
  with diverse experiences
- Epochs 50–100: transitioning to exploitation
- Epochs 100+: nearly pure exploitation, refining the learned policy

Slower decay (0.99) wastes training budget on random actions. Faster decay
(0.9) risks converging to a suboptimal policy before exploring enough.

### Why a target network synced every 5 epochs?

The target network provides stable Q-value targets during training. Syncing
too often (every epoch) makes it equivalent to not having one. Syncing too
rarely (every 50 epochs) means the target is stale. Every 5 epochs is a
good balance — the paper doesn't specify the exact frequency.

---

## Suggestions for Improvement

### 1. Double DQN
Standard DQN overestimates Q-values because it uses the same network to both
select and evaluate actions. Double DQN fixes this by using the eval network
to select the best action but the target network to evaluate it. One-line
change in the update method:

```python
# Current (standard DQN):
next_q = self.target_net(ns).max(1)[0]

# Double DQN:
best_actions = self.eval_net(ns).argmax(1)
next_q = self.target_net(ns).gather(1, best_actions.unsqueeze(1)).squeeze(1)
```

### 2. Dueling DQN
Split the Q-network into two streams: one estimates the state value V(s), the
other estimates the advantage A(s, a) of each action. Q(s, a) = V(s) + A(s, a)
− mean(A). This helps the network learn which states are inherently good or
bad, independent of the action taken.

### 3. Prioritised Experience Replay
Instead of sampling uniformly from the replay buffer, sample transitions with
higher TD-error more frequently. This focuses learning on surprising or
poorly-predicted experiences, which speeds up convergence significantly.

### 4. Multi-Agent DRL
The current approach trains a single shared agent that makes decisions for all
devices sequentially. In reality, each device could have its own agent that
learns cooperatively. Multi-agent approaches (e.g., QMIX, MAPPO) could
capture inter-device coordination — for example, avoiding channel collisions.

### 5. Continuous Action Space
Instead of discretising α and β, use an actor-critic method (e.g., DDPG, TD3,
SAC) that outputs continuous values directly. This eliminates the granularity
tradeoff entirely and could find finer-grained optimal splits.

### 6. Dynamic Environment
The current simulation is static — device positions and EN associations don't
change between episodes. Adding mobility (devices moving between time slots)
and time-varying channel conditions would make the problem more realistic and
better showcase DRL's adaptability over static methods like GA.

### 7. Reward Shaping
The current reward is simply −latency. Adding auxiliary reward terms could
speed up learning:
- Penalty for choosing an already-occupied channel
- Bonus for balanced load across ENs
- Penalty for extreme splits (all local or all cloud)

### 8. Transfer Learning Across Scenarios
Currently we train a separate agent per scenario. A single agent trained on a
mix of scenarios (curriculum learning) could generalize across task sizes
without retraining, which is more practical for real deployments.

---

## File Structure

```
dqn_standalone/
├── dqn_mec.py     # Everything in one file — run this
├── README.md      # This writeup
└── results/       # Generated after running (convergence.png, scenarios.png, densities.png)
```

---

## Output

| File                      | Description                                    |
|---------------------------|------------------------------------------------|
| `results/convergence.png` | Training loss and reward curves (side by side) |
| `results/scenarios.png`   | Average latency across Scenarios I–IV          |
| `results/densities.png`   | Average latency vs. task density (20–45)       |
