"""
main.py — Experiment runner that reproduces Figs. 3-7 from the paper.

This is the entry point.  Run it with:
    python main.py

It will:
  1. Train a DQN agent and plot loss / reward convergence  (Figs. 3 & 4)
  2. Compare DQN against Random, Q-Learning, and GA       (Fig. 5)
  3. Repeat the comparison across four task-size scenarios  (Fig. 6)
  4. Repeat the comparison across six task densities        (Fig. 7)

All figures are saved as PNG files in the `results/` folder.

Paper reference:
  Jiang et al., "Deep-Reinforcement-Learning-Based Task Offloading and
  Resource Allocation in Mobile Edge Computing Network With Heterogeneous
  Tasks", IEEE Internet of Things Journal, Vol. 12, No. 8, April 2025.
"""

import os
import random

import numpy as np
import torch

# Project modules
from config import NUM_EPOCHS, STEPS_PER_EPOCH, TARGET_UPDATE_FREQ
from environment import MECEnvironment
from dqn_agent import DQNAgent
from baselines import evaluate_random, evaluate_qlearning, evaluate_ga
from plotting import (
    plot_loss_convergence,
    plot_reward_convergence,
    plot_scheme_comparison,
    plot_scenario_comparison,
    plot_density_comparison,
)


# ═══════════════════════════════════════════════════════════════════════
# DQN training loop
# ═══════════════════════════════════════════════════════════════════════

def train_dqn(env: MECEnvironment,
              num_epochs: int = NUM_EPOCHS,
              steps_per_epoch: int = STEPS_PER_EPOCH,
              verbose: bool = True):
    """
    Train a DQN agent on the given MEC environment.

    Each "epoch" runs `steps_per_epoch` full episodes.  In each episode
    the environment is reset (new random tasks), and the agent makes one
    decision per device sequentially.

    Parameters
    ----------
    env             : MECEnvironment — the simulation to train on
    num_epochs      : int  — total training epochs
    steps_per_epoch : int  — episodes per epoch
    verbose         : bool — print progress every 50 epochs

    Returns
    -------
    agent          : DQNAgent   — the trained agent
    loss_history   : list[float] — average loss per epoch
    reward_history : list[float] — average episode reward per epoch
    """
    agent = DQNAgent(env.state_dim, env.num_actions)
    loss_history   = []
    reward_history = []

    for epoch in range(num_epochs):
        epoch_loss   = 0.0
        epoch_reward = 0.0
        loss_count   = 0

        for step in range(steps_per_epoch):
            # Start a fresh episode (new random tasks for all devices)
            env.reset()
            episode_reward = 0.0

            # Each device takes one action in sequence
            for n in range(env.num_devices):
                state  = env._get_state(n)
                action = agent.select_action(state)

                next_state, reward, done, _ = env.step(n, action)

                # Store the experience and learn from a mini-batch
                agent.store(state, action, reward, next_state, done)
                loss = agent.update()

                epoch_loss += loss
                loss_count += 1
                episode_reward += reward

            epoch_reward += episode_reward

        # After each epoch: decay exploration and periodically sync target net
        agent.decay_epsilon()
        if (epoch + 1) % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        avg_loss   = epoch_loss / max(loss_count, 1)
        avg_reward = epoch_reward / steps_per_epoch
        loss_history.append(avg_loss)
        reward_history.append(avg_reward)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{num_epochs}  "
                  f"Loss: {avg_loss:.4f}  "
                  f"Reward: {avg_reward:.2f}  "
                  f"ε: {agent.epsilon:.4f}")

    return agent, loss_history, reward_history


# ═══════════════════════════════════════════════════════════════════════
# DQN evaluation (greedy — no exploration)
# ═══════════════════════════════════════════════════════════════════════

def evaluate_dqn(env: MECEnvironment, agent: DQNAgent) -> float:
    """
    Run the trained DQN agent greedily (ε = 0) on a fresh environment
    and return the average per-device latency.
    """
    env.reset()
    total_latency = 0.0

    for n in range(env.num_devices):
        state = env._get_state(n)
        # greedy=True → always pick the best action, no randomness
        action = agent.select_action(state, greedy=True)
        _, _, _, latency = env.step(n, action)
        total_latency += latency

    return total_latency / env.num_devices


# ═══════════════════════════════════════════════════════════════════════
# Helper: evaluate all four schemes on a given environment config
# ═══════════════════════════════════════════════════════════════════════

def evaluate_all_schemes(agent, scenario, num_devices, task_density,
                         num_runs=3):
    """
    Run each of the four schemes `num_runs` times and return the
    averaged latency for each.

    Returns dict: {"Random": float, "Q-Learning": float, "GA": float, "DQN": float}
    """
    results = {"Random": 0.0, "Q-Learning": 0.0, "GA": 0.0, "DQN": 0.0}

    for _ in range(num_runs):
        env_r = MECEnvironment(num_devices, scenario, task_density)
        results["Random"] += evaluate_random(env_r)

        env_q = MECEnvironment(num_devices, scenario, task_density)
        results["Q-Learning"] += evaluate_qlearning(env_q, num_episodes=150)

        env_g = MECEnvironment(num_devices, scenario, task_density)
        results["GA"] += evaluate_ga(env_g)

        env_d = MECEnvironment(num_devices, scenario, task_density)
        results["DQN"] += evaluate_dqn(env_d, agent)

    for k in results:
        results[k] /= num_runs

    return results


# ═══════════════════════════════════════════════════════════════════════
# Main experiment pipeline
# ═══════════════════════════════════════════════════════════════════════

def main():
    # ── Reproducibility ──────────────────────────────────────────────
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs("results", exist_ok=True)

    # Default setting: 30 devices, Scenario II, density 30
    num_devices = 30

    # ─────────────────────────────────────────────────────────────────
    # Experiment 1: DQN training convergence  (Figs. 3 & 4)
    # ─────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Experiment 1: DQN Training Convergence (Figs. 3 & 4)")
    print("=" * 60)

    env = MECEnvironment(num_devices, scenario="II", task_density=30)
    agent, loss_history, reward_history = train_dqn(env)

    plot_loss_convergence(loss_history)
    plot_reward_convergence(reward_history)

    # ─────────────────────────────────────────────────────────────────
    # Experiment 2: Scheme comparison  (Fig. 5)
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Experiment 2: Scheme Comparison (Fig. 5)")
    print("=" * 60)

    scheme_results = evaluate_all_schemes(
        agent, scenario="II", num_devices=num_devices,
        task_density=30, num_runs=5,
    )
    for name, val in scheme_results.items():
        print(f"  {name:12s}: {val:.4f} s")

    plot_scheme_comparison(scheme_results)

    # ─────────────────────────────────────────────────────────────────
    # Experiment 3: Scenario comparison  (Fig. 6)
    #   Train a separate DQN per scenario, then compare all schemes.
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Experiment 3: Scenario Comparison (Fig. 6)")
    print("=" * 60)

    scenario_results = {}
    for sc in ["I", "II", "III", "IV"]:
        print(f"\n  --- Scenario {sc} ---")
        env_sc = MECEnvironment(num_devices, scenario=sc, task_density=30)
        agent_sc, _, _ = train_dqn(
            env_sc, num_epochs=250, steps_per_epoch=25, verbose=True,
        )
        sc_res = evaluate_all_schemes(
            agent_sc, scenario=sc, num_devices=num_devices,
            task_density=30, num_runs=3,
        )
        scenario_results[sc] = sc_res
        print(f"  Results: { {k: f'{v:.4f}' for k, v in sc_res.items()} }")

    plot_scenario_comparison(scenario_results)

    # ─────────────────────────────────────────────────────────────────
    # Experiment 4: Density comparison  (Fig. 7)
    #   Train a separate DQN per density level, then compare.
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Experiment 4: Density Comparison (Fig. 7)")
    print("=" * 60)

    density_results = {}
    for density in [20, 25, 30, 35, 40, 45]:
        print(f"\n  --- Density = {density} ---")
        env_d = MECEnvironment(density, scenario="II", task_density=density)
        agent_d, _, _ = train_dqn(
            env_d, num_epochs=250, steps_per_epoch=25, verbose=True,
        )
        d_res = evaluate_all_schemes(
            agent_d, scenario="II", num_devices=density,
            task_density=density, num_runs=3,
        )
        density_results[density] = d_res
        print(f"  Results: { {k: f'{v:.4f}' for k, v in d_res.items()} }")

    plot_density_comparison(density_results)

    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("All experiments complete!  Check the results/ folder for PNGs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
