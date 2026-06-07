# DRL-Based Task Offloading in Mobile Edge Computing

A notebook-based replication of the experiments from:

> **Jiang, T., Chen, Z., Zhao, Z., Feng, M., & Zhou, J.** (2025).
> *Deep-Reinforcement-Learning-Based Task Offloading and Resource Allocation
> in Mobile Edge Computing Network With Heterogeneous Tasks.*
> IEEE Internet of Things Journal, 12(8), 10899–10906.
> [Full Paper](paper/Deep-Reinforcement-Learning-Based_Task_Offloading_and_Resource_Allocation_in_Mobile_Edge_Computing_Network_With_Heterogeneous_Tasks.pdf)

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
├── DRL_Task_Offloading.ipynb   # Complete self-contained implementation
├── paper/                      # Reference paper PDF
├── results/                    # Optional output directory
└── README.md
```

Everything needed to run experiments (environment, DQN, baselines, plotting,
training loop) is contained in **`DRL_Task_Offloading.ipynb`**.

---

## How to Run

### Prerequisites

- Python 3.10+
- Jupyter Notebook or JupyterLab

### Installation

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install jupyter torch numpy matplotlib
```

### Run the Notebook

```bash
jupyter notebook DRL_Task_Offloading.ipynb
```

Or with JupyterLab:

```bash
jupyter lab DRL_Task_Offloading.ipynb
```

Run all cells from top to bottom to reproduce the full pipeline.

---

## What Gets Produced

- Figures are displayed inline in the notebook output.
- Depending on notebook settings, outputs may also be persisted in notebook state.

---

## Tuning

All key hyperparameters are defined in the **Configuration** section of
`DRL_Task_Offloading.ipynb`.

Common knobs:

| Parameter         | Default | Effect                                                      |
|-------------------|---------|-------------------------------------------------------------|
| `NUM_EPOCHS`      | 200     | More epochs -> better DQN but slower training               |
| `STEPS_PER_EPOCH` | 30      | More steps -> more data per epoch                           |
| `EPSILON_DECAY`   | 0.97    | Lower -> faster shift from exploration to exploitation      |
| `LEARNING_RATE`   | 0.001   | Adam LR; lower for more stable but slower learning          |
| `HIDDEN_DIM`      | 128     | Neurons per hidden layer; increase for harder problems      |
| `NUM_CHANNELS`    | 10      | More channels -> larger action space                        |
| `BATCH_SIZE`      | 64      | Larger batches -> smoother gradient estimates               |

---

## License

This code is provided for educational and research purposes. The original
paper is © 2024 IEEE.
