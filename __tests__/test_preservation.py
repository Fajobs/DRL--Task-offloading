"""
test_preservation.py — Preservation property tests for DRL Task Offloading notebook.

These tests observe and validate behaviours that MUST remain unchanged after the bugfix.
They parse the notebook JSON directly to verify structural properties of the code.

Observation-first methodology:
- Observe on UNFIXED code, write tests capturing those observations
- Tests should PASS on unfixed code (confirming baseline)
- Tests should PASS on fixed code (confirming preservation)
"""

import os
import json
import re
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_PATH = os.path.join(ROOT_DIR, "DRL_Task_Offloading.ipynb")


def load_notebook_cells():
    """Load and return notebook cell sources as joined strings."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            cells.append(source)
    return cells


def get_full_source():
    """Get all code cells concatenated into one string."""
    cells = load_notebook_cells()
    return "\n".join(cells)


def find_function_body(full_source, func_name):
    """Extract a function body from the full source by name."""
    # Find function definition
    pattern = rf"(def {func_name}\(.*?\):.*?)(?=\ndef |\nclass |\Z)"
    match = re.search(pattern, full_source, re.DOTALL)
    if match:
        return match.group(1)
    return None


# ═══════════════════════════════════════════════════════════════════════
# TEST RESULTS TRACKING
# ═══════════════════════════════════════════════════════════════════════

results = []


def record(test_name, passed, detail=""):
    results.append((test_name, passed, detail))
    status = "PASS" if passed else "FAIL"
    msg = f"  [{status}] {test_name}"
    if detail:
        msg += f" — {detail}"
    print(msg)


# ═══════════════════════════════════════════════════════════════════════
# PROPERTY TESTS
# ═══════════════════════════════════════════════════════════════════════

def test_latency_parallel_execution_model():
    """
    Property: For any device configuration, latency is computed as max of
    three parallel subtask latencies (t_L, t_E, t_C).

    Observation: _compute_latency() ends with `return max(t_L, t_E, t_C)`
    This confirms the parallel execution model from Eq. 13 of the paper.
    """
    full_source = get_full_source()
    func_body = find_function_body(full_source, "_compute_latency")

    if func_body is None:
        record("Latency parallel model (_compute_latency exists)", False,
               "_compute_latency function not found")
        return

    record("Latency parallel model (_compute_latency exists)", True)

    # Check that the function uses max(t_L, t_E, t_C) as return value
    # The pattern should match: return max(t_L, t_E, t_C)
    has_max_return = bool(re.search(r"return\s+max\(t_L,\s*t_E,\s*t_C\)", func_body))
    record("Latency uses max(t_L, t_E, t_C) parallel model", has_max_return,
           "Expected return max(t_L, t_E, t_C) in _compute_latency")

    # Verify t_L, t_E, t_C are computed from local, edge, cloud paths
    has_t_L = bool(re.search(r"t_L\s*=\s*t_local", func_body))
    has_t_E = bool(re.search(r"t_E\s*=\s*t_comm_en\s*\+\s*t_en", func_body))
    has_t_C = bool(re.search(r"t_C\s*=\s*t_comm_en\s*\+\s*t_comm_cloud\s*\+\s*t_cloud", func_body))

    record("t_L = t_local (local path)", has_t_L)
    record("t_E = t_comm_en + t_en (edge path)", has_t_E)
    record("t_C = t_comm_en + t_comm_cloud + t_cloud (cloud path)", has_t_C)


def test_scenarios_match_table_ii():
    """
    Property: For any task scenario key (I–IV), SCENARIOS values match Table II.

    Observation: SCENARIOS dict contains:
      I:   text=256, image=512, audio=768, video=1024
      II:  text=512, image=1024, audio=1536, video=2048
      III: text=768, image=1536, audio=2304, video=3072
      IV:  text=1024, image=2048, audio=3072, video=4096
    These are task sizes in KB from Table II of the paper.
    """
    full_source = get_full_source()

    # Expected Table II values (in KB)
    expected_scenarios = {
        "I":   {"text": 256,  "image": 512,  "audio": 768,  "video": 1024},
        "II":  {"text": 512,  "image": 1024, "audio": 1536, "video": 2048},
        "III": {"text": 768,  "image": 1536, "audio": 2304, "video": 3072},
        "IV":  {"text": 1024, "image": 2048, "audio": 3072, "video": 4096},
    }

    # Check SCENARIOS dict exists and contains expected values
    has_scenarios = "SCENARIOS" in full_source
    record("SCENARIOS dict exists", has_scenarios)

    if not has_scenarios:
        return

    # Check each scenario key and its values
    all_match = True
    for scenario_key, expected_values in expected_scenarios.items():
        for task_type, expected_size in expected_values.items():
            # Look for the pattern: "task_type": size in the scenario
            # Pattern accounts for the notebook source format
            pattern = rf'"{task_type}":\s*{expected_size}'
            if not re.search(pattern, full_source):
                record(f"Scenario {scenario_key} {task_type}={expected_size}",
                       False, f"Expected {task_type}: {expected_size} in scenario {scenario_key}")
                all_match = False

    if all_match:
        record("All SCENARIOS values match Table II", True)


def test_dqn_alpha_levels_step_02():
    """
    Property: DQN action-space discretization uses step 0.2 (6 levels).
    This must NOT be affected by the Q-Learning fix.

    Observation: ALPHA_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    This gives 6 levels with step size 0.2 for the DQN action space.
    """
    full_source = get_full_source()

    # Check ALPHA_LEVELS definition
    alpha_pattern = r"ALPHA_LEVELS\s*=\s*\[0\.0,\s*0\.2,\s*0\.4,\s*0\.6,\s*0\.8,\s*1\.0\]"
    has_alpha = bool(re.search(alpha_pattern, full_source))
    record("ALPHA_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]", has_alpha,
           "DQN uses 6 alpha levels with step 0.2")

    # Verify BETA_LEVELS also has step 0.2 (6 levels)
    beta_pattern = r"BETA_LEVELS\s*=\s*\[0\.0,\s*0\.2,\s*0\.4,\s*0\.6,\s*0\.8,\s*1\.0\]"
    has_beta = bool(re.search(beta_pattern, full_source))
    record("BETA_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]", has_beta,
           "DQN uses 6 beta levels with step 0.2")

    # Verify the count of levels is 6
    alpha_match = re.search(r"ALPHA_LEVELS\s*=\s*\[(.*?)\]", full_source)
    if alpha_match:
        levels = [x.strip() for x in alpha_match.group(1).split(",")]
        count_ok = len(levels) == 6
        record("ALPHA_LEVELS has exactly 6 elements", count_ok,
               f"Found {len(levels)} elements")
    else:
        record("ALPHA_LEVELS has exactly 6 elements", False, "Could not parse ALPHA_LEVELS")


def test_qnetwork_architecture():
    """
    Property: QNetwork has two hidden layers of 128 neurons each.

    Observation: QNetwork.__init__ builds nn.Sequential with:
      Linear(state_dim, 128), ReLU,
      Linear(128, 128), ReLU,
      Linear(128, action_dim)
    """
    full_source = get_full_source()

    # Check QNetwork class exists
    has_qnetwork = "class QNetwork" in full_source
    record("QNetwork class exists", has_qnetwork)

    if not has_qnetwork:
        return

    # Check HIDDEN_DIM = 128
    hidden_pattern = r"HIDDEN_DIM\s*=\s*128"
    has_hidden_128 = bool(re.search(hidden_pattern, full_source))
    record("HIDDEN_DIM = 128", has_hidden_128)

    # Check the network has two hidden layers (3 Linear layers total)
    # Pattern: nn.Linear(..., hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, ...)
    network_pattern = (
        r"nn\.Linear\(state_dim,\s*hidden\).*?"
        r"nn\.ReLU\(\).*?"
        r"nn\.Linear\(hidden,\s*hidden\).*?"
        r"nn\.ReLU\(\).*?"
        r"nn\.Linear\(hidden,\s*action_dim\)"
    )
    has_two_hidden = bool(re.search(network_pattern, full_source, re.DOTALL))
    record("QNetwork has 2 hidden layers (3 Linear layers)", has_two_hidden,
           "Expected: Linear(in, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, out)")


def test_fig5_bar_chart():
    """
    Property: Fig. 5 (plot_scheme_comparison) uses plt.bar (bar chart format).

    Observation: plot_scheme_comparison contains `plt.bar(schemes, values, ...)`
    This format already matches the paper and must be preserved.
    """
    full_source = get_full_source()
    func_body = find_function_body(full_source, "plot_scheme_comparison")

    if func_body is None:
        record("Fig. 5 plot_scheme_comparison exists", False,
               "Function not found")
        return

    record("Fig. 5 plot_scheme_comparison exists", True)

    # Check it uses plt.bar
    has_bar = bool(re.search(r"plt\.bar\(", func_body))
    record("Fig. 5 uses plt.bar (bar chart)", has_bar,
           "plot_scheme_comparison must remain a bar chart")

    # Ensure it does NOT use plt.plot for the main chart
    # (plt.text is ok for annotations)
    has_plot_line = bool(re.search(r"plt\.plot\(", func_body))
    record("Fig. 5 does NOT use plt.plot", not has_plot_line,
           "Bar chart should not contain plt.plot calls")


def test_fig7_line_graph():
    """
    Property: Fig. 7 (plot_density_comparison) uses plt.plot (line graph)
    with task density on x-axis.

    Observation: plot_density_comparison contains:
      - plt.plot(densities, vals, marker=..., ...)
      - plt.xlabel("Task Density (tasks per 100 s)")
    This format already matches the paper and must be preserved.
    """
    full_source = get_full_source()
    func_body = find_function_body(full_source, "plot_density_comparison")

    if func_body is None:
        record("Fig. 7 plot_density_comparison exists", False,
               "Function not found")
        return

    record("Fig. 7 plot_density_comparison exists", True)

    # Check it uses plt.plot
    has_plot = bool(re.search(r"plt\.plot\(", func_body))
    record("Fig. 7 uses plt.plot (line graph)", has_plot,
           "plot_density_comparison must remain a line graph")

    # Check x-axis is density-related
    has_density_xlabel = bool(re.search(r"plt\.xlabel\(.*[Dd]ensity", func_body))
    record("Fig. 7 x-axis is task density", has_density_xlabel,
           "Expected xlabel containing 'Density'")

    # Check it does NOT use plt.bar
    has_bar = bool(re.search(r"plt\.bar\(", func_body))
    record("Fig. 7 does NOT use plt.bar", not has_bar,
           "Line graph should not contain plt.bar calls")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("PRESERVATION PROPERTY TESTS")
    print("=" * 60)
    print()

    print("Property: Latency parallel execution model")
    print("-" * 50)
    test_latency_parallel_execution_model()
    print()

    print("Property: SCENARIOS match Table II")
    print("-" * 50)
    test_scenarios_match_table_ii()
    print()

    print("Property: DQN alpha discretization step 0.2")
    print("-" * 50)
    test_dqn_alpha_levels_step_02()
    print()

    print("Property: QNetwork architecture")
    print("-" * 50)
    test_qnetwork_architecture()
    print()

    print("Property: Fig. 5 bar chart preserved")
    print("-" * 50)
    test_fig5_bar_chart()
    print()

    print("Property: Fig. 7 line graph preserved")
    print("-" * 50)
    test_fig7_line_graph()
    print()

    # Summary
    print("=" * 60)
    total = len(results)
    passed = sum(1 for _, p, _ in results if p)
    failed = sum(1 for _, p, _ in results if not p)
    print(f"SUMMARY: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        print("\nFAILED TESTS:")
        for name, p, detail in results:
            if not p:
                print(f"  - {name}: {detail}")
        sys.exit(1)
    else:
        print("\nAll preservation tests PASSED.")
        print("Baseline behavior confirmed — these properties must remain")
        print("unchanged after the bugfix is applied.")
        sys.exit(0)


if __name__ == "__main__":
    main()
