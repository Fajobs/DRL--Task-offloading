"""
Bug Condition Exploration Test
==============================
This test validates that the notebook parameters and plot configurations
align with the paper (Jiang et al. 2025).

EXPECTED BEHAVIOR on UNFIXED code: ALL assertions FAIL (confirming the bug exists).
EXPECTED BEHAVIOR on FIXED code: ALL assertions PASS (confirming the fix works).
"""

import os
import json
import re
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_PATH = os.path.join(ROOT_DIR, "DRL_Task_Offloading.ipynb")


def load_notebook_cells():
    """Load the notebook and return all code cell sources as a list of strings."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            cells.append(source)
    return cells


def get_all_source(cells):
    """Concatenate all code cells into a single string for searching."""
    return "\n".join(cells)


def find_assignment(source, var_name):
    """Find a simple variable assignment like `VAR = value` and return the value string."""
    pattern = rf"^\s*{re.escape(var_name)}\s*=\s*(.+?)(?:\s*#.*)?$"
    match = re.search(pattern, source, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def find_function_def(source, func_name):
    """Extract the full function body for a given function name."""
    # Find the function definition and capture until next def/class or end
    pattern = rf"(def\s+{re.escape(func_name)}\s*\(.*?\).*?:.*?)(?=\ndef\s|\nclass\s|\Z)"
    match = re.search(pattern, source, re.DOTALL)
    if match:
        return match.group(1)
    return None


class TestResult:
    def __init__(self):
        self.passed = []
        self.failed = []

    def assert_true(self, condition, description):
        if condition:
            self.passed.append(description)
        else:
            self.failed.append(description)

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n{'=' * 70}")
        print(f"BUG CONDITION EXPLORATION TEST RESULTS")
        print(f"{'=' * 70}")
        print(f"Total assertions: {total}")
        print(f"Passed: {len(self.passed)}")
        print(f"Failed: {len(self.failed)}")
        print(f"{'=' * 70}")

        if self.failed:
            print("\nFAILED ASSERTIONS (counterexamples proving bug exists):")
            for i, desc in enumerate(self.failed, 1):
                print(f"  {i}. {desc}")

        if self.passed:
            print("\nPASSED ASSERTIONS:")
            for i, desc in enumerate(self.passed, 1):
                print(f"  {i}. {desc}")

        print(f"\n{'=' * 70}")
        if self.failed:
            print("OVERALL: FAIL — Bug condition confirmed.")
        else:
            print("OVERALL: PASS — All parameters match paper.")
        print(f"{'=' * 70}")

        return len(self.failed) == 0


def main():
    cells = load_notebook_cells()
    source = get_all_source(cells)
    results = TestResult()

    # ─── NUM_EPOCHS == 100 (currently 200) ───
    num_epochs_val = find_assignment(source, "NUM_EPOCHS")
    results.assert_true(
        num_epochs_val == "100",
        f"NUM_EPOCHS == 100 (paper Fig.4 x-axis 0-100) | Found: NUM_EPOCHS = {num_epochs_val}"
    )

    # ─── STEPS_PER_EPOCH == 100 (currently 30) ───
    steps_val = find_assignment(source, "STEPS_PER_EPOCH")
    results.assert_true(
        steps_val == "100",
        f"STEPS_PER_EPOCH == 100 (paper: 'each epoch consists of 100 steps') | Found: STEPS_PER_EPOCH = {steps_val}"
    )

    # ─── EN_CPU == 20e9 (currently 10e9) ───
    en_cpu_val = find_assignment(source, "EN_CPU")
    results.assert_true(
        en_cpu_val is not None and eval(en_cpu_val) == 20e9,
        f"EN_CPU == 20e9 (paper Table I: 20 GHz) | Found: EN_CPU = {en_cpu_val}"
    )

    # ─── CLOUD_CPU == 100e9 (currently 20e9) ───
    cloud_cpu_val = find_assignment(source, "CLOUD_CPU")
    results.assert_true(
        cloud_cpu_val is not None and eval(cloud_cpu_val) == 100e9,
        f"CLOUD_CPU == 100e9 (paper Table I: 100 GHz) | Found: CLOUD_CPU = {cloud_cpu_val}"
    )

    # ─── num_devices == 50 in main pipeline (currently 30) ───
    num_devices_val = find_assignment(source, "num_devices")
    results.assert_true(
        num_devices_val == "50",
        f"num_devices == 50 (paper Table I: N=50) | Found: num_devices = {num_devices_val}"
    )

    # ─── Q_ALPHA_LEVELS constant with 11 levels at step 0.1 ───
    q_alpha_exists = "Q_ALPHA_LEVELS" in source
    q_alpha_val = find_assignment(source, "Q_ALPHA_LEVELS")
    q_alpha_correct = False
    if q_alpha_val is not None:
        try:
            levels = eval(q_alpha_val)
            q_alpha_correct = (
                isinstance(levels, list)
                and len(levels) == 11
                and abs(levels[1] - levels[0] - 0.1) < 1e-9
            )
        except Exception:
            pass
    results.assert_true(
        q_alpha_exists and q_alpha_correct,
        f"Q_ALPHA_LEVELS exists with 11 levels at step 0.1 (paper: alpha in {{0,0.1,...,1}}) | Found: {'exists=' + str(q_alpha_exists) + ', value=' + str(q_alpha_val)}"
    )

    # ─── evaluate_ga() uses gene_len=64 and bits_per_var=32 ───
    # Paper says "gene length 128, each variable occupies 32 sites".
    # With 2 variables per device (alpha, beta) at 32 bits each = 64 bits per device.
    ga_func = find_function_def(source, "evaluate_ga")
    ga_gene_len_correct = False
    ga_bits_per_var_correct = False
    if ga_func is not None:
        # Check default gene_len parameter (64 = 32 bits/var × 2 vars)
        gene_len_match = re.search(r"gene_len\s*(?::\s*int\s*)?=\s*(\d+)", ga_func)
        if gene_len_match:
            ga_gene_len_correct = gene_len_match.group(1) == "64"

        # Check bits_per_var is 32 (either explicit or gene_len // 2)
        bits_match = re.search(r"bits_per_var\s*=\s*(.+?)(?:\s*#.*)?(?:\n|$)", ga_func)
        if bits_match:
            bits_expr = bits_match.group(1).strip()
            # Accept either explicit "32" or "gene_len // 2" (which gives 32)
            ga_bits_per_var_correct = (
                bits_expr == "32" or "gene_len // 2" in bits_expr
            )

    results.assert_true(
        ga_gene_len_correct and ga_bits_per_var_correct,
        f"evaluate_ga() uses gene_len=64 and bits_per_var=32 (paper: 32 bits per var × 2 vars = 64 per device) | Found: gene_len={'64' if ga_gene_len_correct else 'NOT 64'}, bits_per_var={'32' if ga_bits_per_var_correct else 'NOT 32'}"
    )

    # ─── plot_loss_convergence() uses "Steps" x-axis label ───
    loss_func = find_function_def(source, "plot_loss_convergence")
    loss_xlabel_correct = False
    if loss_func is not None:
        # Check for "Steps" in xlabel call
        loss_xlabel_correct = bool(re.search(r'xlabel\s*\(\s*["\']Steps["\']', loss_func))
    results.assert_true(
        loss_xlabel_correct,
        f"plot_loss_convergence() x-axis label is 'Steps' (paper Fig.3) | Found: {'Steps' if loss_xlabel_correct else 'NOT Steps (likely Epoch)'}"
    )

    # ─── plot_scenario_comparison() uses plt.plot not plt.bar ───
    scenario_func = find_function_def(source, "plot_scenario_comparison")
    scenario_uses_plot = False
    scenario_uses_bar = False
    if scenario_func is not None:
        scenario_uses_plot = bool(re.search(r"plt\.plot\s*\(", scenario_func))
        scenario_uses_bar = bool(re.search(r"plt\.bar\s*\(", scenario_func))
    results.assert_true(
        scenario_uses_plot and not scenario_uses_bar,
        f"plot_scenario_comparison() uses plt.plot not plt.bar (paper Fig.6: line graph) | Found: uses_plot={scenario_uses_plot}, uses_bar={scenario_uses_bar}"
    )

    # ─── plot_scheme_comparison() y-axis is "Average Latency(ms)" ───
    scheme_func = find_function_def(source, "plot_scheme_comparison")
    scheme_ylabel_correct = False
    if scheme_func is not None:
        scheme_ylabel_correct = bool(
            re.search(r'ylabel\s*\(\s*["\']Average Latency\(ms\)["\']', scheme_func)
        )
    results.assert_true(
        scheme_ylabel_correct,
        f"plot_scheme_comparison() y-axis is 'Average Latency(ms)' (paper Fig.5) | Found: {'correct' if scheme_ylabel_correct else 'NOT Average Latency(ms) (likely Average Latency (s))'}"
    )

    # ─── plot_density_comparison() y-axis is "Average Latency(ms)" ───
    density_func = find_function_def(source, "plot_density_comparison")
    density_ylabel_correct = False
    if density_func is not None:
        density_ylabel_correct = bool(
            re.search(r'ylabel\s*\(\s*["\']Average Latency\(ms\)["\']', density_func)
        )
    results.assert_true(
        density_ylabel_correct,
        f"plot_density_comparison() y-axis is 'Average Latency(ms)' (paper Fig.7) | Found: {'correct' if density_ylabel_correct else 'NOT Average Latency(ms) (likely Average Latency (s))'}"
    )

    # ─── train_dqn() tracks per-step loss (not per-epoch average) ───
    train_func = find_function_def(source, "train_dqn")
    tracks_per_step = False
    if train_func is not None:
        # Per-step tracking means there's a step_losses list or loss is appended inside
        # the inner step loop (not just averaged per epoch)
        # The unfixed code does: loss_history.append(avg_loss) which is per-epoch average
        # Fixed code should have: step_losses.append(loss) or similar inside step loop
        has_step_losses = "step_losses" in train_func
        # Also check if loss_history is appended with avg_loss (per-epoch pattern)
        has_per_epoch_pattern = bool(re.search(r"loss_history\.append\s*\(\s*avg_loss\s*\)", train_func))
        tracks_per_step = has_step_losses and not has_per_epoch_pattern
    results.assert_true(
        tracks_per_step,
        f"train_dqn() tracks per-step loss (paper Fig.3: per-step data points) | Found: {'per-step tracking' if tracks_per_step else 'per-epoch average (avg_loss appended once per epoch)'}"
    )

    # ─── Print summary ───
    all_passed = results.summary()

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
