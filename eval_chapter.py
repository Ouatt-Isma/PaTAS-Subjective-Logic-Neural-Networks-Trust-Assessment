"""
eval_chapter.py — Chapter evaluation for PaTAS on the 5G energy dataset.

Produces figures and tables for §3 (Trust Assessment) and §4 (Trust Propagation)
of the dissertation chapter.

Experiment grid
---------------
  nn_base  : Baseline NN (no PTAS)
  ptas_tt  : PTAS, trusted features + trusted labels     [§3 fully-trusted scenario]
  ptas_vv  : PTAS, vacuous features + vacuous labels     [§3 uncertain scenario]
  ptas_tv  : PTAS, trusted features + vacuous labels     [§3 mixed scenario]
  ptas_dt  : PTAS, distrusted features + trusted labels  [§3 adversarial features]
  ptas_dd  : PTAS, distrusted features + distrusted labels [§3 fully distrusted]

Outputs (saved under --output, default: results_chapter/)
---------------------------------------------------------
  plots/accuracy_comparison.pdf   -- §3 final accuracy bar chart
  plots/learning_curves.pdf       -- §3 accuracy vs epoch
  plots/trust_convergence_tt.pdf  -- §4 trust mass evolution (ptas_tt)
  plots/trust_convergence_vv.pdf  -- §4 trust mass evolution (ptas_vv)
  plots/trust_convergence_dt.pdf  -- §4 trust mass evolution (ptas_dt)
  plots/omega_beliefs.pdf         -- §4 weight opinion heatmaps
  plots/calibration.pdf           -- §3.3 uncertainty vs accuracy (per config)
  tables/accuracy_summary.tex     -- §3 final accuracy table
  tables/trust_masses.tex         -- §4 final trust mass summary

Usage
-----
  python eval_chapter.py data/
  python eval_chapter.py data/ --plots-only
  python eval_chapter.py data/ --force
  python eval_chapter.py data/ --output results_chapter/
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from data_loader import load_5g_dataset, make_synthetic_5g
from external_bridge import run_with_external_implementation

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

@dataclass
class ChapterExp:
    label: str       # short identifier used as directory name
    name: str        # display name for plots
    use_ptas: bool
    x_trust: str     # "trusted" | "vacuous" | "distrusted"
    y_trust: str
    port: int        # each experiment needs its own port

EXPERIMENTS: list[ChapterExp] = [
    ChapterExp("nn_base",  "NN Baseline",    False, "trusted",    "trusted",    6560),
    ChapterExp("ptas_tt",  "PTAS (T,T)",     True,  "trusted",    "trusted",    6562),
    ChapterExp("ptas_vv",  "PTAS (V,V)",     True,  "vacuous",    "vacuous",    6564),
    ChapterExp("ptas_tv",  "PTAS (T,V)",     True,  "trusted",    "vacuous",    6566),
    ChapterExp("ptas_dt",  "PTAS (D,T)",     True,  "distrusted", "trusted",    6568),
    ChapterExp("ptas_dd",  "PTAS (D,D)",     True,  "distrusted", "distrusted", 6570),
]

# Configs for which we save & plot trust convergence data
EVAL_EXPS = {"ptas_tt", "ptas_vv", "ptas_dt"}

# Color palette (consistent across all figures)
COLORS = {
    "nn_base":  "#555555",
    "ptas_tt":  "#1f77b4",
    "ptas_vv":  "#ff7f0e",
    "ptas_tv":  "#2ca02c",
    "ptas_dt":  "#d62728",
    "ptas_dd":  "#9467bd",
}

MARKERS = {
    "nn_base": "s",
    "ptas_tt": "o",
    "ptas_vv": "^",
    "ptas_tv": "D",
    "ptas_dt": "v",
    "ptas_dd": "P",
}

# Training hyper-parameters (shared by all experiments)
EPOCHS     = 10
BATCH_SIZE = 64
N_HIDDEN   = 32
LR         = 0.05
EPS_LOW    = 0.05

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exp_path(output_dir: str, label: str) -> str:
    return os.path.join(output_dir, label)


def _load_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _load_pkl(path: str):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _results_path(output_dir: str, label: str) -> str:
    return os.path.join(_exp_path(output_dir, label), "results.json")


def _eval_path(output_dir: str, label: str) -> str:
    return os.path.join(_exp_path(output_dir, label), "eval_ptas.pkl")


def _omega_path(output_dir: str, label: str) -> str:
    return os.path.join(_exp_path(output_dir, label), "omega_thetas.pkl")


# ---------------------------------------------------------------------------
# PTAS in-process inference (for calibration analysis §3.3)
# ---------------------------------------------------------------------------

def ptas_forward(omega_tensors: list, n_samples: int, n_features: int,
                 fill_method: str = "trusted") -> np.ndarray:
    """
    Run a PTAS feedforward pass in-process using saved omega_thetas.

    omega_tensors : list of 2 numpy arrays [(in+1, hidden, 3), (hidden+1, out, 3)]
    n_samples     : batch size for the probe
    n_features    : number of input features
    fill_method   : trust opinion for input ("trusted" | "vacuous" | "distrusted")

    Returns numpy array (n_samples, n_classes, 3).
    """
    from patas_module.concrete.TensorTO import TensorArrayTO, fill as tfill

    W0 = TensorArrayTO(omega_tensors[0].astype(np.float32))  # (in+1, hidden, 3)
    W1 = TensorArrayTO(omega_tensors[1].astype(np.float32))  # (hidden+1, out, 3)

    tx_np = tfill((n_samples, n_features), method=fill_method, dtype=np.float32)
    bias  = tfill((n_samples, 1),          method="one",       dtype=np.float32)

    Tx_bias = TensorArrayTO(np.concatenate([tx_np, bias], axis=1))
    Ty1     = TensorArrayTO.dot(Tx_bias, W0)  # (n, hidden, 3)

    bias2   = tfill((n_samples, 1), method="one", dtype=np.float32)
    Ty1_b   = TensorArrayTO(np.concatenate([Ty1.value, bias2], axis=1))
    Ty2     = TensorArrayTO.dot(Ty1_b, W1)    # (n, out, 3)

    return Ty2.value  # (n_samples, n_classes, 3)


# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------

def run_chapter_experiments(
    data_dir: Optional[str],
    output_dir: str,
    force: bool = False,
) -> dict[str, dict]:
    """
    Run all chapter experiments and return a mapping label → results dict.
    Results are cached; set force=True to re-run.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset once
    if data_dir is None:
        print("No data directory — generating synthetic 5G data...")
        data_dir = make_synthetic_5g(n_bs=20, n_hours=72, cells_per_bs=2, seed=0)

    ds = load_5g_dataset(data_dir, n_classes=3, test_frac=0.2, seed=0)
    n_classes = int(ds.y_train.max()) + 1
    y_train_oh = np.eye(n_classes, dtype=np.float32)[ds.y_train]
    y_test_oh  = np.eye(n_classes, dtype=np.float32)[ds.y_test]

    all_results: dict[str, dict] = {}

    for exp in EXPERIMENTS:
        cache = _results_path(output_dir, exp.label)
        if not force and os.path.exists(cache):
            all_results[exp.label] = _load_json(cache)
            print(f"[cached] {exp.label}")
            continue

        print(f"\n{'='*60}")
        print(f"Running: {exp.name}  (port {exp.port})")
        print(f"{'='*60}")

        run_label = os.path.join(output_dir, exp.label)
        r = run_with_external_implementation(
            data_dir=None,
            dataset="5g",
            n_hidden=N_HIDDEN,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            lr=LR,
            eps_low=EPS_LOW,
            x_trust=exp.x_trust,
            y_trust=exp.y_trust,
            use_ptas=exp.use_ptas,
            port=exp.port,
            run_label=run_label,
            X_train=ds.X_train.copy(),
            y_train_oh=y_train_oh,
            X_test=ds.X_test.copy(),
            y_test_oh=y_test_oh,
            enable_eval=(exp.label in EVAL_EXPS),
        )
        all_results[exp.label] = r
        print(f"  → test acc: {r.get('final_test_acc', float('nan')):.4f}")

    return all_results


# ---------------------------------------------------------------------------
# §3 — Trust Assessment plots
# ---------------------------------------------------------------------------

def plot_accuracy_comparison(all_results: dict, output_dir: str):
    """Bar chart of final test accuracy for all configurations."""
    labels = [e.label for e in EXPERIMENTS]
    names  = [e.name  for e in EXPERIMENTS]
    accs   = [all_results[l].get("final_test_acc", 0.0) for l in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
        names, accs,
        color=[COLORS[l] for l in labels],
        edgecolor="white", linewidth=0.5,
    )
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylim(0, min(1.05, max(accs) + 0.12))
    ax.set_ylabel("Final Test Accuracy")
    ax.set_title("§3 — Trust Assessment: Final Test Accuracy by Configuration")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()

    out = os.path.join(output_dir, "plots", "accuracy_comparison.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_learning_curves(all_results: dict, output_dir: str):
    """Test accuracy vs epoch for all configurations."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for exp in EXPERIMENTS:
        r = all_results.get(exp.label, {})
        curve = r.get("epoch_test_acc", [])
        if not curve:
            continue
        ax.plot(
            range(1, len(curve) + 1), curve,
            label=exp.name,
            color=COLORS[exp.label],
            marker=MARKERS[exp.label],
            markersize=5,
            linewidth=1.5,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("§3 — Trust Assessment: Learning Curves by Configuration")
    ax.legend(loc="lower right", framealpha=0.8)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    out = os.path.join(output_dir, "plots", "learning_curves.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_calibration(all_results: dict, output_dir: str):
    """
    §3.3 — Calibration plot.

    For each PTAS configuration, use the saved omega_thetas to compute the
    output uncertainty mass under three probe trust inputs (trusted/vacuous/distrusted).
    Then scatter (avg uncertainty, final test accuracy) as a proxy calibration.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    probe_methods = ["trusted", "vacuous", "distrusted"]
    probe_markers = {"trusted": "o", "vacuous": "^", "distrusted": "s"}
    probe_labels  = {"trusted": "Trusted probe", "vacuous": "Vacuous probe",
                     "distrusted": "Distrusted probe"}

    # Collect points: (uncertainty, test_accuracy) per (exp, probe)
    points: dict[str, list[tuple[float, float]]] = {m: [] for m in probe_methods}

    for exp in EXPERIMENTS:
        if not exp.use_ptas:
            continue
        omega_path = _omega_path(output_dir, exp.label)
        if not os.path.exists(omega_path):
            continue
        omega = _load_pkl(omega_path)  # list of 2 numpy arrays
        n_features = omega[0].shape[0] - 1  # in+1 → in

        for method in probe_methods:
            try:
                out = ptas_forward(omega, n_samples=1, n_features=n_features,
                                   fill_method=method)  # (1, n_classes, 3)
                # Average uncertainty across output classes
                avg_u = float(out[0, :, 2].mean())
                acc   = all_results[exp.label].get("final_test_acc", float("nan"))
                points[method].append((avg_u, acc))
            except Exception as e:
                print(f"  [calibration] skipped {exp.label}/{method}: {e}")

    for method, pts in points.items():
        if not pts:
            continue
        us, accs = zip(*pts)
        ax.scatter(us, accs, marker=probe_markers[method], s=60,
                   label=probe_labels[method], zorder=3)
        for (u, a), exp in zip(pts, [e for e in EXPERIMENTS if e.use_ptas]):
            ax.annotate(exp.name, (u, a), textcoords="offset points",
                        xytext=(4, 4), fontsize=8)

    # Reference: uncertainty=0 → expected accuracy=1 (ideal anti-correlation)
    ax.axline((0, 1.0), slope=-1.0, linestyle="--", color="gray",
               linewidth=0.8, label="Ideal (u ↑ → acc ↓)")
    ax.set_xlabel("Avg. Output Uncertainty Mass (u)")
    ax.set_ylabel("Final Test Accuracy")
    ax.set_title("§3.3 — Calibration: Uncertainty vs. Test Accuracy")
    ax.legend(framealpha=0.8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    out = os.path.join(output_dir, "plots", "calibration.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# §4 — Trust Propagation plots
# ---------------------------------------------------------------------------

def plot_trust_convergence(exp_label: str, exp_name: str,
                           output_dir: str, n_train: int):
    """
    §4 convergence plot for one PTAS configuration.

    Plots belief / uncertainty / disbelief mass vs. training batch for
    three probe inputs (trusted, vacuous, distrusted).
    """
    eval_file = _eval_path(output_dir, exp_label)
    if not os.path.exists(eval_file):
        print(f"  [skip] no eval_ptas.pkl for {exp_label}")
        return

    data = _load_pkl(eval_file)
    EVAL = data["EVAL"]

    n_steps = len(EVAL.get("trust", []))
    if n_steps == 0:
        print(f"  [skip] empty EVAL for {exp_label}")
        return

    batches_per_epoch = math.ceil(n_train / BATCH_SIZE)
    steps = np.arange(1, n_steps + 1)
    epoch_boundaries = [(i + 1) * batches_per_epoch for i in range(EPOCHS - 1)]

    # EVAL[key] is list of arrays shape (1, n_classes, 3)
    # index [..., 0]=belief, [..., 1]=disbelief, [..., 2]=uncertainty
    probe_keys = {
        "trust":    ("Trusted input",    "#1f77b4"),
        "untrust":  ("Vacuous input",    "#ff7f0e"),
        "distrust": ("Distrusted input", "#d62728"),
    }

    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    comp_names = ["Belief (b)", "Disbelief (d)", "Uncertainty (u)"]
    comp_idx   = [0, 1, 2]

    for row_i, (key, (probe_label, probe_color)) in enumerate(probe_keys.items()):
        series = EVAL.get(key, [])
        if not series:
            continue
        # Average across output classes: arr shape (1, n_classes, 3) → mean over axis 1
        b_arr = np.array([s[0, :, 0].mean() for s in series])
        d_arr = np.array([s[0, :, 1].mean() for s in series])
        u_arr = np.array([s[0, :, 2].mean() for s in series])
        comp_arrays = [b_arr, d_arr, u_arr]

        for col_i, (comp_label, comp_arr) in enumerate(zip(comp_names, comp_arrays)):
            ax = axes[row_i, col_i]
            ax.plot(steps, comp_arr, color=probe_color, linewidth=0.9, alpha=0.85)
            for xb in epoch_boundaries:
                ax.axvline(xb, color="gray", linestyle="--", linewidth=0.5, alpha=0.6)
            ax.set_ylim(-0.05, 1.05)
            if col_i == 0:
                ax.set_ylabel(probe_label, fontsize=10)
            if row_i == 0:
                ax.set_title(comp_label, fontsize=11)
            if row_i == 2:
                ax.set_xlabel("Training batch")

    fig.suptitle(f"§4 — Trust Propagation Convergence: {exp_name}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = os.path.join(output_dir, "plots", f"trust_convergence_{exp_label}.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_convergence_overlay(output_dir: str, n_train: int):
    """
    §4 overlay: compare uncertainty mass evolution across configurations on one axis.
    Shows how different trust assignments lead to different convergence dynamics.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    probe_keys = {
        "trust":    "Trusted probe",
        "untrust":  "Vacuous probe",
        "distrust": "Distrusted probe",
    }
    batches_per_epoch = math.ceil(n_train / BATCH_SIZE)
    epoch_boundaries  = [(i + 1) * batches_per_epoch for i in range(EPOCHS - 1)]

    for ax, (probe_key, probe_title) in zip(axes, probe_keys.items()):
        for exp_label in sorted(EVAL_EXPS):
            eval_file = _eval_path(output_dir, exp_label)
            if not os.path.exists(eval_file):
                continue
            data  = _load_pkl(eval_file)["EVAL"]
            series = data.get(probe_key, [])
            if not series:
                continue
            n_steps = len(series)
            u_arr   = np.array([s[0, :, 2].mean() for s in series])
            exp_obj = next(e for e in EXPERIMENTS if e.label == exp_label)
            ax.plot(
                np.arange(1, n_steps + 1), u_arr,
                label=exp_obj.name,
                color=COLORS[exp_label],
                linewidth=1.2,
            )
        for xb in epoch_boundaries:
            ax.axvline(xb, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_title(probe_title)
        ax.set_xlabel("Training batch")
        ax.set_ylim(-0.05, 1.05)
        if ax is axes[0]:
            ax.set_ylabel("Avg. output uncertainty mass (u)")

    axes[0].legend(loc="upper right", framealpha=0.85)
    fig.suptitle("§4 — Uncertainty Evolution During Training (comparison)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out = os.path.join(output_dir, "plots", "convergence_overlay.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_omega_beliefs(output_dir: str):
    """
    §4 — Visualise the learned weight opinions (omega_thetas) for PTAS configs.

    For each configuration, plot a heatmap of the belief mass in layer-0 weights.
    """
    ptas_exps = [e for e in EXPERIMENTS if e.use_ptas]
    n_exps = len(ptas_exps)

    fig, axes = plt.subplots(2, n_exps, figsize=(14, 6))

    for col, exp in enumerate(ptas_exps):
        omega_path = _omega_path(output_dir, exp.label)
        if not os.path.exists(omega_path):
            for row in range(2):
                axes[row, col].set_visible(False)
            continue

        omega = _load_pkl(omega_path)   # list: [array(in+1,hidden,3), array(hidden+1,out,3)]

        for row, (layer_w, layer_name) in enumerate(zip(omega, ["Layer 0 (in→hid)", "Layer 1 (hid→out)"])):
            belief = layer_w[..., 0]    # shape (rows, cols)
            im = axes[row, col].imshow(
                belief, vmin=0, vmax=1, cmap="Blues", aspect="auto"
            )
            if row == 0:
                axes[row, col].set_title(exp.name, fontsize=10)
            if col == 0:
                axes[row, col].set_ylabel(layer_name, fontsize=9)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="Belief mass (b)")
    fig.suptitle("§4 — Learned Weight Opinions: Belief Mass Heatmaps", fontsize=13)

    out = os.path.join(output_dir, "plots", "omega_beliefs.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_omega_uncertainty(output_dir: str):
    """
    §4 — Uncertainty mass (u) in learned weights: distribution per configuration.
    Shows how certain/uncertain the PTAS is about its learned weight opinions.
    """
    ptas_exps = [e for e in EXPERIMENTS if e.use_ptas]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for layer_idx, (ax, layer_name) in enumerate(zip(axes, ["Layer 0 (in→hid)", "Layer 1 (hid→out)"])):
        for exp in ptas_exps:
            omega_path = _omega_path(output_dir, exp.label)
            if not os.path.exists(omega_path):
                continue
            omega = _load_pkl(omega_path)
            u_vals = omega[layer_idx][..., 2].ravel()
            ax.hist(u_vals, bins=30, alpha=0.5, color=COLORS[exp.label],
                    label=exp.name, density=True)
        ax.set_xlabel("Uncertainty mass (u)")
        ax.set_ylabel("Density")
        ax.set_title(f"Weight Opinion Uncertainty — {layer_name}")
        ax.legend(fontsize=8, framealpha=0.8)

    fig.suptitle("§4 — Distribution of Uncertainty in Learned Weight Opinions", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out = os.path.join(output_dir, "plots", "omega_uncertainty.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def _booktabs_table(rows: list[list], headers: list[str], caption: str,
                    label: str, path: str, fmt: list[str] | None = None):
    """Write a booktabs LaTeX table to *path*."""
    n_cols = len(headers)
    col_spec = "l" + "c" * (n_cols - 1)
    if fmt is None:
        fmt = ["{}"] * n_cols

    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        "    " + " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\",
        "    \\midrule",
    ]
    for row in rows:
        cells = [fmt[i].format(v) for i, v in enumerate(row)]
        lines.append("    " + " & ".join(cells) + " \\\\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Saved {path}")


def table_accuracy(all_results: dict, output_dir: str):
    """§3 accuracy summary table."""
    headers = ["Configuration", "x-Trust", "y-Trust", "Train Acc.", "Test Acc."]
    fmt     = ["{}", "{}", "{}", "{:.3f}", "{:.3f}"]
    rows = []
    for exp in EXPERIMENTS:
        r = all_results.get(exp.label, {})
        rows.append([
            exp.name,
            exp.x_trust.capitalize(),
            exp.y_trust.capitalize(),
            r.get("final_train_acc", float("nan")),
            r.get("final_test_acc",  float("nan")),
        ])
    _booktabs_table(
        rows, headers,
        caption="Final train and test accuracy for each trust configuration on the 5G dataset.",
        label="tab:chapter_accuracy",
        path=os.path.join(output_dir, "tables", "accuracy_summary.tex"),
        fmt=fmt,
    )


def table_trust_masses(output_dir: str):
    """
    §4 trust mass summary table.

    For each PTAS config, show the final output trust masses (b, d, u)
    under three probe inputs (trusted, vacuous, distrusted).
    """
    headers = ["Configuration", "Probe input", "Belief (b)", "Disbelief (d)", "Uncertainty (u)"]
    fmt     = ["{}", "{}", "{:.3f}", "{:.3f}", "{:.3f}"]
    rows    = []

    probe_keys   = {"trust": "Trusted", "untrust": "Vacuous", "distrust": "Distrusted"}
    n_features   = None

    for exp in EXPERIMENTS:
        if not exp.use_ptas:
            continue
        omega_path = _omega_path(output_dir, exp.label)
        if not os.path.exists(omega_path):
            continue
        omega = _load_pkl(omega_path)
        if n_features is None:
            n_features = omega[0].shape[0] - 1

        for fill_method, probe_label in probe_keys.items():
            try:
                out = ptas_forward(omega, n_samples=1,
                                   n_features=n_features,
                                   fill_method=fill_method)  # (1, n_classes, 3)
                b = float(out[0, :, 0].mean())
                d = float(out[0, :, 1].mean())
                u = float(out[0, :, 2].mean())
                rows.append([exp.name, probe_label, b, d, u])
            except Exception as e:
                print(f"  [table] skipped {exp.label}/{fill_method}: {e}")

    _booktabs_table(
        rows, headers,
        caption=(
            "Average output trust masses (b, d, u) of the trained PTAS under three probe "
            "input trust opinions. Values averaged over output classes."
        ),
        label="tab:chapter_trust_masses",
        path=os.path.join(output_dir, "tables", "trust_masses.tex"),
        fmt=fmt,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all(
    data_dir: Optional[str],
    output_dir: str,
    force: bool = False,
    plots_only: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"),  exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)

    # --- Experiments ---
    if plots_only:
        # Load cached results
        all_results = {}
        for exp in EXPERIMENTS:
            p = _results_path(output_dir, exp.label)
            all_results[exp.label] = _load_json(p) if os.path.exists(p) else {}
    else:
        all_results = run_chapter_experiments(data_dir, output_dir, force)

    # Infer n_train from one of the datasets
    n_train = None
    for exp in EXPERIMENTS:
        p = _results_path(output_dir, exp.label)
        if os.path.exists(p):
            r = _load_json(p)
            curve = r.get("epoch_train_acc", [])
            # epoch_train_acc has EPOCHS entries; total batches ≈ n_entries in EVAL
            # we derive n_train from the EVAL length if available
            break

    # Try to get n_train from EVAL data
    for exp_label in EVAL_EXPS:
        ep = _eval_path(output_dir, exp_label)
        if os.path.exists(ep):
            data = _load_pkl(ep)
            n_steps = len(data["EVAL"].get("trust", []))
            if n_steps > 0:
                n_train = max(1, round(n_steps / EPOCHS)) * BATCH_SIZE
                break
    if n_train is None:
        n_train = 2880  # fallback: typical 5G dataset size

    print(f"\nEstimated n_train ≈ {n_train}")

    # --- §3 plots ---
    print("\n── §3 Trust Assessment plots ──")
    plot_accuracy_comparison(all_results, output_dir)
    plot_learning_curves(all_results, output_dir)
    plot_calibration(all_results, output_dir)

    # --- §4 plots ---
    print("\n── §4 Trust Propagation plots ──")
    for exp_label in sorted(EVAL_EXPS):
        exp_name = next(e.name for e in EXPERIMENTS if e.label == exp_label)
        plot_trust_convergence(exp_label, exp_name, output_dir, n_train)

    plot_convergence_overlay(output_dir, n_train)
    plot_omega_beliefs(output_dir)
    plot_omega_uncertainty(output_dir)

    # --- Tables ---
    print("\n── LaTeX tables ──")
    table_accuracy(all_results, output_dir)
    table_trust_masses(output_dir)

    print("\n✓ Chapter evaluation complete.")
    print(f"  Output directory: {os.path.abspath(output_dir)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Chapter evaluation: PaTAS on 5G energy dataset"
    )
    ap.add_argument(
        "data_dir", nargs="?", default=None,
        help="Path to 5G CSV directory (omit for synthetic data)",
    )
    ap.add_argument(
        "--output", default="results_chapter",
        help="Output directory for results, plots, and tables (default: results_chapter/)",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Re-run experiments even if cached results exist",
    )
    ap.add_argument(
        "--plots-only", action="store_true",
        help="Skip training; regenerate plots from cached results",
    )
    return ap.parse_args()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    args = _parse_args()
    run_all(
        data_dir=args.data_dir,
        output_dir=args.output,
        force=args.force,
        plots_only=args.plots_only,
    )
