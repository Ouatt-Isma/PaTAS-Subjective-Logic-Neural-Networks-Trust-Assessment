"""eval_5g_noise.py
Noise-robustness evaluation of PaTAS on the 5G energy dataset.

Experiment grid
---------------
(a) Feature noise   : Gaussian measurement noise at 4 relative sigma levels
(b) Label noise     : random mislabelling at 4 flip rates
(c) Combined noise  : three paired (sigma, flip) conditions

For every noise condition, three configurations are compared
  nn        -- standard neural network, no trust reasoning
  ptas-fix  -- PaTAS with fully-trusted opinions regardless of noise level
  ptas-cal  -- PaTAS with trust opinions calibrated to the noise level

Outputs (saved under --output, default: results_noise/)
--------------------------------------------------------
  plots/trust_mapping.pdf
  plots/feature_noise_accuracy.pdf
  plots/label_noise_accuracy.pdf
  plots/combined_accuracy.pdf
  plots/learning_curves.pdf
  tables/trust_mapping.tex
  tables/feature_noise_results.tex
  tables/label_noise_results.tex
  tables/combined_results.tex
  data/<run_label>/results.json    (one per experiment, cached)

Usage
-----
  python eval_5g_noise.py data/                      # run all + plot
  python eval_5g_noise.py data/ --force              # re-run even if cached
  python eval_5g_noise.py data/ --plots-only         # skip training, plot from cache
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional

import copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from data_loader import load_5g_dataset, make_synthetic_5g
from noise_utils import (
    add_feature_noise,
    add_label_noise,
    add_feature_noise_per_sample,
    feature_noise_to_trust,
    label_noise_to_trust,
    trust_components,
    trust_str,
)
from external_bridge import run_with_external_implementation
from calibration_trust_eval import calibration_trust_analysis
from latency_eval import latency_analysis

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         14,
    "axes.titlesize":    14,
    "axes.labelsize":    14,
    "legend.fontsize":   14,
    "xtick.labelsize":   14,
    "ytick.labelsize":   14,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

# Consistent colours across all plots
_COLORS  = {"nn": "#555555", 
            "ptas-obs": "#e07b39", 
            "ptas-cal-fb": "#8e44ad"}
_LABELS  = {"nn": "NN", 
            "ptas-obs": "PaTAS (observer)", 
            "ptas-cal-fb": "PaTAS"}
_MARKERS = {"nn": "s", "ptas-obs": "D", "ptas-cal-fb": "P"}

# Colors / labels / markers for coverage-accuracy and threshold plots
_THR_COLORS  = {
    "nn":         "#555555",
    "comb_pi":    "#2980b9",
    # "comb_bdu":   "#e74c3c",
    # "comb_b": "#8e44ad",
    # "comb_bu":   "#1abc9c",
    # "comb_bd":   "#e07b39",
}
_THR_LABELS  = {
    "nn":         "NN softmax",
    "comb_b": r"$b$",
    "comb_bu":   r"$(b{-}u)$",
    "comb_bdu":   r"$(b{-}d{-}\frac{u}{2})$",
    "comb_pi":    r"$\pi$  $(b+\frac{u}{2})$",
    "comb_bd":   r"$(b{-}d)$",
}
_THR_MARKERS    = {"nn": "D", "comb_bd": "P", "comb_b": "o",
                   "comb_bu": "s", "comb_bdu": "^", "comb_pi": "X"}
_THR_LINESTYLES = {"nn": "-", "comb_bd": "-", "comb_b": "--",
                   "comb_bu": ":", "comb_bdu": "-.", "comb_pi": "-"}
_THR_SOURCES = list(_THR_COLORS)


# ---------------------------------------------------------------------------
# Experiment grid definition
# ---------------------------------------------------------------------------

from hyperparams import FEATURE_SIGMAS, LABEL_FLIPS, COMBINED, N_RUNS
from hyperparams import EPOCHS, BATCH, LR, EPS_LOW, N_HIDDEN, BASE_PORT
from hyperparams import FEATURE_SIGMAS_plots, LABEL_FLIPS_plots

CONFIGS           = ["nn", "ptas-cal-fb"]
ACC_PLOTS_CONFIGS = ["nn"]


@dataclass
class ExpSpec:
    """Specification for a single training run."""
    label: str        # unique identifier  e.g. "fn_0.10_ptas-cal"
    config: str       # one of CONFIGS
    sigma: float      # feature noise level
    flip: float       # label flip rate
    use_ptas: bool
    x_trust: object   # str | TrustOpinion
    y_trust: object   # str | TrustOpinion
    port: int = BASE_PORT
    run_seed: int = 0


def _build_grid(base_port: int = BASE_PORT) -> list[ExpSpec]:
    """Return the full list of ExpSpec objects: nn baseline vs ptas-cal-fb."""
    specs: list[ExpSpec] = []
    port = base_port

    def _next_port():
        nonlocal port
        p = port
        port += 1
        return p

    # Each (noise_condition, config) pair runs N_RUNS times with different
    # noise-injection seeds so results can be averaged into mean ± std.

    # ------------------------------------------------------------------
    # (a) Feature noise study  (labels always clean)
    # ------------------------------------------------------------------
    for sigma in FEATURE_SIGMAS:
        xt_cal = feature_noise_to_trust(sigma)
        for run in range(N_RUNS):
            specs.append(ExpSpec(
                label=f"fn_{sigma:.2f}_nn_r{run}",
                config="nn", sigma=sigma, flip=0.0,
                use_ptas=False, x_trust="trusted", y_trust="trusted",
                port=BASE_PORT, run_seed=run,
            ))
            specs.append(ExpSpec(
                label=f"fn_{sigma:.2f}_ptas-cal-fb_r{run}",
                config="ptas-cal-fb", sigma=sigma, flip=0.0,
                use_ptas=True, x_trust=xt_cal, y_trust="trusted",
                port=_next_port(), run_seed=run,
            ))

    # ------------------------------------------------------------------
    # (b) Label noise study  (features always clean)
    # ------------------------------------------------------------------
    for flip in LABEL_FLIPS:
        yt_cal = label_noise_to_trust(flip)
        for run in range(N_RUNS):
            specs.append(ExpSpec(
                label=f"ln_{flip:.2f}_nn_r{run}",
                config="nn", sigma=0.0, flip=flip,
                use_ptas=False, x_trust="trusted", y_trust="trusted",
                port=BASE_PORT, run_seed=run,
            ))
            specs.append(ExpSpec(
                label=f"ln_{flip:.2f}_ptas-cal-fb_r{run}",
                config="ptas-cal-fb", sigma=0.0, flip=flip,
                use_ptas=True, x_trust="trusted", y_trust=yt_cal,
                port=_next_port(), run_seed=run,
            ))

    # ------------------------------------------------------------------
    # (d) Extended feature noise — nn only, for feature_noise_accuracy plot
    # ------------------------------------------------------------------
    for sigma in FEATURE_SIGMAS_plots:
        if sigma in FEATURE_SIGMAS:
            continue
        for run in range(N_RUNS):
            specs.append(ExpSpec(
                label=f"fn_{sigma:.2f}_nn_r{run}",
                config="nn", sigma=sigma, flip=0.0,
                use_ptas=False, x_trust="trusted", y_trust="trusted",
                port=BASE_PORT, run_seed=run,
            ))

    # ------------------------------------------------------------------
    # (e) Extended label noise — nn only, for label_noise_accuracy plot
    # ------------------------------------------------------------------
    for flip in LABEL_FLIPS_plots:
        if flip in LABEL_FLIPS:
            continue
        for run in range(N_RUNS):
            specs.append(ExpSpec(
                label=f"ln_{flip:.2f}_nn_r{run}",
                config="nn", sigma=0.0, flip=flip,
                use_ptas=False, x_trust="trusted", y_trust="trusted",
                port=BASE_PORT, run_seed=run,
            ))

    # ------------------------------------------------------------------
    # (c) Combined noise study
    # ------------------------------------------------------------------
    for sigma, flip in COMBINED:
        xt_cal = feature_noise_to_trust(sigma)
        yt_cal = label_noise_to_trust(flip)
        for run in range(N_RUNS):
            specs.append(ExpSpec(
                label=f"comb_{sigma:.2f}_{flip:.2f}_nn_r{run}",
                config="nn", sigma=sigma, flip=flip,
                use_ptas=False, x_trust="trusted", y_trust="trusted",
                port=BASE_PORT, run_seed=run,
            ))
            specs.append(ExpSpec(
                label=f"comb_{sigma:.2f}_{flip:.2f}_ptas-cal-fb_r{run}",
                config="ptas-cal-fb", sigma=sigma, flip=flip,
                use_ptas=True, x_trust=xt_cal, y_trust=yt_cal,
                port=_next_port(), run_seed=run,
            ))

    return specs


# ---------------------------------------------------------------------------
# Running experiments
# ---------------------------------------------------------------------------

def run_experiment(
    spec: ExpSpec,
    ds,               # loaded Dataset object (from data_loader)
    n_classes: int,
    output_dir: str,
    force: bool = False,
) -> dict:
    """Run one experiment (or load from cache) and return the results dict."""
    results_file = os.path.join(output_dir, "data", spec.label, "results.json")

    if not force and os.path.exists(results_file):
        print(f"  [cache]  {spec.label}")
        with open(results_file) as fh:
            return json.load(fh)

    print(f"  [run]    {spec.label}")

    # Seeded RNG — varied across N_RUNS so results can be averaged.
    rng = np.random.default_rng(spec.run_seed)

    # Build noisy training data for this experiment.
    X_train = add_feature_noise(ds.X_train, spec.sigma, rng=rng)

    y_train_int = add_label_noise(ds.y_train, spec.flip, n_classes, rng=rng)
    y_train_oh = np.eye(n_classes, dtype=np.float32)[y_train_int]
    X_test    = ds.X_test
    y_test_oh = np.eye(n_classes, dtype=np.float32)[ds.y_test]

    run_label = os.path.join("data", spec.label)   # subdir under output_dir
    orig_cwd = os.getcwd()
    os.chdir(output_dir)          # results.json will land inside output_dir
    try:
        results = run_with_external_implementation(
            data_dir=None,
            dataset="5g",
            n_hidden=N_HIDDEN,
            epochs=EPOCHS,
            batch=BATCH,
            lr=LR,
            eps_low=EPS_LOW,
            x_trust=spec.x_trust,
            y_trust=spec.y_trust,
            use_ptas=spec.use_ptas,
            port=spec.port,
            run_label=run_label,
            X_train=X_train,
            y_train_oh=y_train_oh,
            X_test=X_test,
            y_test_oh=y_test_oh,
        )
    finally:
        os.chdir(orig_cwd)

    results["label"] = spec.label
    results["config"] = spec.config
    results["sigma"] = spec.sigma
    results["flip"] = spec.flip
    return results


def run_all(data_dir: Optional[str], output_dir: str, force: bool = False) -> dict:
    """Run all experiments and return a nested results dict keyed by label."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)

    if data_dir is None:
        print("No data dir — generating synthetic 5G dataset...")
        data_dir = make_synthetic_5g(n_bs=20, n_hours=72, cells_per_bs=2, seed=0)

    ds = load_5g_dataset(data_dir, n_classes=3, test_frac=0.2, seed=0)
    n_classes = int(ds.y_train.max()) + 1

    specs = _build_grid()
    all_results: dict[str, dict] = {}

    print(f"\nRunning {len(specs)} experiments ...\n")
    for spec in specs:
        r = run_experiment(spec, ds, n_classes, output_dir, force=force)
        all_results[spec.label] = r

    return all_results, ds, n_classes


# ---------------------------------------------------------------------------
# Utilities for extracting structured data from results
# ---------------------------------------------------------------------------

def _epoch_curve(all_results: dict, label: str) -> list[float]:
    r = all_results.get(label, {})
    return [v * 100 for v in r.get("epoch_test_acc", [])]


# ---------------------------------------------------------------------------
# Multi-run aggregation helpers (for ptas-percal)
# ---------------------------------------------------------------------------

def _acc_multi(all_results: dict, base_label: str,
               n_runs: int = N_RUNS) -> tuple[float, float]:
    """Return (mean%, std%) over N_RUNS independent runs of base_label."""
    accs = []
    for run in range(n_runs):
        r = all_results.get(f"{base_label}_r{run}", {})
        a = float(r.get("final_test_acc", float("nan")))
        if not np.isnan(a):
            accs.append(a * 100)
    if not accs:
        return float("nan"), 0.0
    return float(np.mean(accs)), float(np.std(accs))


def _epoch_curve_multi(all_results: dict, base_label: str,
                       n_runs: int = N_RUNS) -> tuple[list[float], list[float]]:
    """Return (mean_curve%, std_curve%) over N_RUNS runs."""
    curves = []
    for run in range(n_runs):
        c = _epoch_curve(all_results, f"{base_label}_r{run}")
        if c:
            curves.append(c)
    if not curves:
        return [], []
    min_len = min(len(c) for c in curves)
    arr = np.array([c[:min_len] for c in curves])
    return arr.mean(axis=0).tolist(), arr.std(axis=0).tolist()


# ---------------------------------------------------------------------------
# Plot 1 — Trust opinion mapping
# ---------------------------------------------------------------------------

def plot_trust_mapping(output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    sigmas = np.linspace(0, 0.6, 200)
    flips  = np.linspace(0, 0.35, 200)

    def _curves(opinions):
        b = [trust_components(o)[0] for o in opinions]
        d = [trust_components(o)[1] for o in opinions]
        u = [trust_components(o)[2] for o in opinions]
        return b, d, u

    ax = axes[0]
    ops_feat = [feature_noise_to_trust(s) for s in sigmas]
    b, d, u = _curves(ops_feat)
    ax.plot(sigmas, b, color="#2b7bba", lw=2, label="Belief $b$")
    ax.plot(sigmas, d, color="#c0392b", lw=2, label="Disbelief $d$")
    ax.plot(sigmas, u, color="#e07b39", lw=2, label="Uncertainty $u$")
    ax.set_xlabel(r"Relative noise level $\sigma_{rel}$")
    ax.set_ylabel("Opinion mass")
    ax.set_title("Feature noise → trust opinion")
    ax.legend()
    ax.set_xlim(0, 0.6)
    ax.set_ylim(-0.02, 1.02)

    ax = axes[1]
    ops_label = [label_noise_to_trust(p) for p in flips]
    b, d, u = _curves(ops_label)
    ax.plot(flips, b, color="#2b7bba", lw=2, label="Belief $b$")
    ax.plot(flips, d, color="#c0392b", lw=2, label="Disbelief $d$")
    ax.plot(flips, u, color="#e07b39", lw=2, label="Uncertainty $u$")
    ax.set_xlabel(r"Label flip rate $p$")
    ax.set_title("Label noise → trust opinion")
    ax.legend()
    ax.set_xlim(0, 0.35)
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    path = os.path.join(output_dir, "plots", "trust_mapping.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Shared helper: plot one noise-axis comparison (nn vs ptas-cal-fb)
# ---------------------------------------------------------------------------

def _plot_noise_axis(all_results: dict, ax, prefix_fn, x_vals,
                     xlabel: str, xlim: tuple):
    """Fill ax with mean±std curves for nn and ptas-cal-fb along one noise axis."""
    for cfg in ACC_PLOTS_CONFIGS:
        ms = [_acc_multi(all_results, f"{prefix_fn(v)}_{cfg}") for v in x_vals]
        means = np.array([m for m, _ in ms])
        stds  = np.array([s for _, s in ms])
        ax.plot(x_vals, means, color=_COLORS[cfg], marker=_MARKERS[cfg],
                lw=2, ms=7, label=_LABELS[cfg])
        ax.fill_between(x_vals, means - stds, means + stds,
                        alpha=0.18, color=_COLORS[cfg])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test accuracy (%)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xlim(*xlim)


# ---------------------------------------------------------------------------
# Plot 2 — Feature noise accuracy
# ---------------------------------------------------------------------------

def plot_feature_noise(all_results: dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    _plot_noise_axis(all_results, ax,
                     prefix_fn=lambda s: f"fn_{s:.2f}",
                     x_vals=FEATURE_SIGMAS_plots,
                     xlabel=r"Relative feature noise $\sigma_{rel}$",
                     xlim=(-0.02, 1.05))
    ax.set_title("NN: feature measurement noise")
    fig.tight_layout()
    path = os.path.join(output_dir, "plots", "feature_noise_accuracy.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot 3 — Label noise accuracy
# ---------------------------------------------------------------------------

def plot_label_noise(all_results: dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    _plot_noise_axis(all_results, ax,
                     prefix_fn=lambda p: f"ln_{p:.2f}",
                     x_vals=LABEL_FLIPS_plots,
                     xlabel=r"Label flip rate $p$",
                     xlim=(-0.01, 1.05))
    ax.set_title("NN: label flipping")
    fig.tight_layout()
    path = os.path.join(output_dir, "plots", "label_noise_accuracy.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot 4 — Combined noise (grouped bars with error bars)
# ---------------------------------------------------------------------------

def plot_combined(all_results: dict, output_dir: str):
    cond_labels = [f"σ={s}, p={p}" for s, p in COMBINED]
    x = np.arange(len(COMBINED))
    width = 0.30
    offsets = [-width / 2, width / 2]

    fig, ax = plt.subplots(figsize=(7, 4))
    for offset, cfg in zip(offsets, ACC_PLOTS_CONFIGS):
        ms   = [_acc_multi(all_results, f"comb_{s:.2f}_{p:.2f}_{cfg}") for s, p in COMBINED]
        accs = [m for m, _ in ms]
        errs = [s for _, s in ms]
        bars = ax.bar(x + offset, accs, width,
                      color=_COLORS[cfg], label=_LABELS[cfg],
                      yerr=errs, capsize=4,
                      error_kw={"elinewidth": 1.4, "ecolor": "black"})
        for bar, val in zip(bars, accs):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("NN: combined feature + label noise")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(output_dir, "plots", "combined_accuracy.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot 5 — PaTAS improvement (Δ accuracy = ptas-cal-fb − nn)
# ---------------------------------------------------------------------------

def plot_improvement(all_results: dict, output_dir: str):
    """Show the accuracy gain of PaTAS over the NN baseline across all conditions."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    def _delta(prefix_fn, x_vals):
        deltas, errs = [], []
        for v in x_vals:
            nn_m, nn_s  = _acc_multi(all_results, f"{prefix_fn(v)}_nn")
            fb_m, fb_s  = _acc_multi(all_results, f"{prefix_fn(v)}_ptas-cal-fb")
            deltas.append(fb_m - nn_m)
            errs.append(np.sqrt(nn_s**2 + fb_s**2))  # propagated std
        return np.array(deltas), np.array(errs)

    panels = [
        (axes[0], lambda s: f"fn_{s:.2f}", FEATURE_SIGMAS,
         r"Feature noise $\sigma_{rel}$", (-0.02, 0.55)),
        (axes[1], lambda p: f"ln_{p:.2f}", LABEL_FLIPS,
         r"Label flip rate $p$", (-0.01, 0.33)),
        (axes[2], lambda sp: f"comb_{sp[0]:.2f}_{sp[1]:.2f}", COMBINED,
         "Combined condition", None),
    ]

    for ax, prefix_fn, x_vals, xlabel, xlim in panels:
        deltas, errs = _delta(prefix_fn, x_vals)
        x_idx = np.arange(len(x_vals))
        ax.bar(x_idx, deltas, yerr=errs, color=_COLORS["ptas-cal-fb"],
               capsize=4, error_kw={"elinewidth": 1.4, "ecolor": "black"},
               zorder=3)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x_idx)
        if xlim is None:
            ax.set_xticklabels([f"σ={s}\np={p}" for s, p in x_vals], fontsize=8)
        else:
            ax.set_xticklabels([f"{v:.2f}" for v in x_vals])
            ax.set_xlabel(xlabel)
        ax.set_title(xlabel if xlim is not None else "Combined noise")
        ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)

    axes[0].set_ylabel(r"$\Delta$ accuracy (PaTAS − NN)  (%)")
    fig.suptitle("PaTAS accuracy gain over NN baseline (mean ± propagated std)",
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(output_dir, "plots", "patas_improvement.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot 6 — Learning curves (selected conditions)
# ---------------------------------------------------------------------------

def plot_learning_curves(all_results: dict, output_dir: str):
    """2×2 grid: feature noise and label noise conditions, nn vs ptas-cal-fb."""
    selected = [
        ("fn_0.10_", r"Feature noise $\sigma_{rel}=0.10$"),
        ("fn_0.30_", r"Feature noise $\sigma_{rel}=0.30$"),
        ("ln_0.05_", r"Label noise $p=0.05$"),
        ("ln_0.15_", r"Label noise $p=0.15$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)

    for ax, (prefix, title) in zip(axes.flat, selected):
        n_epochs_ax = 0
        for cfg in ACC_PLOTS_CONFIGS:
            mean_c, std_c = _epoch_curve_multi(all_results, f"{prefix}{cfg}")
            if mean_c:
                epochs_x = list(range(1, len(mean_c) + 1))
                n_epochs_ax = max(n_epochs_ax, len(mean_c))
                ax.plot(epochs_x, mean_c, color=_COLORS[cfg], lw=2, label=_LABELS[cfg])
                lo = np.array(mean_c) - np.array(std_c)
                hi = np.array(mean_c) + np.array(std_c)
                ax.fill_between(epochs_x, lo, hi, alpha=0.18, color=_COLORS[cfg])
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test accuracy (%)")
        ax.legend(fontsize=8)
        if n_epochs_ax > 0:
            ax.set_xlim(1, n_epochs_ax)

    fig.suptitle("Learning curves: NN under noise", fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "plots", "learning_curves.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def _booktabs_table(header: list[str], rows: list[list], caption: str,
                    label: str) -> str:
    col_fmt = "l" + "c" * (len(header) - 1)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(c) for c in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_trust_mapping(output_dir: str):
    header = [r"$\sigma_{rel}$ / flip $p$", "Belief $b$", "Disbelief $d$",
              "Uncertainty $u$", "Dominant component"]
    rows = []

    rows.append([r"\multicolumn{5}{l}{\textit{Feature noise (uncertainty-based)}}"])
    for s in FEATURE_SIGMAS:
        op = feature_noise_to_trust(s)
        b, d, u = op.t, op.d, op.u
        dom = "b" if b == max(b, d, u) else ("u" if u == max(b, d, u) else "d")
        rows.append([f"{s:.2f}", f"{b:.4f}", f"{d:.4f}", f"{u:.4f}", dom])

    rows.append([r"\midrule"])
    rows.append([r"\multicolumn{5}{l}{\textit{Label noise (disbelief-based)}}"])
    for p in LABEL_FLIPS:
        op = label_noise_to_trust(p)
        b, d, u = op.t, op.d, op.u
        dom = "b" if b == max(b, d, u) else ("d" if d == max(b, d, u) else "u")
        rows.append([f"{p:.2f}", f"{b:.4f}", f"{d:.4f}", f"{u:.4f}", dom])

    tex = _booktabs_table(
        header, rows,
        caption="Trust opinion mapping from noise parameters to subjective opinion $(b, d, u)$.",
        label="tab:trust-mapping",
    )
    path = os.path.join(output_dir, "tables", "trust_mapping.tex")
    with open(path, "w") as fh:
        fh.write(tex)
    print(f"  Saved {path}")


def table_feature_noise(all_results: dict, output_dir: str):
    header = [r"$\sigma_{rel}$",
              r"Trust opinion $(b,d,u)$",
              r"NN mean$\pm$std (\%)",
              r"PaTAS mean$\pm$std (\%)",
              r"$\Delta$ (\%)"]
    rows = []
    for s in FEATURE_SIGMAS:
        op = feature_noise_to_trust(s)
        b, d, u = op.t, op.d, op.u
        opinion_str = f"({b:.3f}, {d:.3f}, {u:.3f})"
        nn_m, nn_s  = _acc_multi(all_results, f"fn_{s:.2f}_nn")
        fb_m, fb_s  = _acc_multi(all_results, f"fn_{s:.2f}_ptas-cal-fb")
        nn_str  = f"{nn_m:.2f}$\\pm${nn_s:.2f}" if not np.isnan(nn_m) else "—"
        fb_str  = f"{fb_m:.2f}$\\pm${fb_s:.2f}" if not np.isnan(fb_m) else "—"
        delta   = fb_m - nn_m
        delta_str = f"{delta:+.2f}" if not (np.isnan(fb_m) or np.isnan(nn_m)) else "—"
        rows.append([f"{s:.2f}", opinion_str, nn_str, fb_str, delta_str])

    tex = _booktabs_table(
        header, rows,
        caption=(r"Test accuracy (\%) under feature measurement noise "
                 r"(mean$\pm$std over " + str(N_RUNS) + r" independent noise draws). "
                 r"$\Delta$ = PaTAS $-$ NN; positive values indicate PaTAS improvement."),
        label="tab:feature-noise",
    )
    path = os.path.join(output_dir, "tables", "feature_noise_results.tex")
    with open(path, "w") as fh:
        fh.write(tex)
    print(f"  Saved {path}")


def table_label_noise(all_results: dict, output_dir: str):
    header = [r"Flip rate $p$",
              r"Trust opinion $(b,d,u)$",
              r"NN mean$\pm$std (\%)",
              r"PaTAS mean$\pm$std (\%)",
              r"$\Delta$ (\%)"]
    rows = []
    for p in LABEL_FLIPS:
        op = label_noise_to_trust(p)
        b, d, u = op.t, op.d, op.u
        opinion_str = f"({b:.3f}, {d:.3f}, {u:.3f})"
        nn_m, nn_s  = _acc_multi(all_results, f"ln_{p:.2f}_nn")
        fb_m, fb_s  = _acc_multi(all_results, f"ln_{p:.2f}_ptas-cal-fb")
        nn_str  = f"{nn_m:.2f}$\\pm${nn_s:.2f}" if not np.isnan(nn_m) else "—"
        fb_str  = f"{fb_m:.2f}$\\pm${fb_s:.2f}" if not np.isnan(fb_m) else "—"
        delta   = fb_m - nn_m
        delta_str = f"{delta:+.2f}" if not (np.isnan(fb_m) or np.isnan(nn_m)) else "—"
        rows.append([f"{p:.2f}", opinion_str, nn_str, fb_str, delta_str])

    tex = _booktabs_table(
        header, rows,
        caption=(r"Test accuracy (\%) under label mislabelling "
                 r"(mean$\pm$std over " + str(N_RUNS) + r" independent noise draws). "
                 r"PaTAS scales each gradient update by the label trust belief mass, "
                 r"dampening updates from mislabelled batches. "
                 r"$\Delta$ = PaTAS $-$ NN."),
        label="tab:label-noise",
    )
    path = os.path.join(output_dir, "tables", "label_noise_results.tex")
    with open(path, "w") as fh:
        fh.write(tex)
    print(f"  Saved {path}")


def table_combined(all_results: dict, output_dir: str):
    header = [r"$(\sigma_{rel},\, p)$",
              r"x-trust $(b,d,u)$",
              r"y-trust $(b,d,u)$",
              r"NN mean$\pm$std (\%)",
              r"PaTAS mean$\pm$std (\%)",
              r"$\Delta$ (\%)"]
    rows = []
    for s, p in COMBINED:
        xop = feature_noise_to_trust(s)
        yop = label_noise_to_trust(p)
        xb, xd, xu = xop.t, xop.d, xop.u
        yb, yd, yu = yop.t, yop.d, yop.u
        cond_str  = f"({s:.2f}, {p:.2f})"
        xop_str   = f"({xb:.3f}, {xd:.3f}, {xu:.3f})"
        yop_str   = f"({yb:.3f}, {yd:.3f}, {yu:.3f})"
        nn_m, nn_s = _acc_multi(all_results, f"comb_{s:.2f}_{p:.2f}_nn")
        fb_m, fb_s = _acc_multi(all_results, f"comb_{s:.2f}_{p:.2f}_ptas-cal-fb")
        nn_str  = f"{nn_m:.2f}$\\pm${nn_s:.2f}" if not np.isnan(nn_m) else "—"
        fb_str  = f"{fb_m:.2f}$\\pm${fb_s:.2f}" if not np.isnan(fb_m) else "—"
        delta   = fb_m - nn_m
        delta_str = f"{delta:+.2f}" if not (np.isnan(fb_m) or np.isnan(nn_m)) else "—"
        rows.append([cond_str, xop_str, yop_str, nn_str, fb_str, delta_str])

    tex = _booktabs_table(
        header, rows,
        caption=(r"Test accuracy (\%) under combined feature and label noise "
                 r"(mean$\pm$std over " + str(N_RUNS) + r" noise draws). "
                 r"PaTAS applies calibrated uncertainty trust on features and "
                 r"disbelief trust on labels, with gradient feedback scaling. "
                 r"$\Delta$ = PaTAS $-$ NN."),
        label="tab:combined-noise",
    )
    path = os.path.join(output_dir, "tables", "combined_results.tex")
    with open(path, "w") as fh:
        fh.write(tex)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# PTAS effectiveness analysis
# ---------------------------------------------------------------------------

def ptas_effectiveness_analysis(ds, n_classes: int, output_dir: str, per_sample: str = "uniform"):
    """Assess whether PTAS output trust distinguishes NN correct vs. wrong predictions.

    Metric: projected probability π = b + u/K in the NN's predicted class,
    where K = n_classes.  This absorbs vacuous uncertainty via the base rate
    and is the canonical subjective-logic confidence scalar.

    Two noise axes are probed using run_seed=0 omega_thetas:

      Feature noise axis : omega from fn_{σ:.2f}_ptas-cal-fb_r0
                           query trust = feature_noise_to_trust(σ)
                           (same distribution as training)

      Label noise axis   : omega from ln_{p:.2f}_ptas-cal-fb_r0
                           query trust = TrustOpinion(1,0,0)
                           (test inputs are clean; PTAS learned label quality)

    For each axis, π is split by correct vs. wrong NN prediction.  A visible
    gap means PaTAS assigns higher projected confidence to correct predictions
    even though it never saw the test labels.

    Produces (one file per opinion source):
      plots/patas_effectiveness_agg.pdf  — aggregated (feedforward propagated)
      plots/patas_effectiveness_out.pdf  — output-layer weights (ABF-averaged)
      plots/patas_effectiveness_wt.pdf   — softmax-weighted blend
      tables/patas_effectiveness.tex     — summary table (agg source)
    """
    import pickle

    try:
        from patas_module.concrete.TensorTO import TensorArrayTO, av_fuse_gen, normalize_tensor
    except ImportError:
        print("  [skip] TensorArrayTO not importable — skipping effectiveness analysis")
        return

    from patas_module.NN.PTAStemplate import PTAS as PTASClass
    from patas_module.concrete.TrustOpinion import TrustOpinion
    from patas_module.NN.primaryNN import relu, softmax

    y_test_int = ds.y_test
    n_test, dim = ds.X_test.shape
    idx_eff    = np.arange(n_test)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_nn(nn_pkl_path: str, X_eval=None):
        """Return (nn_preds, a2) evaluated on X_eval (defaults to clean ds.X_test).

        Both NN and PTAS receive the same X_eval; only PTAS gets the calibrated
        trust opinion — the NN processes noisy input as-is (noise-unaware).
        """
        if not os.path.exists(nn_pkl_path):
            return None, None
        if X_eval is None:
            X_eval = ds.X_test
        with open(nn_pkl_path, "rb") as fh:
            w = pickle.load(fh)
        a1 = relu(X_eval @ w["W1"] + w["b1"])
        a2 = softmax(a1 @ w["W2"] + w["b2"])
        return a2.argmax(axis=1), a2

    def _mean_split(arr, mask, n_c, n_w):
        c = float(arr[mask].mean())  if n_c > 0 else float("nan")
        w = float(arr[~mask].mean()) if n_w > 0 else float("nan")
        return c, w

    def _store_comb(comb_recs, axis, key, b, d, u, nn_sc, correct, n_c, n_w):
        """Compute and store all five combined scores for one condition."""
        comb_recs["comb_bd"  ][axis][key] = _mean_split(nn_sc * ((b - d            + 1.0) / 2.0), correct, n_c, n_w)
        comb_recs["comb_b"][axis][key] = _mean_split(nn_sc *   b,                               correct, n_c, n_w)
        comb_recs["comb_bu"  ][axis][key] = _mean_split(nn_sc * ((b - u            + 1.0) / 2.0), correct, n_c, n_w)
        comb_recs["comb_bdu"  ][axis][key] = _mean_split(nn_sc * ((b - d - u / 2.0 + 1.0) / 2.0), correct, n_c, n_w)
        comb_recs["comb_pi"   ][axis][key] = _mean_split(nn_sc *  (b + u / 2.0),                   correct, n_c, n_w)

    def _compute_all_opinions(omega_pkl_path: str, a2: np.ndarray,
                               trust_or_tens, nn_preds: np.ndarray):
        """Return dict {"agg": (b,d,u), "out": (b,d,u), "wt": (b,d,u)}.

        trust_or_tens: either a TrustOpinion (uniform over all samples) or an
        ndarray of shape (n_test, dim, 3) for per-sample trust opinions.
        Per-sample trust is used for the mixed clean/noisy evaluation so that
        PTAS is aware of which samples were perturbed.
        """
        if not os.path.exists(omega_pkl_path):
            return None
        with open(omega_pkl_path, "rb") as fh:
            omega_data = pickle.load(fh)

        omega_thetas = [TensorArrayTO(w.astype(np.float32)) for w in omega_data]
        ptas = PTASClass(
            omega_thetas=omega_thetas,
            operator_mapping=None,
            nn_interface=None,
            trust_assessment_func=None,
            structure=[dim, N_HIDDEN, n_classes],
            use_tensor=True,
        )

        if isinstance(trust_or_tens, np.ndarray):
            tens = trust_or_tens.astype(np.float32)
        else:
            bv, dv, uv = float(trust_or_tens.t), float(trust_or_tens.d), float(trust_or_tens.u)
            tens = np.empty((n_test, dim, 3), dtype=np.float32)
            tens[..., 0] = bv; tens[..., 1] = dv; tens[..., 2] = uv
        Ty2 = ptas.apply_feedforward(TensorArrayTO(tens), tmp=False)  # (n_test, n_classes, 3)

        idx = np.arange(n_test)

        nn_score = a2[idx, nn_preds]

        _b_agg = Ty2.value[idx, nn_preds, 0]
        _d_agg = Ty2.value[idx, nn_preds, 1]
        _u_agg = Ty2.value[idx, nn_preds, 2]
        agg = bdu_with_weights(_b_agg, _d_agg, _u_agg, nn_score)
        return {"agg": agg}

    # ------------------------------------------------------------------
    # Collect metrics — feature noise axis
    # ------------------------------------------------------------------

    records = {"agg": {"fn": {}, "ln": {}, "cb": {}}}
    comb_records = {k: {"fn": {}, "ln": {}, "cb": {}}
                    for k in ("comb_bd", "comb_b",
                               "comb_bu", "comb_bdu", "comb_pi")}
    fn_tex_rows: list[list] = []

    # Build trust LUT for vectorised per-sample trust tensor construction.
    # sigma_i values lie in [0, 1] so 1001 steps give 0.001 resolution.
    _fn_sg_lut = np.linspace(0.0, 1.0, 1001)
    _fn_b_lut  = np.array([feature_noise_to_trust(float(s)).t for s in _fn_sg_lut], dtype=np.float32)
    _fn_d_lut  = np.array([feature_noise_to_trust(float(s)).d for s in _fn_sg_lut], dtype=np.float32)
    _fn_u_lut  = np.array([feature_noise_to_trust(float(s)).u for s in _fn_sg_lut], dtype=np.float32)

    def _per_sample_tens(sigma_arr: np.ndarray, n_dim: int) -> np.ndarray:
        """Return (n, dim, 3) float32 trust tensor from per-sample sigma array."""
        si = (np.asarray(sigma_arr, dtype=np.float32) * 1000).astype(int).clip(0, 1000)
        t = np.empty((len(sigma_arr), n_dim, 3), dtype=np.float32)
        t[:, :, 0] = _fn_b_lut[si][:, None]
        t[:, :, 1] = _fn_d_lut[si][:, None]
        t[:, :, 2] = _fn_u_lut[si][:, None]
        return t

    for sigma in FEATURE_SIGMAS:
        base    = os.path.join(output_dir, "data")
        omega_p = os.path.join(base, f"fn_{sigma:.2f}_ptas-cal-fb_r0", "omega_thetas.pkl")
        nn_p    = os.path.join(base, f"fn_{sigma:.2f}_ptas-cal-fb_r0", "nn_weights.pkl")
        if per_sample == "per_sample":
            X_noisy_fn, sigma_arr_fn = add_feature_noise_per_sample(
                ds.X_test, sigma, rng=np.random.default_rng(42)
            )
            fn_trust = _per_sample_tens(sigma_arr_fn, dim)
            nn_preds, a2 = _load_nn(nn_p, X_eval=X_noisy_fn)
        elif per_sample == "fully_trusted":
            fn_trust = TrustOpinion(1.0, 0.0, 0.0)
            nn_preds, a2 = _load_nn(nn_p)
        else:
            fn_trust = feature_noise_to_trust(sigma)
            nn_preds, a2 = _load_nn(nn_p)
        if nn_preds is None:
            continue
        correct   = (nn_preds == y_test_int)
        n_correct = int(correct.sum())
        n_wrong   = int((~correct).sum())
        opinions = _compute_all_opinions(omega_p, a2, fn_trust, nn_preds)
        if opinions is None:
            continue
        for src in ["agg"]:
            b_arr, d_arr, u_arr = opinions[src]
            b_c, b_w = _mean_split(b_arr, correct, n_correct, n_wrong)
            d_c, d_w = _mean_split(d_arr, correct, n_correct, n_wrong)
            u_c, u_w = _mean_split(u_arr, correct, n_correct, n_wrong)
            records[src]["fn"][sigma] = (b_c, b_w, d_c, d_w, u_c, u_w)
        b_agg, d_agg, u_agg = opinions["agg"]
        nn_sc = a2[idx_eff, nn_preds]
        _store_comb(comb_records, "fn", sigma, b_agg, d_agg, u_agg, nn_sc, correct, n_correct, n_wrong)
        b_c, b_w, d_c, d_w, u_c, u_w = records["agg"]["fn"][sigma]
        b_gap = f"{b_c - b_w:+.4f}" if not (np.isnan(b_c) or np.isnan(b_w)) else "—"
        d_gap = f"{d_c - d_w:+.4f}" if not (np.isnan(d_c) or np.isnan(d_w)) else "—"
        u_gap = f"{u_c - u_w:+.4f}" if not (np.isnan(u_c) or np.isnan(u_w)) else "—"
        fn_tex_rows.append([
            f"{sigma:.2f}", str(n_correct), str(n_wrong),
            f"{b_c:.4f}", f"{b_w:.4f}", b_gap,
            f"{d_c:.4f}", f"{d_w:.4f}", d_gap,
            f"{u_c:.4f}", f"{u_w:.4f}", u_gap,
        ])

    # ------------------------------------------------------------------
    # Collect metrics — label noise axis (clean query trust)
    # ------------------------------------------------------------------

    trusted_op = TrustOpinion(1.0, 0.0, 0.0)
    ln_tex_rows: list[list] = []

    for flip in LABEL_FLIPS:
        base    = os.path.join(output_dir, "data")
        omega_p = os.path.join(base, f"ln_{flip:.2f}_ptas-cal-fb_r0", "omega_thetas.pkl")
        nn_p    = os.path.join(base, f"ln_{flip:.2f}_ptas-cal-fb_r0", "nn_weights.pkl")
        nn_preds, a2 = _load_nn(nn_p)
        if nn_preds is None:
            continue
        correct   = (nn_preds == y_test_int)
        n_correct = int(correct.sum())
        n_wrong   = int((~correct).sum())
        opinions = _compute_all_opinions(omega_p, a2, trusted_op, nn_preds)
        if opinions is None:
            continue
        for src in ["agg"]:
            b_arr, d_arr, u_arr = opinions[src]
            b_c, b_w = _mean_split(b_arr, correct, n_correct, n_wrong)
            d_c, d_w = _mean_split(d_arr, correct, n_correct, n_wrong)
            u_c, u_w = _mean_split(u_arr, correct, n_correct, n_wrong)
            records[src]["ln"][flip] = (b_c, b_w, d_c, d_w, u_c, u_w)
        b_agg, d_agg, u_agg = opinions["agg"]
        nn_sc = a2[idx_eff, nn_preds]
        _store_comb(comb_records, "ln", flip, b_agg, d_agg, u_agg, nn_sc, correct, n_correct, n_wrong)
        # tex rows from agg only
        b_c, b_w, d_c, d_w, u_c, u_w = records["agg"]["ln"][flip]
        b_gap = f"{b_c - b_w:+.4f}" if not (np.isnan(b_c) or np.isnan(b_w)) else "—"
        d_gap = f"{d_c - d_w:+.4f}" if not (np.isnan(d_c) or np.isnan(d_w)) else "—"
        u_gap = f"{u_c - u_w:+.4f}" if not (np.isnan(u_c) or np.isnan(u_w)) else "—"
        ln_tex_rows.append([
            f"{flip:.2f}", str(n_correct), str(n_wrong),
            f"{b_c:.4f}", f"{b_w:.4f}", b_gap,
            f"{d_c:.4f}", f"{d_w:.4f}", d_gap,
            f"{u_c:.4f}", f"{u_w:.4f}", u_gap,
        ])

    # ------------------------------------------------------------------
    # Collect metrics — combined noise axis (calibrated feature trust)
    # ------------------------------------------------------------------

    cb_tex_rows: list[list] = []

    for sigma, flip in COMBINED:
        base    = os.path.join(output_dir, "data")
        omega_p = os.path.join(base, f"comb_{sigma:.2f}_{flip:.2f}_ptas-cal-fb_r0",
                               "omega_thetas.pkl")
        nn_p    = os.path.join(base, f"comb_{sigma:.2f}_{flip:.2f}_ptas-cal-fb_r0",
                               "nn_weights.pkl")
        if per_sample == "per_sample":
            X_noisy_cb, sigma_arr_cb = add_feature_noise_per_sample(
                ds.X_test, sigma, rng=np.random.default_rng(42)
            )
            cb_trust = _per_sample_tens(sigma_arr_cb, dim)
            nn_preds, a2 = _load_nn(nn_p, X_eval=X_noisy_cb)
        elif per_sample == "fully_trusted":
            cb_trust = TrustOpinion(1.0, 0.0, 0.0)
            nn_preds, a2 = _load_nn(nn_p)
        else:
            cb_trust = feature_noise_to_trust(sigma)
            nn_preds, a2 = _load_nn(nn_p)
        if nn_preds is None:
            continue
        correct   = (nn_preds == y_test_int)
        n_correct = int(correct.sum())
        n_wrong   = int((~correct).sum())
        opinions = _compute_all_opinions(omega_p, a2, cb_trust, nn_preds)
        if opinions is None:
            continue
        for src in ["agg"]:
            b_arr, d_arr, u_arr = opinions[src]
            b_c, b_w = _mean_split(b_arr, correct, n_correct, n_wrong)
            d_c, d_w = _mean_split(d_arr, correct, n_correct, n_wrong)
            u_c, u_w = _mean_split(u_arr, correct, n_correct, n_wrong)
            records[src]["cb"][(sigma, flip)] = (b_c, b_w, d_c, d_w, u_c, u_w)
        b_agg, d_agg, u_agg = opinions["agg"]
        nn_sc = a2[idx_eff, nn_preds]
        _store_comb(comb_records, "cb", (sigma, flip), b_agg, d_agg, u_agg, nn_sc, correct, n_correct, n_wrong)
        # tex rows from agg only
        b_c, b_w, d_c, d_w, u_c, u_w = records["agg"]["cb"][(sigma, flip)]
        b_gap = f"{b_c - b_w:+.4f}" if not (np.isnan(b_c) or np.isnan(b_w)) else "—"
        d_gap = f"{d_c - d_w:+.4f}" if not (np.isnan(d_c) or np.isnan(d_w)) else "—"
        u_gap = f"{u_c - u_w:+.4f}" if not (np.isnan(u_c) or np.isnan(u_w)) else "—"
        cb_tex_rows.append([
            f"({sigma:.2f}, {flip:.2f})", str(n_correct), str(n_wrong),
            f"{b_c:.4f}", f"{b_w:.4f}", b_gap,
            f"{d_c:.4f}", f"{d_w:.4f}", d_gap,
            f"{u_c:.4f}", f"{u_w:.4f}", u_gap,
        ])

    has_data = any(records["agg"][axis] for axis in ("fn", "ln", "cb"))
    if not has_data:
        print("  [skip] No ptas-cal-fb runs found — skipping effectiveness analysis")
        return

    # ------------------------------------------------------------------
    # Plot: one 2×3 grid per opinion source
    # ------------------------------------------------------------------

    _C = {"b": "#2b7bba", "d": "#c0392b", "u": "#e07b39"}

    src_meta = {
        "agg": ("Aggregated (feedforward propagated)", "patas_effectiveness_agg"),
    }

    cb_x     = list(range(len(COMBINED)))
    cb_ticks = [f"$\\sigma$={s:.2f}\n$p$={p:.2f}" for s, p in COMBINED]

    for src, (src_title, src_fname) in src_meta.items():
        fn_recs = records[src]["fn"]
        ln_recs = records[src]["ln"]
        cb_recs = records[src]["cb"]
        if not fn_recs and not ln_recs and not cb_recs:
            continue

        fig, axes = plt.subplots(2, 3, figsize=(15, 7),
                                 gridspec_kw={"height_ratios": [2, 1]})

        # Panel specs: (ax_top, ax_bot, recs, x_vals, tick_labels, xlabel, title)
        panels = [
            (axes[0, 0], axes[1, 0],
             {v: fn_recs[v] for v in sorted(FEATURE_SIGMAS) if v in fn_recs},
             sorted(FEATURE_SIGMAS), None,
             r"Feature noise $\sigma_{rel}$",
             r"Feature noise ($\sigma_{rel}$-calibrated $T_x$)"),
            (axes[0, 1], axes[1, 1],
             {v: ln_recs[v] for v in sorted(LABEL_FLIPS) if v in ln_recs},
             sorted(LABEL_FLIPS), None,
             r"Label flip rate $p$",
             r"Label noise (trusted $T_x$)"),
            (axes[0, 2], axes[1, 2],
             {i: cb_recs[k] for i, k in enumerate(COMBINED) if k in cb_recs},
             cb_x, cb_ticks,
             "Condition",
             "Combined noise ($T_x$ calibrated)"),
        ]

        nan6 = (float("nan"),) * 6
        first_col = True
        for ax_top, ax_bot, recs, x_vals, tick_labels, xlabel, title in panels:
            b_c = [recs.get(v, nan6)[0] for v in x_vals]
            b_w = [recs.get(v, nan6)[1] for v in x_vals]
            d_c = [recs.get(v, nan6)[2] for v in x_vals]
            d_w = [recs.get(v, nan6)[3] for v in x_vals]
            u_c = [recs.get(v, nan6)[4] for v in x_vals]
            u_w = [recs.get(v, nan6)[5] for v in x_vals]

            for vals, color, ls, mk, lbl in [
                (b_c, _C["b"], "-",  "o", "Belief (correct)"),
                (b_w, _C["b"], "--", "s", "Belief (wrong)"),
                (d_c, _C["d"], "-",  "o", "Disbelief (correct)"),
                (d_w, _C["d"], "--", "s", "Disbelief (wrong)"),
                (u_c, _C["u"], "-",  "o", "Uncertainty (correct)"),
                (u_w, _C["u"], "--", "s", "Uncertainty (wrong)"),
            ]:
                ax_top.plot(x_vals, vals, color=color, linestyle=ls, marker=mk,
                            lw=2, ms=6, label=lbl)

            if first_col:
                ax_top.set_ylabel("Mean opinion mass in NN's pred. class")
            ax_top.set_title(title, fontsize=10)
            ax_top.legend(fontsize=8, ncol=2)
            all_vals = [v for lst in [b_c, b_w, d_c, d_w, u_c, u_w]
                        for v in lst if not np.isnan(v)]
            if all_vals:
                ylo, yhi = min(all_vals), max(all_vals)
                margin = max(0.03, (yhi - ylo) * 0.3)
                ax_top.set_ylim(max(0.0, ylo - margin), min(1.0, yhi + margin))

            xs = np.array(x_vals, dtype=float)
            bw_bar = min(np.diff(xs).min() if len(xs) > 1 else 1.0,
                         0.04 if tick_labels is None else 0.25) * 0.9
            xlim   = (xs[0] - 2.5 * bw_bar, xs[-1] + 2.5 * bw_bar)
            ax_top.set_xlim(*xlim)
            if tick_labels is not None:
                ax_top.set_xticks(x_vals)
                ax_top.set_xticklabels(tick_labels, fontsize=12)

            gaps_b = [bc - bw if not (np.isnan(bc) or np.isnan(bw)) else 0.0
                      for bc, bw in zip(b_c, b_w)]
            gaps_d = [dc - dw if not (np.isnan(dc) or np.isnan(dw)) else 0.0
                      for dc, dw in zip(d_c, d_w)]
            gaps_u = [uc - uw if not (np.isnan(uc) or np.isnan(uw)) else 0.0
                      for uc, uw in zip(u_c, u_w)]

            for off, gaps, color, lbl in zip(
                [-bw_bar, 0.0, bw_bar],
                [gaps_b, gaps_d, gaps_u],
                [_C["b"], _C["d"], _C["u"]],
                ["b", "d", "u"],
            ):
                bars = ax_bot.bar(xs + off, gaps, width=bw_bar, color=color,
                                  label=lbl, zorder=3)
                for bar, g in zip(bars, gaps):
                    if g == 0.0:
                        continue
                    va  = "bottom" if g >= 0 else "top"
                    pad = 0.0005 if g >= 0 else -0.0005
                    ax_bot.text(bar.get_x() + bar.get_width() / 2, g + pad,
                                f"{g:+.3f}", ha="center", va=va, fontsize=8,
                                color=color)
            ax_bot.axhline(0, color="black", lw=0.8)
            ax_bot.set_xlabel(xlabel)
            if first_col:
                ax_bot.set_ylabel(r"$\Delta$")
            ax_bot.set_xlim(*xlim)
            ax_bot.legend(fontsize=8, ncol=3)
            ax_bot.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
            if tick_labels is not None:
                ax_bot.set_xticks(x_vals)
                ax_bot.set_xticklabels(tick_labels, fontsize=12)
            first_col = False

        fig.suptitle(f"PaTAS confidence signal: opinion masses in predicted class",
                     fontsize=11)
        try:
            fig.tight_layout()
        except Exception as e:
            print(f"Error occurred while adjusting layout: {e}")
        path = os.path.join(output_dir, "plots", f"{src_fname}.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")

    # ------------------------------------------------------------------
    # Combined-signal effectiveness plots: comb_bd and comb_b
    # Each shows mean combined score for correct vs. wrong predictions.
    # ------------------------------------------------------------------
    _comb_meta = [
        ("comb_bd",   r"$(b{-}d)$",                    "#e07b39", "patas_effectiveness_comb_bd"),
        ("comb_b", r"$b$",                           "#8e44ad", "patas_effectiveness_comb_b"),
        ("comb_bu",   r"$(b{-}u)$",                    "#1abc9c", "patas_effectiveness_comb_bu"),
        ("comb_bdu",   r"$(b{-}d{-}\frac{u}{2})$",      "#e74c3c", "patas_effectiveness_comb_bdu"),
        ("comb_pi",    r"$\pi$  $(b+\frac{u}{2})$",     "#2980b9", "patas_effectiveness_comb_pi"),
    ]
    cb_x     = list(range(len(COMBINED)))
    cb_ticks = [f"$\\sigma$={s:.2f}\n$p$={p:.2f}" for s, p in COMBINED]
    nan2     = (float("nan"),) * 2

    for comb_key, comb_title, comb_color, comb_fname in _comb_meta:
        fn_r = comb_records[comb_key]["fn"]
        ln_r = comb_records[comb_key]["ln"]
        cb_r = comb_records[comb_key]["cb"]
        if not fn_r and not ln_r and not cb_r:
            continue

        fig, axes = plt.subplots(2, 3, figsize=(15, 7),
                                 gridspec_kw={"height_ratios": [2, 1]})
        panels = [
            (axes[0, 0], axes[1, 0],
             {v: fn_r.get(v, nan2) for v in sorted(FEATURE_SIGMAS)},
             sorted(FEATURE_SIGMAS), None,
             r"Feature noise $\sigma_{rel}$", "Feature noise"),
            (axes[0, 1], axes[1, 1],
             {v: ln_r.get(v, nan2) for v in sorted(LABEL_FLIPS)},
             sorted(LABEL_FLIPS), None,
             r"Label flip rate $p$", "Label noise"),
            (axes[0, 2], axes[1, 2],
             {i: cb_r.get(k, nan2) for i, k in enumerate(COMBINED)},
             cb_x, cb_ticks, "Condition", "Combined noise"),
        ]

        first_col = True
        for ax_top, ax_bot, panel_r, x_vals, tick_labels, xlabel, title in panels:
            m_c = [panel_r.get(v, nan2)[0] for v in x_vals]
            m_w = [panel_r.get(v, nan2)[1] for v in x_vals]

            ax_top.plot(x_vals, m_c, color=comb_color, lw=2, marker="o",
                        linestyle="-",  label="Correct predictions")
            ax_top.plot(x_vals, m_w, color=comb_color, lw=2, marker="s",
                        linestyle="--", label="Wrong predictions")

            if first_col:
                ax_top.set_ylabel(f"Mean {comb_title}")
            ax_top.set_title(title, fontsize=10)
            ax_top.legend(fontsize=12)
            all_v = [v for v in m_c + m_w if not np.isnan(v)]
            if all_v:
                ylo, yhi = min(all_v), max(all_v)
                margin = max(0.01, (yhi - ylo) * 0.3)
                ax_top.set_ylim(max(0.0, ylo - margin), min(1.0, yhi + margin))

            xs     = np.array(x_vals, dtype=float)
            bw_bar = min(np.diff(xs).min() if len(xs) > 1 else 1.0,
                         0.04 if tick_labels is None else 0.25) * 0.9
            xlim   = (xs[0] - 2.5 * bw_bar, xs[-1] + 2.5 * bw_bar)
            ax_top.set_xlim(*xlim)
            if tick_labels is not None:
                ax_top.set_xticks(x_vals)
                ax_top.set_xticklabels(tick_labels, fontsize=12)

            gaps = [c - w if not (np.isnan(c) or np.isnan(w)) else 0.0
                    for c, w in zip(m_c, m_w)]
            bars = ax_bot.bar(xs, gaps, width=bw_bar, color=comb_color, zorder=3)
            for bar, g in zip(bars, gaps):
                if g == 0.0:
                    continue
                va  = "bottom" if g >= 0 else "top"
                pad = 0.0005 if g >= 0 else -0.0005
                ax_bot.text(bar.get_x() + bar.get_width() / 2, g + pad,
                            f"{g:+.3f}", ha="center", va=va,
                            fontsize=12, color=comb_color)
            ax_bot.axhline(0, color="black", lw=0.8)
            ax_bot.set_xlabel(xlabel)
            if first_col:
                ax_bot.set_ylabel(r"$\Delta$ (correct $-$ wrong)")
            ax_bot.set_xlim(*xlim)
            ax_bot.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
            if tick_labels is not None:
                ax_bot.set_xticks(x_vals)
                ax_bot.set_xticklabels(tick_labels, fontsize=12)
            first_col = False

        fig.suptitle(
            f"PaTAS signal: correct vs. wrong predictions",
            fontsize=11)
        fig.tight_layout()
        path = os.path.join(output_dir, "plots", f"{comb_fname}.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")

    # ------------------------------------------------------------------
    # Table (agg source only)
    # ------------------------------------------------------------------

    col_header = [
        r"Noise", r"$N_c$", r"$N_w$",
        r"$\bar{b}_c$", r"$\bar{b}_w$", r"$\Delta b$",
        r"$\bar{d}_c$", r"$\bar{d}_w$", r"$\Delta d$",
        r"$\bar{u}_c$", r"$\bar{u}_w$", r"$\Delta u$",
    ]

    all_rows: list[list] = []
    if fn_tex_rows:
        all_rows.append(
            [r"\multicolumn{12}{l}{\textit{Feature noise (calibrated query trust)}}"])
        all_rows += fn_tex_rows
    if ln_tex_rows:
        if all_rows:
            all_rows.append([r"\midrule"])
        all_rows.append(
            [r"\multicolumn{12}{l}{\textit{Label noise (clean query trust)}}"])
        all_rows += ln_tex_rows
    if cb_tex_rows:
        if all_rows:
            all_rows.append([r"\midrule"])
        all_rows.append(
            [r"\multicolumn{12}{l}{\textit{Combined noise (calibrated query trust)}}"])
        all_rows += cb_tex_rows

    tex = _booktabs_table(
        col_header, all_rows,
        caption=(
            r"Mean PaTAS opinion masses $(b, d, u)$ in the NN's predicted class, "
            r"split by correct ($c$) vs.\ wrong ($w$) NN predictions. "
            r"$\Delta = \text{correct} - \text{wrong}$; "
            r"positive $\Delta b$ and negative $\Delta d$/$\Delta u$ indicate discriminability."
        ),
        label="tab:patas-effectiveness",
    )
    tpath = os.path.join(output_dir, "tables", "patas_effectiveness.tex")
    with open(tpath, "w") as fh:
        fh.write(tex)
    print(f"  Saved {tpath}")


def bdu_with_weights(b, d, u, nn_score):
    b_w = nn_score * b
    d_w = (1 - nn_score) * d
    u_w = nn_score * (1 - nn_score) * u
    denom = b_w + d_w + u_w
    # When denom=0 (degenerate opinion), fall back to the original (b, d, u).
    denom_safe = np.where(denom > 0, denom, 1.0)
    b_f = np.where(denom > 0, b_w / denom_safe, b)
    d_f = np.where(denom > 0, d_w / denom_safe, d)
    u_f = 1 - (b_f + d_f)
    return b_f, d_f, u_f
# ---------------------------------------------------------------------------
# Coverage vs. accuracy curves
# ---------------------------------------------------------------------------

def plot_coverage_accuracy(ds, n_classes: int, output_dir: str):
    """Coverage vs. accuracy curves sweeping confidence thresholds.

    For PaTAS sources (agg, out, wt): threshold on projected probability π = b + u/K.
    For NN baseline: threshold on softmax score.

    Produces one plot per noise axis:
      plots/coverage_accuracy_feature.pdf
      plots/coverage_accuracy_label.pdf
      plots/coverage_accuracy_combined.pdf
    """
    import pickle

    try:
        from patas_module.concrete.TensorTO import TensorArrayTO
        from patas_module.NN.PTAStemplate import PTAS as PTASClass
        from patas_module.concrete.TrustOpinion import TrustOpinion
        from patas_module.NN.primaryNN import relu, softmax
    except ImportError:
        print("  [skip] skipping coverage-accuracy plot")
        return

    y_test  = ds.y_test
    n_test, dim = ds.X_test.shape
    idx     = np.arange(n_test)

    # taus_nn is built per-run from actual score quantiles (see _load_curves).

    # ------------------------------------------------------------------
    # Pre-build one shared noisy test set (all samples, σᵢ ~ U(0,1)).
    # Both NN and PTAS receive this same X; PTAS gets per-sample calibrated
    # trust; NN processes noisy X as-is (noise-unaware baseline).
    # ------------------------------------------------------------------
    from noise_utils import add_feature_noise as _afn_cv
    _sigmas_cv   = np.random.default_rng(1).uniform(0.0, 1.0, n_test)
    _noise_cv    = np.random.default_rng(2).normal(0, 1, ds.X_test.shape).astype(np.float32)
    _X_noisy_cv  = (ds.X_test
                    + _noise_cv * (_sigmas_cv[:, None] * float(ds.X_test.std()))
                    ).astype(np.float32)

    _sg_cv  = np.linspace(0.0, 1., 1001)
    _bg_cv  = np.array([feature_noise_to_trust(float(s)).t for s in _sg_cv], dtype=np.float32)
    _dg_cv  = np.array([feature_noise_to_trust(float(s)).d for s in _sg_cv], dtype=np.float32)
    _ug_cv  = np.array([feature_noise_to_trust(float(s)).u for s in _sg_cv], dtype=np.float32)
    _si_cv  = (_sigmas_cv * 1000).astype(int).clip(0, 1000)
    _tens_cv = np.empty((n_test, dim, 3), dtype=np.float32)
    _tens_cv[:, :, 0] = _bg_cv[_si_cv][:, None]
    _tens_cv[:, :, 1] = _dg_cv[_si_cv][:, None]
    _tens_cv[:, :, 2] = _ug_cv[_si_cv][:, None]

    def _load_curves(omega_p, nn_p):
        """Return {src: (cov_arr, acc_arr)} or None if pkl files missing.

        NN runs on _X_noisy_cv (noise-unaware).
        PaTAS feedforward uses _tens_cv (per-sample calibrated trust, noise-aware).
        PaTAS sources sweep τ ∈ [-1, 1]; NN / combined sweep τ ∈ [0, 1].
        """
        if not (os.path.exists(omega_p) and os.path.exists(nn_p)):
            return None
        with open(nn_p, "rb") as fh:
            w = pickle.load(fh)
        a1 = relu(_X_noisy_cv @ w["W1"] + w["b1"])
        a2 = softmax(a1 @ w["W2"] + w["b2"])
        nn_preds = a2.argmax(axis=1)

        with open(omega_p, "rb") as fh:
            omega_data = pickle.load(fh)
        omega_thetas = [TensorArrayTO(ow.astype(np.float32)) for ow in omega_data]
        ptas = PTASClass(
            omega_thetas=omega_thetas,
            operator_mapping=None,
            nn_interface=None,
            trust_assessment_func=None,
            structure=[dim, N_HIDDEN, n_classes],
            use_tensor=True,
        )
        Ty2 = ptas.apply_feedforward(TensorArrayTO(_tens_cv), tmp=False)  # (n, cls, 3)

        b_agg    = Ty2.value[idx, nn_preds, 0]
        d_agg    = Ty2.value[idx, nn_preds, 1]
        u_agg    = Ty2.value[idx, nn_preds, 2]
        nn_score = a2[idx, nn_preds]

        # All combined signals = nn_score × normalised PTAS score (∈ [0,1])
        # (b−d) ∈ [−1,1]       → normalise: (b−d+1)/2
        # b     ∈ [0,1]         → already normalised
        # (b−u) ∈ [−1,1]       → normalise: (b−u+1)/2
        # (b−d−u/2) ∈ [−1,1]  → normalise: (b−d−u/2+1)/2
        # π=b+u/2 ∈ [0,1]      → already normalised
        # s = {
        #     "comb_bd":   nn_score * ((b_agg - d_agg          + 1.0) / 2.0),
        #     "comb_b": nn_score *   b_agg,
        #     "comb_bu":   nn_score * ((b_agg - u_agg          + 1.0) / 2.0),
        #     "comb_bdu":   nn_score * ((b_agg - d_agg - u_agg / 2.0 + 1.0) / 2.0),
        #     "comb_pi":    nn_score *  (b_agg + u_agg / 2.0),
        # }

        b_agg_n, d_agg_n, u_agg_n = bdu_with_weights(b_agg, d_agg, u_agg, nn_score)
        s = {
            "comb_bd":   ((b_agg_n - d_agg_n          + 1.0) / 2.0),
            "comb_b": b_agg_n,
            "comb_bu":   ((b_agg_n - u_agg_n          + 1.0) / 2.0),
            "comb_bdu":   ((b_agg_n - d_agg_n - u_agg_n / 2.0 + 1.0) / 2.0),
            "comb_pi":    (b_agg_n + u_agg_n / 2.0),
        }

        # Build quantile-based thresholds from the actual score distributions
        # so the sweep covers the full range without coarse-sampling the tail.
        _all_scores = np.concatenate([nn_score] + [v for v in s.values()])
        _all_scores = _all_scores[np.isfinite(_all_scores)]
        taus_nn = np.unique(np.quantile(_all_scores, np.linspace(0.0, 1.0, 600)))

        def _sweep(scores, taus_arr):
            covs = np.empty(len(taus_arr))
            accs = np.full(len(taus_arr), np.nan)
            for i, tau in enumerate(taus_arr):
                mask  = scores >= tau
                n_cov = int(mask.sum())
                covs[i] = 100.0 * n_cov / n_test
                if n_cov > 0:
                    accs[i] = 100.0 * float(
                        (nn_preds[mask] == y_test[mask]).mean())
            return covs, accs

        return {"nn": _sweep(nn_score, taus_nn),
                **{k: _sweep(v, taus_nn) for k, v in s.items()}}

    axes_spec = [
        ("feature",
         "Feature noise",
         [(os.path.join(output_dir, "data", f"fn_{s:.2f}_ptas-cal-fb_r0", "omega_thetas.pkl"),
           os.path.join(output_dir, "data", f"fn_{s:.2f}_ptas-cal-fb_r0", "nn_weights.pkl"),
           f"$\\sigma_{{rel,train}}={s:.2f}$")
          for s in FEATURE_SIGMAS[:4]]),
        ("label",
         "Label noise",
         [(os.path.join(output_dir, "data", f"ln_{f:.2f}_ptas-cal-fb_r0", "omega_thetas.pkl"),
           os.path.join(output_dir, "data", f"ln_{f:.2f}_ptas-cal-fb_r0", "nn_weights.pkl"),
           f"$p_{{train}}={f:.2f}$")
          for f in LABEL_FLIPS[:4]]),
        ("combined",
         "Combined noise",
         [(os.path.join(output_dir, "data", f"comb_{s:.2f}_{f:.2f}_ptas-cal-fb_r0", "omega_thetas.pkl"),
           os.path.join(output_dir, "data", f"comb_{s:.2f}_{f:.2f}_ptas-cal-fb_r0", "nn_weights.pkl"),
           f"$\\sigma_{{train}}={s:.2f},\\,p_{{train}}={f:.2f}$")
          for s, f in COMBINED[:4]]),
    ]

    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    try:
        from sklearn.metrics import roc_curve as _sk_roc, auc as _sk_auc
        _has_sklearn = True
    except ImportError:
        _has_sklearn = False

    for ax_key, ax_title, conditions in axes_spec:
        panels = []
        for omega_p, nn_p, panel_title in conditions:
            result = _load_curves(omega_p, nn_p)
            if result is not None:
                panels.append((panel_title, result))

        if not panels:
            continue

        ncols = 4
        nrows = 2  # row 0: coverage-accuracy; row 1: ROC curves

        fig, axes_arr = plt.subplots(nrows, ncols,
                                     figsize=(5.5 * ncols, 4.5 * nrows),
                                     squeeze=False)

        for pi, (panel_title, curves) in enumerate(panels):
            ax_cov = axes_arr[0][pi % ncols]
            ax_roc = axes_arr[1][pi % ncols]

            # ---- row 0: coverage-accuracy ----
            for src in _THR_SOURCES:
                covs, accs = curves[src]
                valid = ~np.isnan(accs)
                if not valid.any():
                    continue
                ax_cov.plot(covs[valid], accs[valid],
                            color=_THR_COLORS[src], lw=2, ls=_THR_LINESTYLES[src],
                            label=_THR_LABELS[src])
                ax_cov.scatter([covs[0]], [accs[0] if not np.isnan(accs[0]) else np.nan],
                               color=_THR_COLORS[src], s=35, zorder=5)

            ax_cov.set_title(panel_title, fontsize=16)
            ax_cov.set_xlabel("Coverage (%)")
            ax_cov.set_ylabel("Accuracy on covered (%)")
            ax_cov.set_xlim(-2, 105)
            ax_cov.legend(fontsize=16, loc="lower left")
            ax_cov.grid(linestyle=":", alpha=0.4)

            # ---- row 1: ROC curves ----
            if _has_sklearn and "nn" in curves and hasattr(curves["nn"], "__len__"):
                # Recover binary correct-vs-incorrect labels from the sweep data.
                # We re-derive them from the raw scores stored in the closure.
                # Since _load_curves closes over nn_score and s, recompute here.
                _omega_p, _nn_p, _ = conditions[pi]
                raw = _load_curves(_omega_p, _nn_p)
                if raw is not None:
                    # Re-run _load_curves gives same object; extract stored arrays.
                    pass

            # ROC requires (y_true_binary, score) per sample — we cannot recover
            # them from the already-swept (covs, accs) pairs alone.  Re-derive
            # from the raw model outputs by calling _load_curves again and
            # capturing per-sample data via a thin wrapper.
            _omega_p, _nn_p, _ = conditions[pi]
            if _has_sklearn and os.path.exists(_omega_p) and os.path.exists(_nn_p):
                import pickle as _pk
                with open(_nn_p, "rb") as _fh:
                    _w = _pk.load(_fh)
                from patas_module.NN.primaryNN import relu as _relu, softmax as _softmax_fn
                _a1 = _relu(_X_noisy_cv @ _w["W1"] + _w["b1"])
                _a2 = _softmax_fn(_a1 @ _w["W2"] + _w["b2"])
                _nn_pred = _a2.argmax(axis=1)
                _correct_bin = (_nn_pred == y_test).astype(int)
                _nn_score_roc = _a2[idx, _nn_pred]

                with open(_omega_p, "rb") as _fh:
                    _omega_data = _pk.load(_fh)
                from patas_module.concrete.TensorTO import TensorArrayTO as _TATO
                from patas_module.NN.PTAStemplate import PTAS as _PTASClass
                _omega_thetas = [_TATO(ow.astype(np.float32)) for ow in _omega_data]
                from patas_module.concrete.TrustOpinion import TrustOpinion as _TO
                _ptas = _PTASClass(
                    omega_thetas=_omega_thetas,
                    operator_mapping=None, nn_interface=None,
                    trust_assessment_func=None,
                    structure=[ds.X_test.shape[1], N_HIDDEN, n_classes],
                    use_tensor=True,
                )
                _Ty2 = _ptas.apply_feedforward(_TATO(_tens_cv), tmp=False)
                _b = _Ty2.value[idx, _nn_pred, 0]
                _d = _Ty2.value[idx, _nn_pred, 1]
                _u = _Ty2.value[idx, _nn_pred, 2]
                _b_n, _d_n, _u_n = bdu_with_weights(_b, _d, _u, _nn_score_roc)
                _pi_roc = _b_n + _u_n / 2.0

                _roc_scores = {"nn": _nn_score_roc, "comb_pi": _pi_roc}
                for src in _THR_SOURCES:
                    if src not in _roc_scores:
                        continue
                    _sc = _roc_scores[src]
                    if not np.isfinite(_sc).any():
                        continue
                    _fpr, _tpr, _ = _sk_roc(_correct_bin, _sc)
                    _auc_val = _sk_auc(_fpr, _tpr)
                    ax_roc.plot(_fpr, _tpr,
                                color=_THR_COLORS[src], lw=2, ls=_THR_LINESTYLES[src],
                                label=f"{_THR_LABELS[src]}  AUC={_auc_val:.3f}")
                ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.set_title(f"ROC — {panel_title}", fontsize=12)
                ax_roc.legend(fontsize=11, loc="lower right")
                ax_roc.grid(linestyle=":", alpha=0.4)
                ax_roc.set_xlim(-0.02, 1.02)
                ax_roc.set_ylim(-0.02, 1.02)
            else:
                ax_roc.set_visible(False)

        for pi in range(len(panels), ncols):
            axes_arr[0][pi].set_visible(False)
            axes_arr[1][pi].set_visible(False)

        fig.suptitle(ax_title, fontsize=16)
        fig.tight_layout()
        path = os.path.join(output_dir, "plots", f"coverage_accuracy_{ax_key}.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Mixed-noise evaluation  (per-sample trust vs. uniform strategies)
# ---------------------------------------------------------------------------

def eval_mixed_noise(ds, n_classes: int, output_dir: str,
                     noise_fractions: list = None):
    """Evaluate per-sample trust under a mixed clean/noisy test set.

    For each trained sigma model and each noise fraction f:
      - f of test samples receive Gaussian noise at sigma_rel
      - Clean samples → trusted opinion (1, 0, 0)
      - Noisy samples → feature_noise_to_trust(sigma_rel)

    Three strategies are compared:
      per-sample : correct per-sample trust (the ideal)
      uniform-cal: all samples get calibrated trust (sigma_rel)
      uniform-tru: all samples get fully trusted opinion
      nn-softmax : NN softmax score (no trust)

    Produces (one set per sigma with a trained model):
      plots/mixed_noise_{sigma:.2f}_curves.pdf   — coverage-accuracy per fraction
      plots/mixed_noise_{sigma:.2f}_separation.pdf — mean π split by noise status
    """
    import pickle
    from noise_utils import add_feature_noise

    try:
        from patas_module.concrete.TensorTO import TensorArrayTO, normalize_tensor
        from patas_module.NN.PTAStemplate import PTAS as PTASClass
        from patas_module.NN.primaryNN import relu, softmax
    except ImportError:
        print("  [skip] skipping mixed-noise evaluation")
        return

    if noise_fractions is None:
        noise_fractions = [0.0, 0.25, 0.50, 0.75, 1.0]

    y_test  = ds.y_test
    n_test, dim = ds.X_test.shape
    idx     = np.arange(n_test)
    a_base  = 1.0 / n_classes
    taus    = np.linspace(0.0, 1.0, 300)
    SEED    = 42

    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    # Style for each strategy
    _S = {
        "per":  ("#8e44ad", "Per-sample trust",    "-",  "o"),
        "cal":  ("#c0392b", "Uniform calibrated",  "--", "s"),
        "tru":  ("#2b7bba", "Uniform trusted",     ":",  "^"),
        "nn":   ("#555555", "NN softmax",           "-.", "D"),
    }

    for sigma in FEATURE_SIGMAS:
        if sigma == 0.0:
            continue

        omega_p = os.path.join(output_dir, "data",
                               f"fn_{sigma:.2f}_ptas-cal-fb_r0", "omega_thetas.pkl")
        nn_p    = os.path.join(output_dir, "data",
                               f"fn_{sigma:.2f}_ptas-cal-fb_r0", "nn_weights.pkl")
        if not (os.path.exists(omega_p) and os.path.exists(nn_p)):
            continue

        with open(omega_p, "rb") as fh:
            omega_data = pickle.load(fh)
        omega_thetas = [TensorArrayTO(ow.astype(np.float32)) for ow in omega_data]
        ptas = PTASClass(
            omega_thetas=omega_thetas,
            operator_mapping=None,
            nn_interface=None,
            trust_assessment_func=None,
            structure=[dim, N_HIDDEN, n_classes],
            use_tensor=True,
        )

        op_cal = feature_noise_to_trust(sigma)
        cal_vec = np.array([op_cal.t, op_cal.d, op_cal.u], dtype=np.float32)
        tru_vec = np.array([1.0, 0.0, 0.0],                dtype=np.float32)

        # Pre-generate a fully noisy X_test for this sigma
        rng_noise = np.random.default_rng(SEED)
        X_noisy = add_feature_noise(ds.X_test, sigma, rng=rng_noise).astype(np.float32)

        def _pi(Ty2_value, preds):
            return Ty2_value[idx, preds, 0] + a_base * Ty2_value[idx, preds, 2]

        def _ff(tens):
            return ptas.apply_feedforward(TensorArrayTO(tens), tmp=False)

        def _sweep(scores, preds):
            covs, accs = np.empty(len(taus)), np.full(len(taus), np.nan)
            for i, tau in enumerate(taus):
                mask  = scores >= tau
                n_cov = int(mask.sum())
                covs[i] = 100.0 * n_cov / n_test
                if n_cov > 0:
                    accs[i] = 100.0 * float(
                        (preds[mask] == y_test[mask]).mean())
            return covs, accs

        # Collect results per fraction
        frac_results = []   # list of (frac, noisy_mask, curves_dict, sep_dict)

        for frac in noise_fractions:
            rng_mask = np.random.default_rng(SEED + int(frac * 1000))
            noisy_mask = rng_mask.random(n_test) < frac

            # Mixed X: noisy samples from X_noisy, clean from X_test
            X_mix = ds.X_test.copy().astype(np.float32)
            X_mix[noisy_mask] = X_noisy[noisy_mask]

            with open(nn_p, "rb") as fh:
                w = pickle.load(fh)
            a1 = relu(X_mix @ w["W1"] + w["b1"])
            a2 = softmax(a1 @ w["W2"] + w["b2"])
            preds = a2.argmax(axis=1)
            nn_score = a2[idx, preds]

            # Per-sample trust tensor
            tens_per = np.tile(tru_vec, (n_test, dim, 1))
            tens_per[noisy_mask] = cal_vec

            # Uniform calibrated
            tens_cal = np.broadcast_to(cal_vec, (n_test, dim, 3)).copy()

            # Uniform trusted
            tens_tru = np.broadcast_to(tru_vec, (n_test, dim, 3)).copy()

            pi_per = _pi(_ff(tens_per).value, preds)
            pi_cal = _pi(_ff(tens_cal).value, preds)
            pi_tru = _pi(_ff(tens_tru).value, preds)

            curves = {
                "per": _sweep(pi_per, preds),
                "cal": _sweep(pi_cal, preds),
                "tru": _sweep(pi_tru, preds),
                "nn":  _sweep(nn_score, preds),
            }

            # Separation: mean π for clean vs noisy subsets (for strategies that differ)
            n_clean = int((~noisy_mask).sum())
            n_noisy = int(noisy_mask.sum())
            sep = {
                "per_clean": float(pi_per[~noisy_mask].mean()) if n_clean > 0 else np.nan,
                "per_noisy": float(pi_per[noisy_mask].mean())  if n_noisy > 0 else np.nan,
                "cal_clean": float(pi_cal[~noisy_mask].mean()) if n_clean > 0 else np.nan,
                "cal_noisy": float(pi_cal[noisy_mask].mean())  if n_noisy > 0 else np.nan,
                "tru_clean": float(pi_tru[~noisy_mask].mean()) if n_clean > 0 else np.nan,
                "tru_noisy": float(pi_tru[noisy_mask].mean())  if n_noisy > 0 else np.nan,
                "nn_clean":  float(nn_score[~noisy_mask].mean()) if n_clean > 0 else np.nan,
                "nn_noisy":  float(nn_score[noisy_mask].mean())  if n_noisy > 0 else np.nan,
            }
            frac_results.append((frac, noisy_mask, curves, sep))

        # ---- Figure 1: coverage-accuracy curves ----
        n_fracs = len(noise_fractions)
        ncols   = min(n_fracs, 3)
        nrows   = (n_fracs + ncols - 1) // ncols

        fig1, axes1 = plt.subplots(nrows, ncols,
                                   figsize=(5.5 * ncols, 4.5 * nrows),
                                   squeeze=False)
        for fi, (frac, _, curves, _) in enumerate(frac_results):
            ax = axes1[fi // ncols][fi % ncols]
            for key, (color, label, ls, _) in _S.items():
                covs, accs = curves[key]
                valid = ~np.isnan(accs)
                if valid.any():
                    ax.plot(covs[valid], accs[valid],
                            color=color, lw=2, ls=ls, label=label)
            ax.set_title(f"{int(frac*100):d}% noisy", fontsize=10)
            ax.set_xlabel("Coverage (%)")
            ax.set_ylabel("Accuracy on covered (%)")
            ax.set_xlim(-2, 105)
            ax.legend(fontsize=12, loc="lower right")
            ax.grid(linestyle=":", alpha=0.4)

        for fi in range(n_fracs, nrows * ncols):
            axes1[fi // ncols][fi % ncols].set_visible(False)

        fig1.suptitle(
            f"Mixed noise  σ={sigma:.2f}: coverage-accuracy per strategy",
            fontsize=12)
        fig1.tight_layout()
        path1 = os.path.join(output_dir, "plots",
                             f"mixed_noise_{sigma:.2f}_curves.pdf")
        fig1.savefig(path1, bbox_inches="tight")
        fig1.savefig(path1.replace(".pdf", ".png"), bbox_inches="tight")
        plt.close(fig1)
        print(f"  Saved {path1}")

        # ---- Figure 2: mean π separation (clean vs noisy) ----
        fig2, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

        fracs_x = [r[0] for r in frac_results]
        sep_styles = {
            "per":  ("#8e44ad", "Per-sample"),
            "cal":  ("#c0392b", "Uniform cal."),
            "tru":  ("#2b7bba", "Uniform tru."),
            "nn":   ("#555555", "NN softmax"),
        }
        for key, (color, lbl) in sep_styles.items():
            cleans = [r[3][f"{key}_clean"] for r in frac_results]
            noisys = [r[3][f"{key}_noisy"] for r in frac_results]
            ax_top.plot(fracs_x, cleans, color=color, lw=2, marker="o",
                        linestyle="-",  label=f"{lbl} (clean)")
            ax_top.plot(fracs_x, noisys, color=color, lw=2, marker="s",
                        linestyle="--", label=f"{lbl} (noisy)")

            gaps = [c - n if not (np.isnan(c) or np.isnan(n)) else np.nan
                    for c, n in zip(cleans, noisys)]
            ax_bot.plot(fracs_x, gaps, color=color, lw=2, marker="o",
                        label=lbl)

        ax_top.set_ylabel(r"Mean $\pi = b + u/K$")
        ax_top.set_title(f"σ={sigma:.2f}: mean projected probability by noise status")
        ax_top.legend(fontsize=8, ncol=2)
        ax_top.grid(linestyle=":", alpha=0.4)

        ax_bot.axhline(0, color="black", lw=0.8)
        ax_bot.set_xlabel("Fraction of noisy samples")
        ax_bot.set_ylabel(r"$\Delta\pi$ (clean $-$ noisy)")
        ax_bot.set_title("Trust separation gap")
        ax_bot.legend(fontsize=12)
        ax_bot.grid(linestyle=":", alpha=0.4)

        fig2.tight_layout()
        path2 = os.path.join(output_dir, "plots",
                             f"mixed_noise_{sigma:.2f}_separation.pdf")
        fig2.savefig(path2, bbox_inches="tight")
        fig2.savefig(path2.replace(".pdf", ".png"), bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved {path2}")


# ---------------------------------------------------------------------------
# Generate all plots and tables
# ---------------------------------------------------------------------------

def generate_outputs(all_results: dict, output_dir: str, ds=None, n_classes: int = 3, run_latency: bool = False, per_sample: str = "uniform"):
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)

    print("\nGenerating plots ...")
    plot_trust_mapping(output_dir)
    plot_feature_noise(all_results, output_dir)
    plot_label_noise(all_results, output_dir)
    plot_combined(all_results, output_dir)
    plot_improvement(all_results, output_dir)
    plot_learning_curves(all_results, output_dir)

    print("\nGenerating LaTeX tables ...")
    table_trust_mapping(output_dir)
    table_feature_noise(all_results, output_dir)
    table_label_noise(all_results, output_dir)
    table_combined(all_results, output_dir)

    if ds is not None:
        ds_copy = copy.deepcopy(ds)
        print("\nGenerating PTAS effectiveness analysis ...")
        ptas_effectiveness_analysis(ds, n_classes, output_dir, per_sample=per_sample)

        print("\nGenerating coverage-accuracy curves ...")
        plot_coverage_accuracy(ds, n_classes, output_dir)

        print("\nGenerating mixed-noise evaluation ...")
        # eval_mixed_noise(ds, n_classes, output_dir)

        print("\nGenerating Algorithm 5 calibration trust analysis ...")
        calibration_trust_analysis(ds_copy, output_dir)

        if run_latency:
            print("\nGenerating latency benchmark ...")
            latency_analysis(ds, output_dir)

# ---------------------------------------------------------------------------
# Threshold-eval plot helper
# ---------------------------------------------------------------------------

def _plot_threshold_eval_results(plot_data: dict, output_dir: str,
                                  tau: float, filter_expr: str):
    """Generate coverage and accuracy-on-covered plots for all three opinion sources."""
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    axes_spec = [
        ("feature",  "Feature noise",  r"Relative noise $\sigma_{rel}$"),
        ("label",    "Label noise",    r"Label flip rate $p$"),
        ("combined", "Combined noise", "Condition"),
    ]

    for section_key, section_name, x_label in axes_spec:
        data = plot_data.get(section_key, [])
        if not data:
            continue

        is_combined = (section_key == "combined")

        fig, (ax_cov, ax_acc) = plt.subplots(2, 1, figsize=(7, 6),
                                              sharex=(not is_combined))

        if is_combined:
            xs      = np.arange(len(data))
            n_src   = len(_THR_SOURCES)
            w       = 0.18
            offsets = [(i - (n_src - 1) / 2) * w for i in range(n_src)]
            for src, off in zip(_THR_SOURCES, offsets):
                mcovs = np.array([d[src][0] for d in data])
                maccs = np.array([d[src][2] for d in data])
                ax_cov.bar(xs + off, mcovs, w,
                           color=_THR_COLORS[src], label=_THR_LABELS[src])
                ax_acc.bar(xs + off, maccs, w, color=_THR_COLORS[src])
            tick_labels = [d["label"] for d in data]
            for ax in (ax_cov, ax_acc):
                ax.set_xticks(xs)
                ax.set_xticklabels(tick_labels, fontsize=8)
        else:
            x_vals = [d["x"] for d in data]
            for src in _THR_SOURCES:
                mcovs = np.array([d[src][0] for d in data])
                scovs = np.array([d[src][1] for d in data])
                maccs = np.array([d[src][2] for d in data])
                saccs = np.array([d[src][3] for d in data])
                ax_cov.plot(x_vals, mcovs, color=_THR_COLORS[src],
                            marker=_THR_MARKERS[src], lw=2, ms=7,
                            label=_THR_LABELS[src])
                ax_cov.fill_between(x_vals, mcovs - scovs, mcovs + scovs,
                                    alpha=0.15, color=_THR_COLORS[src])
                ax_acc.plot(x_vals, maccs, color=_THR_COLORS[src],
                            marker=_THR_MARKERS[src], lw=2, ms=7)
                ax_acc.fill_between(x_vals, maccs - saccs, maccs + saccs,
                                    alpha=0.15, color=_THR_COLORS[src])
            ax_acc.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        ax_cov.set_ylabel("Coverage (%)")
        ax_cov.set_ylim(0, 108)
        ax_acc.set_ylabel("Accuracy on covered (%)")
        ax_acc.set_xlabel(x_label)
        ax_cov.legend(fontsize=8, ncol=2)
        ax_cov.set_title(
            f"{section_name}  —  filter: {filter_expr!r}  (τ={tau:.3f})",
            fontsize=10,
        )

        fig.tight_layout()
        path = os.path.join(output_dir, "plots",
                            f"threshold_{section_key}_{tau:.2f}.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Belief-threshold selective prediction evaluation
# ---------------------------------------------------------------------------

def eval_belief_threshold(ds, n_classes: int, output_dir: str, threshold: float,
                           filter_expr: str = "b >= tau"):
    """Selective prediction across all three noise regimes.

    For each PaTAS run, three opinion sources are evaluated, all filtered by
    the same `filter_expr` (a Python boolean expression in b, d, u, tau):

      Source 1 – aggregated : trust propagated feedforward through omega_thetas network
      Source 2 – output     : ABF-averaged opinion of last-layer weights for predicted class
      Source 3 – weighted   : per-class propagated opinions blended by NN softmax scores

    NN softmax score ≥ τ is always shown as a baseline.
    Four LaTeX tables are produced, one per source.
    """
    import pickle

    try:
        from patas_module.concrete.TensorTO import (
            TensorArrayTO, av_fuse_gen, normalize_tensor,
        )
        from patas_module.NN.PTAStemplate import PTAS as PTASClass
        from patas_module.NN.primaryNN import relu, softmax
    except ImportError:
        print("  [skip] patas_module not importable — skipping threshold eval")
        return

    tau = threshold

    def _apply_filter(b, d, u):
        try:
            return eval(filter_expr,                        # noqa: S307
                        {"__builtins__": {}},
                        {"b": b, "d": d, "u": u, "tau": tau, "np": np})
        except Exception as exc:
            raise ValueError(f"Invalid filter expression {filter_expr!r}: {exc}") from exc

    y_test  = ds.y_test
    n_test, dim = ds.X_test.shape
    idx     = np.arange(n_test)

    print(f"\nThreshold evaluation  τ = {tau:.3f}   filter: {filter_expr!r}")
    print(f"  {'cond':<22}  {'run':>3}  "
          f"{'cov[agg]':>9}  {'acc[agg]':>9}  "
          f"{'cov[out]':>9}  {'acc[out]':>9}  "
          f"{'cov[wt]':>8}  {'acc[wt]':>8}  "
          f"{'cov[nn]':>8}  {'acc[nn]':>8}  {'all%':>7}")
    print("  " + "-" * 118)

    rows_aggr: list[list] = []
    rows_out:  list[list] = []
    rows_wt:   list[list] = []
    rows_nn:   list[list] = []

    NCOLS = 4

    # plot_data collects per-condition aggregated results for visual plots
    plot_data: dict[str, list] = {"feature": [], "label": [], "combined": []}
    current_section: list[str] = ["feature"]  # mutable ref; updated per section

    conditions: list = []
    conditions.append(('section', r'Feature noise ($T_x$ calibrated to $\sigma_{rel}$)', NCOLS, 'feature'))
    for sigma in FEATURE_SIGMAS:
        op = feature_noise_to_trust(sigma)
        conditions.append((
            f"$\\sigma_{{rel}}={sigma:.2f}$",
            (lambda s: (lambda run: f"fn_{s:.2f}_ptas-cal-fb_r{run}"))(sigma),
            float(op.t), float(op.d), float(op.u),
            sigma,   # x_val for plot
        ))

    conditions.append(('section', r'Label noise ($T_x$ fully trusted)', NCOLS, 'label'))
    for flip in LABEL_FLIPS:
        conditions.append((
            f"$p={flip:.2f}$",
            (lambda f: (lambda run: f"ln_{f:.2f}_ptas-cal-fb_r{run}"))(flip),
            1.0, 0.0, 0.0,
            flip,    # x_val for plot
        ))

    conditions.append(('section', r'Combined noise ($T_x$ calibrated to $\sigma_{rel}$)', NCOLS, 'combined'))
    for i, (sigma, flip) in enumerate(COMBINED):
        op = feature_noise_to_trust(sigma)
        conditions.append((
            f"$\\sigma={sigma:.2f},\\,p={flip:.2f}$",
            (lambda s, f: (lambda run: f"comb_{s:.2f}_{f:.2f}_ptas-cal-fb_r{run}"))(sigma, flip),
            float(op.t), float(op.d), float(op.u),
            i,       # x_val for plot (index)
        ))

    for cond in conditions:
        if cond[0] == 'section':
            _, sec_title, ncols, sec_key = cond
            current_section[0] = sec_key
            mc = f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{sec_title}}}}}"
            for rows in [rows_aggr, rows_out, rows_wt, rows_nn]:
                if rows:
                    rows.append(['\\midrule'])
                rows.append([mc])
            continue

        noise_label, run_dir_fn, bv, dv, uv, x_val = cond
        per_run: list[tuple] = []

        for run in range(N_RUNS):
            base    = os.path.join(output_dir, "data", run_dir_fn(run))
            omega_p = os.path.join(base, "omega_thetas.pkl")
            nn_p    = os.path.join(base, "nn_weights.pkl")
            if not (os.path.exists(omega_p) and os.path.exists(nn_p)):
                continue

            with open(nn_p, "rb") as fh:
                w = pickle.load(fh)
            a1 = relu(ds.X_test @ w["W1"] + w["b1"])
            a2 = softmax(a1 @ w["W2"] + w["b2"])
            nn_preds = a2.argmax(axis=1)

            with open(omega_p, "rb") as fh:
                omega_data = pickle.load(fh)
            omega_thetas = [TensorArrayTO(ow.astype(np.float32)) for ow in omega_data]
            ptas = PTASClass(
                omega_thetas=omega_thetas,
                operator_mapping=None,
                nn_interface=None,
                trust_assessment_func=None,
                structure=[dim, N_HIDDEN, n_classes],
                use_tensor=True,
            )
            tens = np.empty((n_test, dim, 3), dtype=np.float32)
            tens[..., 0] = bv; tens[..., 1] = dv; tens[..., 2] = uv
            Ty2 = ptas.apply_feedforward(TensorArrayTO(tens), tmp=False)  # (n,cls,3)

            # Source 1 – aggregated: propagated opinion for predicted class
            b_aggr = Ty2.value[idx, nn_preds, 0]
            d_aggr = Ty2.value[idx, nn_preds, 1]
            u_aggr = Ty2.value[idx, nn_preds, 2]

            b_aggr, d_aggr, u_aggr = bdu_with_weights(b_aggr, d_aggr, u_aggr)
            # # Source 2 – output-layer weight opinions (ABF across weight rows)
            # last_w = omega_data[-1].astype(np.float32)   # (hidden+1, output, 3)
            # out_op = av_fuse_gen(last_w, axis=0)          # (output, 3)
            # b_out  = out_op[nn_preds, 0]
            # d_out  = out_op[nn_preds, 1]
            # u_out  = out_op[nn_preds, 2]

            # b_out, d_out, u_out = bdu_with_weights(b_out, d_out, u_out)

            # # Source 3 – softmax-weighted blend of all output-class opinions
            # wt_op = np.einsum('ij,ijk->ik', a2, Ty2.value)  # (n, 3)
            # wt_op = normalize_tensor(wt_op)
            # b_wt  = wt_op[:, 0]
            # d_wt  = wt_op[:, 1]
            # u_wt  = wt_op[:, 2]

            nn_score = a2[idx, nn_preds]

            def _stats(mask):
                n_cov = int(mask.sum())
                cov   = 100.0 * n_cov / n_test
                acc   = (100.0 * float((nn_preds[mask] == y_test[mask]).mean())
                         if n_cov > 0 else float("nan"))
                return cov, acc

            cov_aggr, acc_aggr = _stats(_apply_filter(b_aggr, d_aggr, u_aggr))
            # cov_out,  acc_out  = _stats(_apply_filter(b_out,  d_out,  u_out))
            # cov_wt,   acc_wt   = _stats(_apply_filter(b_wt,   d_wt,   u_wt))
            cov_nn,   acc_nn   = _stats(nn_score >= tau)
            acc_all = 100.0 * float((nn_preds == y_test).mean())

            print(f"  {noise_label:<22}  {run:>3}  "
                  f"{cov_aggr:>9.1f}  {acc_aggr:>9.2f}  "
                #   f"{cov_out:>9.1f}  {acc_out:>9.2f}  "
                #   f"{cov_wt:>8.1f}  {acc_wt:>8.2f}  "
                  f"{cov_nn:>8.1f}  {acc_nn:>8.2f}  {acc_all:>7.2f}")
            per_run.append((cov_aggr, acc_aggr,
                            # cov_out,  acc_out,
                            # cov_wt,   acc_wt,
                            cov_nn,   acc_nn,
                            acc_all))

        if not per_run:
            continue

        def _agg(col_indices):
            valid = [tuple(r[i] for i in col_indices) for r in per_run
                     if not any(np.isnan(r[i]) for i in col_indices)]
            if not valid:
                return None
            arr = np.array(valid)
            return arr.mean(axis=0), arr.std(axis=0)

        def _append(res, rows):
            if res:
                (m_cov, m_acc, m_all), (s_cov, s_acc, _) = res
                rows.append([
                    noise_label,
                    f"{m_cov:.1f} $\\pm$ {s_cov:.1f}",
                    f"{m_acc:.2f} $\\pm$ {s_acc:.2f}",
                    f"{m_all:.2f}",
                ])

        _append(_agg([0, 1, 8]), rows_aggr)
        _append(_agg([2, 3, 8]), rows_out)
        _append(_agg([4, 5, 8]), rows_wt)
        _append(_agg([6, 7, 8]), rows_nn)

        def _extract(col_indices):
            r = _agg(col_indices)
            if r is None:
                return (float("nan"), 0.0, float("nan"), 0.0)
            (mcov, macc, _), (scov, sacc, _) = r
            return (mcov, scov, macc, sacc)

        plot_data[current_section[0]].append({
            "x":     x_val,
            "label": noise_label,
            "agg":  _extract([0, 1, 8]),
            "out":   _extract([2, 3, 8]),
            "wt":    _extract([4, 5, 8]),
            "nn":    _extract([6, 7, 8]),
        })

    def _clean(rows):
        out = []
        for i, r in enumerate(rows):
            if r[0] in ('\\midrule',) or r[0].startswith('\\multicolumn'):
                rest = rows[i + 1:]
                has_data = any(
                    not (rr[0] in ('\\midrule',) or rr[0].startswith('\\multicolumn'))
                    for rr in rest
                )
                if has_data:
                    out.append(r)
            else:
                out.append(r)
        return out

    rows_aggr = _clean(rows_aggr)
    rows_out  = _clean(rows_out)
    rows_wt   = _clean(rows_wt)
    rows_nn   = _clean(rows_nn)

    if not any(
        not (r[0] in ('\\midrule',) or r[0].startswith('\\multicolumn'))
        for rows in [rows_aggr, rows_out, rows_wt, rows_nn] for r in rows
    ):
        print("  No saved models found — run experiments first.")
        return

    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    header = [
        "Noise condition",
        r"Coverage (\%)",
        r"Accuracy on covered (\%)",
        r"Overall accuracy (\%)",
    ]
    filt_latex = (filter_expr
                  .replace(">=", r"$\geq$")
                  .replace("<=", r"$\leq$")
                  .replace("tau", r"$\tau$"))

    table_specs = [
        (
            rows_aggr, f"patas_aggr_{tau:.2f}",
            (f"Selective prediction — \\textbf{{aggregated opinion}}: filter \\texttt{{{filt_latex}}}, "
             r"trust propagated feedforward through the full $\omega$-network. "
             r"Values are mean $\pm$ std over " + str(N_RUNS) + r" runs."),
            f"tab:thresh-aggr-{str(tau).replace('.', '-')}",
        ),
        (
            rows_out, f"patas_out_{tau:.2f}",
            (f"Selective prediction — \\textbf{{output-layer opinion}}: filter \\texttt{{{filt_latex}}}, "
             r"ABF-averaged opinion of last-layer $\omega$-weights for the predicted class. "
             r"Values are mean $\pm$ std over " + str(N_RUNS) + r" runs."),
            f"tab:thresh-out-{str(tau).replace('.', '-')}",
        ),
        (
            rows_wt, f"patas_wt_{tau:.2f}",
            (f"Selective prediction — \\textbf{{weighted opinion}}: filter \\texttt{{{filt_latex}}}, "
             r"propagated opinions for all output classes blended by NN softmax probabilities. "
             r"Values are mean $\pm$ std over " + str(N_RUNS) + r" runs."),
            f"tab:thresh-wt-{str(tau).replace('.', '-')}",
        ),
        (
            rows_nn, f"nn_{tau:.2f}",
            (f"Selective prediction — \\textbf{{NN softmax baseline}}: "
             f"softmax score $\\geq \\tau = {tau:.3f}$ for the predicted class. "
             r"Values are mean $\pm$ std over " + str(N_RUNS) + r" runs."),
            f"tab:thresh-nn-{str(tau).replace('.', '-')}",
        ),
    ]
    for rows, slug, caption, label in table_specs:
        if not rows:
            continue
        tex = _booktabs_table(header, rows, caption=caption, label=label)
        tpath = os.path.join(output_dir, "tables", f"threshold_eval_{slug}.tex")
        with open(tpath, "w") as fh:
            fh.write(tex)
        print(f"\n  LaTeX table saved to {tpath}")

    _plot_threshold_eval_results(plot_data, output_dir, tau, filter_expr)


# ---------------------------------------------------------------------------
# Load cached results (for --plots-only)
# ---------------------------------------------------------------------------

def load_cached(output_dir: str) -> dict:
    all_results: dict[str, dict] = {}
    data_root = os.path.join(output_dir, "data")
    if not os.path.isdir(data_root):
        return all_results
    for name in os.listdir(data_root):
        rf = os.path.join(data_root, name, "results.json")
        if os.path.exists(rf):
            with open(rf) as fh:
                all_results[name] = json.load(fh)
    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Noise-robustness evaluation of PaTAS on the 5G dataset"
    )
    ap.add_argument("data_dir", nargs="?", default=None,
                    help="Path to 5G CSVs (omit to use synthetic data)")
    ap.add_argument("--output", default="results_noise",
                    help="Output directory (default: results_noise/)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run experiments even if cached results exist")
    ap.add_argument("--plots-only", action="store_true",
                    help="Skip training; regenerate plots from cached results")
    ap.add_argument("--threshold", type=float, default=None,
                    metavar="TAU",
                    help="Belief threshold τ used in the default filter 'b >= tau' "
                         "(requires saved models). Can be combined with --plots-only.")
    ap.add_argument("--filter", default="b >= tau",
                    metavar="EXPR",
                    help="PaTAS selective-prediction filter: a Python boolean expression "
                         "in b, d, u, tau applied to each opinion source. "
                         "Default: 'b >= tau'. Examples: 'b > u', 'b - d >= 0.3', "
                         "'(b + u/2) >= tau'")
    ap.add_argument("--latency", action="store_true",
                    help="Run latency benchmark (inference + training overhead).")
    _trust_grp = ap.add_mutually_exclusive_group()
    _trust_grp.add_argument("--per-sample", action="store_true",
                    help="Effectiveness analysis: noise each test input with σ_i~Uniform(0,σ) "
                         "and assign it the matching trust opinion feature_noise_to_trust(σ_i). "
                         "Mutually exclusive with --trust.")
    _trust_grp.add_argument("--uniform", action="store_true",
                    help="Effectiveness analysis: use a uniform trust distribution (b=0.5, d=0.5, u=0.5) "
                         "for all test inputs, regardless of noise level. "
                         "Mutually exclusive with --per-sample.")
    args = ap.parse_args()
    trust_mode = "per_sample" if args.per_sample else ("uniform" if args.uniform else "fully_trusted")

    if args.plots_only or args.threshold is not None or args.latency:
        # Load dataset (needed for threshold eval and effectiveness plots)
        try:
            _data_dir = args.data_dir
            if _data_dir is None:
                _data_dir = make_synthetic_5g(n_bs=20, n_hours=72,
                                              cells_per_bs=2, seed=0)
            _ds = load_5g_dataset(_data_dir, n_classes=3, test_frac=0.2, seed=0)
            _ds_copy = copy.deepcopy(_ds)  # for threshold eval (may be modified by PTASClass)
            _nc = int(_ds.y_train.max()) + 1
        except Exception:
            _ds, _nc = None, 3

        if args.plots_only:
            print(f"Loading cached results from {args.output} ...")
            all_results = load_cached(args.output)
            if not all_results:
                print("No cached results found. Run without --plots-only first.")
            else:
                generate_outputs(all_results, args.output, ds=_ds, n_classes=_nc, run_latency=args.latency, per_sample=trust_mode)

        if args.threshold is not None:
            if _ds is None:
                print("Dataset unavailable — cannot run threshold evaluation.")
            else:
                eval_belief_threshold(_ds, _nc, args.output, args.threshold,
                                      filter_expr=args.filter)
    else:
        all_results, ds, n_classes = run_all(args.data_dir, args.output,
                                             force=args.force)
        generate_outputs(all_results, args.output, ds=ds, n_classes=n_classes, run_latency=args.latency, per_sample=trust_mode)

    print("\nDone.")