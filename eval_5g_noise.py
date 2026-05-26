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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from data_loader import load_5g_dataset, make_synthetic_5g
from noise_utils import (
    add_feature_noise,
    add_label_noise,
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
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

# Consistent colours across all plots
_COLORS  = {"nn": "#555555", "ptas-cal-fb": "#8e44ad"}
_LABELS  = {"nn": "NN (no trust)", "ptas-cal-fb": "NN"}
_MARKERS = {"nn": "s", "ptas-cal-fb": "P"}


# ---------------------------------------------------------------------------
# Experiment grid definition
# ---------------------------------------------------------------------------

FEATURE_SIGMAS  = [0, 0.1, 0.3, 0.5]
LABEL_FLIPS     = [0, 0.05, 0.15, 0.30]
COMBINED        = [(0.1, 0.05), (0.3, 0.15), (0.5, 0.30)]
CONFIGS         = ["ptas-cal-fb"]

EPOCHS   = 10
BATCH    = 64
LR       = 0.05
EPS_LOW  = 0.05
N_HIDDEN = 32
BASE_PORT = 6550   # incremented per PTAS run to avoid port collisions

# Both configs run N_RUNS independent noise draws; results are averaged.
N_RUNS = 5


@dataclass
class ExpSpec:
    """Specification for a single training run."""
    label: str           # unique identifier  e.g. "fn_0.10_ptas-cal"
    config: str          # one of CONFIGS
    sigma: float         # feature noise level (global σ_rel for ptas-percal = sigma_max)
    flip: float          # label flip rate
    use_ptas: bool
    x_trust: object      # str | TrustOpinion | "percal"
    y_trust: object      # str | TrustOpinion
    port: int = BASE_PORT
    run_seed: int = 0    # RNG seed for noise injection (varied across N_RUNS for ptas-percal)
    # trust_feedback: bool = False  # True for ptas-cal-fb: PTAS sends belief scalar back to NN


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
            # for config in CONFIGS:
            # specs.append(ExpSpec(
            #     label=f"fn_{sigma:.2f}_nn_r{run}",
            #     config="nn", sigma=sigma, flip=0.0,
            #     use_ptas=False, x_trust="trusted", y_trust="trusted",
            #     port=BASE_PORT, run_seed=run,
            # ))
            specs.append(ExpSpec(
                label=f"fn_{sigma:.2f}_ptas-cal-fb_r{run}",
                config="ptas-cal-fb", sigma=sigma, flip=0.0,
                use_ptas=True, x_trust=xt_cal, y_trust="trusted",
                port=_next_port(), run_seed=run, 
                # trust_feedback=True,
            ))

    # ------------------------------------------------------------------
    # (b) Label noise study  (features always clean)
    # ------------------------------------------------------------------
    for flip in LABEL_FLIPS:
        yt_cal = label_noise_to_trust(flip)
        for run in range(N_RUNS):
            # specs.append(ExpSpec(
            #     label=f"ln_{flip:.2f}_nn_r{run}",
            #     config="nn", sigma=0.0, flip=flip,
            #     use_ptas=False, x_trust="trusted", y_trust="trusted",
            #     port=BASE_PORT, run_seed=run,
            # ))
            specs.append(ExpSpec(
                label=f"ln_{flip:.2f}_ptas-cal-fb_r{run}",
                config="ptas-cal-fb", sigma=0.0, flip=flip,
                use_ptas=True, x_trust="trusted", y_trust=yt_cal,
                port=_next_port(), run_seed=run, 
                # trust_feedback=True,
            ))

    # ------------------------------------------------------------------
    # (c) Combined noise study
    # ------------------------------------------------------------------
    for sigma, flip in COMBINED:
        xt_cal = feature_noise_to_trust(sigma)
        yt_cal = label_noise_to_trust(flip)
        for run in range(N_RUNS):
            # specs.append(ExpSpec(
            #     label=f"comb_{sigma:.2f}_{flip:.2f}_nn_r{run}",
            #     config="nn", sigma=sigma, flip=flip,
            #     use_ptas=False, x_trust="trusted", y_trust="trusted",
            #     port=BASE_PORT, run_seed=run,
            # ))
            specs.append(ExpSpec(
                label=f"comb_{sigma:.2f}_{flip:.2f}_ptas-cal-fb_r{run}",
                config="ptas-cal-fb", sigma=sigma, flip=flip,
                use_ptas=True, x_trust=xt_cal, y_trust=yt_cal,
                port=_next_port(), run_seed=run, 
                # trust_feedback=True,
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
            data_dir=None,          # data already loaded
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
            # trust_feedback=spec.trust_feedback,
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
    for cfg in CONFIGS:
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
                     x_vals=FEATURE_SIGMAS,
                     xlabel=r"Relative feature noise $\sigma_{rel}$",
                     xlim=(-0.02, 0.55))
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
                     x_vals=LABEL_FLIPS,
                     xlabel=r"Label flip rate $p$",
                     xlim=(-0.01, 0.33))
    ax.set_title("NN: label mislabelling")
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
    for offset, cfg in zip(offsets, CONFIGS):
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
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8)

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
    epochs_x = list(range(1, EPOCHS + 1))

    for ax, (prefix, title) in zip(axes.flat, selected):
        for cfg in CONFIGS:
            mean_c, std_c = _epoch_curve_multi(all_results, f"{prefix}{cfg}")
            if mean_c:
                ax.plot(epochs_x, mean_c, color=_COLORS[cfg], lw=2, label=_LABELS[cfg])
                lo = np.array(mean_c) - np.array(std_c)
                hi = np.array(mean_c) + np.array(std_c)
                ax.fill_between(epochs_x, lo, hi, alpha=0.18, color=_COLORS[cfg])
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test accuracy (%)")
        ax.legend(fontsize=8)
        ax.set_xlim(1, EPOCHS)

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

def ptas_effectiveness_analysis(ds, n_classes: int, output_dir: str):
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

    Produces:
      plots/ptas_effectiveness.pdf   — 2-panel line plot
      tables/ptas_effectiveness.tex  — summary table
    """
    import pickle

    try:
        from patas_module.concrete.TensorTO import TensorArrayTO
    except ImportError:
        print("  [skip] TensorArrayTO not importable — skipping effectiveness analysis")
        return

    from patas_module.NN.PTAStemplate import PTAS as PTASClass
    from patas_module.concrete.TrustOpinion import TrustOpinion
    from patas_module.NN.primaryNN import relu, softmax

    y_test_int = ds.y_test
    n_test, dim = ds.X_test.shape
    a_base = 1.0 / n_classes   # kept for reference; metric is now belief b only

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _nn_predictions(nn_pkl_path: str) -> Optional[np.ndarray]:
        if not os.path.exists(nn_pkl_path):
            return None
        with open(nn_pkl_path, "rb") as fh:
            w = pickle.load(fh)
        a1 = relu(ds.X_test @ w["W1"] + w["b1"])
        a2 = softmax(a1 @ w["W2"] + w["b2"])
        return a2.argmax(axis=1)

    def _mean_split(arr, mask, n_c, n_w):
        c = float(arr[mask].mean())  if n_c > 0 else float("nan")
        w = float(arr[~mask].mean()) if n_w > 0 else float("nan")
        return c, w

    def _ptas_opinion(omega_pkl_path: str, trust_op,
                      nn_preds: np.ndarray):
        """Return (b, d, u) arrays in the NN's predicted class. Shape each: (n_test,)"""
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

        bv, dv, uv = float(trust_op.t), float(trust_op.d), float(trust_op.u)
        tens = np.empty((n_test, dim, 3), dtype=np.float32)
        tens[..., 0] = bv
        tens[..., 1] = dv
        tens[..., 2] = uv
        Ty2 = ptas.apply_feedforward(TensorArrayTO(tens), tmp=False)  # (n_test, n_classes, 3)

        idx = np.arange(n_test)
        return (Ty2.value[idx, nn_preds, 0],   # b
                Ty2.value[idx, nn_preds, 1],   # d
                Ty2.value[idx, nn_preds, 2])   # u

    # ------------------------------------------------------------------
    # Collect metrics — feature noise axis
    # ------------------------------------------------------------------

    fn_records: dict[float, tuple] = {}   # sigma -> (b_c, b_w, d_c, d_w, u_c, u_w)
    fn_tex_rows: list[list] = []

    for sigma in FEATURE_SIGMAS:
        base     = os.path.join(output_dir, "data")
        omega_p  = os.path.join(base, f"fn_{sigma:.2f}_ptas-cal-fb_r0", "omega_thetas.pkl")
        nn_p     = os.path.join(base, f"fn_{sigma:.2f}_ptas-cal-fb_r0", "nn_weights.pkl")
        nn_preds = _nn_predictions(nn_p)
        if nn_preds is None:
            continue
        correct   = (nn_preds == y_test_int)
        n_correct = int(correct.sum())
        n_wrong   = int((~correct).sum())
        opinion = _ptas_opinion(omega_p, feature_noise_to_trust(sigma), nn_preds)
        if opinion is None:
            continue
        b_arr, d_arr, u_arr = opinion
        b_c, b_w = _mean_split(b_arr, correct, n_correct, n_wrong)
        d_c, d_w = _mean_split(d_arr, correct, n_correct, n_wrong)
        u_c, u_w = _mean_split(u_arr, correct, n_correct, n_wrong)
        fn_records[sigma] = (b_c, b_w, d_c, d_w, u_c, u_w)
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
    ln_records: dict[float, tuple] = {}   # flip -> (b_c, b_w, d_c, d_w, u_c, u_w)
    ln_tex_rows: list[list] = []

    for flip in LABEL_FLIPS:
        base     = os.path.join(output_dir, "data")
        omega_p  = os.path.join(base, f"ln_{flip:.2f}_ptas-cal-fb_r0", "omega_thetas.pkl")
        nn_p     = os.path.join(base, f"ln_{flip:.2f}_ptas-cal-fb_r0", "nn_weights.pkl")
        nn_preds = _nn_predictions(nn_p)
        if nn_preds is None:
            continue
        correct   = (nn_preds == y_test_int)
        n_correct = int(correct.sum())
        n_wrong   = int((~correct).sum())
        opinion = _ptas_opinion(omega_p, trusted_op, nn_preds)
        if opinion is None:
            continue
        b_arr, d_arr, u_arr = opinion
        b_c, b_w = _mean_split(b_arr, correct, n_correct, n_wrong)
        d_c, d_w = _mean_split(d_arr, correct, n_correct, n_wrong)
        u_c, u_w = _mean_split(u_arr, correct, n_correct, n_wrong)
        ln_records[flip] = (b_c, b_w, d_c, d_w, u_c, u_w)
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

    cb_records: dict[tuple, tuple] = {}   # (sigma,flip) -> (b_c,b_w,d_c,d_w,u_c,u_w)
    cb_tex_rows: list[list] = []

    for sigma, flip in COMBINED:
        base     = os.path.join(output_dir, "data")
        omega_p  = os.path.join(base, f"comb_{sigma:.2f}_{flip:.2f}_ptas-cal-fb_r0",
                                "omega_thetas.pkl")
        nn_p     = os.path.join(base, f"comb_{sigma:.2f}_{flip:.2f}_ptas-cal-fb_r0",
                                "nn_weights.pkl")
        nn_preds = _nn_predictions(nn_p)
        if nn_preds is None:
            continue
        correct   = (nn_preds == y_test_int)
        n_correct = int(correct.sum())
        n_wrong   = int((~correct).sum())
        opinion = _ptas_opinion(omega_p, feature_noise_to_trust(sigma), nn_preds)
        if opinion is None:
            continue
        b_arr, d_arr, u_arr = opinion
        b_c, b_w = _mean_split(b_arr, correct, n_correct, n_wrong)
        d_c, d_w = _mean_split(d_arr, correct, n_correct, n_wrong)
        u_c, u_w = _mean_split(u_arr, correct, n_correct, n_wrong)
        cb_records[(sigma, flip)] = (b_c, b_w, d_c, d_w, u_c, u_w)
        b_gap = f"{b_c - b_w:+.4f}" if not (np.isnan(b_c) or np.isnan(b_w)) else "—"
        d_gap = f"{d_c - d_w:+.4f}" if not (np.isnan(d_c) or np.isnan(d_w)) else "—"
        u_gap = f"{u_c - u_w:+.4f}" if not (np.isnan(u_c) or np.isnan(u_w)) else "—"
        cb_tex_rows.append([
            f"({sigma:.2f}, {flip:.2f})", str(n_correct), str(n_wrong),
            f"{b_c:.4f}", f"{b_w:.4f}", b_gap,
            f"{d_c:.4f}", f"{d_w:.4f}", d_gap,
            f"{u_c:.4f}", f"{u_w:.4f}", u_gap,
        ])

    if not fn_records and not ln_records and not cb_records:
        print("  [skip] No ptas-cal-fb runs found — skipping effectiveness analysis")
        return

    # ------------------------------------------------------------------
    # Plot: 3-column grid (feature / label / combined noise)
    # ------------------------------------------------------------------

    _C = {"b": "#2b7bba", "d": "#c0392b", "u": "#e07b39"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 7),
                             gridspec_kw={"height_ratios": [2, 1]})

    # Panel specs: (ax_top, ax_bot, records, x_vals, tick_labels, xlabel, title)
    cb_x     = list(range(len(COMBINED)))
    cb_ticks = [f"$\\sigma$={s:.2f}\n$p$={p:.2f}" for s, p in COMBINED]
    panels = [
        (axes[0, 0], axes[1, 0],
         {v: fn_records[v] for v in sorted(FEATURE_SIGMAS) if v in fn_records},
         sorted(FEATURE_SIGMAS), None,
         r"Feature noise $\sigma_{rel}$",
         r"Feature noise ($\sigma_{rel}$-calibrated $T_x$)"),
        (axes[0, 1], axes[1, 1],
         {v: ln_records[v] for v in sorted(LABEL_FLIPS) if v in ln_records},
         sorted(LABEL_FLIPS), None,
         r"Label flip rate $p$",
         r"Label noise (trusted $T_x$)"),
        (axes[0, 2], axes[1, 2],
         {i: cb_records[k] for i, k in enumerate(COMBINED) if k in cb_records},
         cb_x, cb_ticks,
         "Condition",
         "Combined noise ($T_x$ calibrated)"),
    ]

    nan6 = (float("nan"),) * 6
    first_col = True
    for ax_top, ax_bot, records, x_vals, tick_labels, xlabel, title in panels:
        b_c = [records.get(v, nan6)[0] for v in x_vals]
        b_w = [records.get(v, nan6)[1] for v in x_vals]
        d_c = [records.get(v, nan6)[2] for v in x_vals]
        d_w = [records.get(v, nan6)[3] for v in x_vals]
        u_c = [records.get(v, nan6)[4] for v in x_vals]
        u_w = [records.get(v, nan6)[5] for v in x_vals]

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
            ax_top.set_ylabel("Mean opinion mass in NN's predicted class")
        ax_top.set_title(title, fontsize=10)
        ax_top.legend(fontsize=6, ncol=2)
        all_vals = [v for lst in [b_c, b_w, d_c, d_w, u_c, u_w]
                    for v in lst if not np.isnan(v)]
        if all_vals:
            ylo, yhi = min(all_vals), max(all_vals)
            margin = max(0.03, (yhi - ylo) * 0.3)
            ax_top.set_ylim(max(0.0, ylo - margin), min(1.0, yhi + margin))

        xs = np.array(x_vals, dtype=float)
        bw_bar = min(np.diff(xs).min() if len(xs) > 1 else 1.0, 0.04 if tick_labels is None else 0.25) * 0.9
        xlim   = (xs[0] - 2.5 * bw_bar, xs[-1] + 2.5 * bw_bar)
        ax_top.set_xlim(*xlim)
        if tick_labels is not None:
            ax_top.set_xticks(x_vals)
            ax_top.set_xticklabels(tick_labels, fontsize=7)

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
                            f"{g:+.3f}", ha="center", va=va, fontsize=6,
                            color=color)
        ax_bot.axhline(0, color="black", lw=0.8)
        ax_bot.set_xlabel(xlabel)
        if first_col:
            ax_bot.set_ylabel(r"$\Delta$ (correct $-$ wrong)")
        ax_bot.set_xlim(*xlim)
        ax_bot.legend(fontsize=6, ncol=3)
        ax_bot.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
        if tick_labels is not None:
            ax_bot.set_xticks(x_vals)
            ax_bot.set_xticklabels(tick_labels, fontsize=7)
        first_col = False

    fig.suptitle("PaTAS as NN confidence signal: opinion masses in predicted class",
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(output_dir, "plots", "patas_effectiveness.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # ------------------------------------------------------------------
    # Table
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
        label="tab:ptas-effectiveness",
    )
    tpath = os.path.join(output_dir, "tables", "ptas_effectiveness.tex")
    with open(tpath, "w") as fh:
        fh.write(tex)
    print(f"  Saved {tpath}")


# ---------------------------------------------------------------------------
# Generate all plots and tables
# ---------------------------------------------------------------------------

def generate_outputs(all_results: dict, output_dir: str, ds=None, n_classes: int = 3, run_latency: bool = False):
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
        print("\nGenerating PTAS effectiveness analysis ...")
        ptas_effectiveness_analysis(ds, n_classes, output_dir)

        print("\nGenerating Algorithm 5 calibration trust analysis ...")
        calibration_trust_analysis(ds, output_dir)

        if run_latency:
            print("\nGenerating latency benchmark ...")
            latency_analysis(ds, output_dir)

# ---------------------------------------------------------------------------
# Belief-threshold selective prediction evaluation
# ---------------------------------------------------------------------------

def eval_belief_threshold(ds, n_classes: int, output_dir: str, threshold: float):
    """Selective prediction across all three noise regimes.

    Three filters evaluated simultaneously per model:
      Table 1 (PaTAS b≥τ)   : projected probability π = b + u/2 ≥ τ
      Table 2 (NN score≥τh) : NN softmax score ≥ τ_high = τ + (1−τ)/2
      Table 3 (b>u)         : PaTAS belief > disbelief (threshold-free)

    Noise regimes covered: feature noise, label noise, combined.
    Each table has three sections separated by a LaTeX midrule.
    """
    import pickle

    try:
        from patas_module.concrete.TensorTO import TensorArrayTO
        from patas_module.NN.PTAStemplate import PTAS as PTASClass
        from patas_module.NN.primaryNN import relu, softmax
        from patas_module.concrete.TrustOpinion import TrustOpinion
    except ImportError:
        print("  [skip] patas_module not importable — skipping threshold eval")
        return

    tau_lo = threshold
    tau_hi = threshold 

    y_test = ds.y_test
    n_test, dim = ds.X_test.shape

    print(f"\nThreshold evaluation  τ(π) = {tau_lo:.3f}   "
          f"τ_high(NN) = {tau_hi:.3f}   b>u filter")
    print(f"  {'cond':<22}  {'run':>3}  "
          f"{'cov%[b≥τ]':>10}  {'acc%[b≥τ]':>10}  "
          f"{'cov%[nn≥τh]':>12}  {'acc%[nn≥τh]':>12}  "
          f"{'cov%[b>u]':>10}  {'acc%[b>u]':>10}  {'acc_all%':>9}")
    print("  " + "-" * 108)

    rows_lo: list[list] = []
    rows_hi: list[list] = []
    rows_bd: list[list] = []

    # ---- Build the list of noise conditions --------------------------------
    # Each entry is either a section marker or a data condition:
    #   (noise_label, run_dir_fn, bv, dv, uv)
    # Section markers: ('section', title_str, ncols)
    NCOLS = 4   # number of LaTeX table columns

    conditions: list = []

    conditions.append(('section', 'Feature noise ($T_x$ calibrated to $\\sigma_{rel}$)', NCOLS))
    for sigma in FEATURE_SIGMAS:
        op = feature_noise_to_trust(sigma)
        conditions.append((
            f"$\\sigma_{{rel}}={sigma:.2f}$",
            (lambda s: (lambda run: f"fn_{s:.2f}_ptas-cal-fb_r{run}"))(sigma),
            float(op.t), float(op.d), float(op.u),
        ))

    conditions.append(('section', 'Label noise ($T_x$ fully trusted)', NCOLS))
    for flip in LABEL_FLIPS:
        conditions.append((
            f"$p={flip:.2f}$",
            (lambda f: (lambda run: f"ln_{f:.2f}_ptas-cal-fb_r{run}"))(flip),
            1.0, 0.0, 0.0,
        ))

    conditions.append(('section', 'Combined noise ($T_x$ calibrated to $\\sigma_{rel}$)', NCOLS))
    for sigma, flip in COMBINED:
        op = feature_noise_to_trust(sigma)
        conditions.append((
            f"$\\sigma={sigma:.2f},\\,p={flip:.2f}$",
            (lambda s, f: (lambda run: f"comb_{s:.2f}_{f:.2f}_ptas-cal-fb_r{run}"))(sigma, flip),
            float(op.t), float(op.d), float(op.u),
        ))

    # ---- Iterate ---------------------------------------------------------
    for cond in conditions:
        if cond[0] == 'section':
            _, sec_title, ncols = cond
            mc = f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{sec_title}}}}}"
            for rows in [rows_lo, rows_hi, rows_bd]:
                if rows:          # separate sections with a midrule
                    rows.append(['\\midrule'])
                rows.append([mc])
            continue

        noise_label, run_dir_fn, bv, dv, uv = cond

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
            tens[..., 0] = bv
            tens[..., 1] = dv
            tens[..., 2] = uv
            Ty2   = ptas.apply_feedforward(TensorArrayTO(tens), tmp=False)
            idx   = np.arange(n_test)
            b_arr = Ty2.value[idx, nn_preds, 0]
            d_arr = Ty2.value[idx, nn_preds, 1]
            u_arr = Ty2.value[idx, nn_preds, 2]
            nn_score = a2[idx, nn_preds]

            def _stats(mask):
                n_cov = int(mask.sum())
                cov   = 100.0 * n_cov / n_test
                acc   = (100.0 * float((nn_preds[mask] == y_test[mask]).mean())
                         if n_cov > 0 else float("nan"))
                return cov, acc

            cov_lo, acc_lo = _stats(b_arr>= tau_lo)
            cov_hi, acc_hi = _stats(nn_score >= tau_hi)
            cov_bd, acc_bd = _stats(b_arr > u_arr)
            acc_all = 100.0 * float((nn_preds == y_test).mean())

            print(f"  {noise_label:<22}  {run:>3}  "
                  f"{cov_lo:>10.1f}  {acc_lo:>10.2f}  "
                  f"{cov_hi:>12.1f}  {acc_hi:>12.2f}  "
                  f"{cov_bd:>10.1f}  {acc_bd:>10.2f}  {acc_all:>9.2f}")
            per_run.append((cov_lo, acc_lo, cov_hi, acc_hi, cov_bd, acc_bd, acc_all))

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

        _append(_agg([0, 1, 6]), rows_lo)
        _append(_agg([2, 3, 6]), rows_hi)
        _append(_agg([4, 5, 6]), rows_bd)

    # Strip leading section/midrule rows that have no data rows after them
    def _clean(rows):
        out = []
        for i, r in enumerate(rows):
            if r[0] in ('\\midrule',) or r[0].startswith('\\multicolumn'):
                # keep only if there is at least one data row after this
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

    rows_lo = _clean(rows_lo)
    rows_hi = _clean(rows_hi)
    rows_bd = _clean(rows_bd)

    if not any(
        not (r[0] in ('\\midrule',) or r[0].startswith('\\multicolumn'))
        for rows in [rows_lo, rows_hi, rows_bd] for r in rows
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

    table_specs = [
        (
            rows_lo, f"patas_{tau_lo:.2f}",
            f"Selective prediction: PaTAS projected probability threshold $\\tau = {tau_lo:.3f}$. "
            r"Coverage is the fraction of test samples where $\pi = b + u/2 \geq \tau$ "
            r"in the NN's predicted class. "
            r"Values are mean $\pm$ std over " + str(N_RUNS) + r" independent runs.",
            "tab:threshold-patas-" + f"{tau_lo:.2f}".replace(".", "-"),
        ),
        (
            rows_hi, f"nn_{tau_hi:.2f}",
            f"Selective prediction: NN softmax score threshold "
            f"$\\tau_{{\\mathrm{{high}}}} = {tau_hi:.3f} = \\tau + (1-\\tau)/2$, "
            f"$\\tau = {tau_lo:.3f}$. "
            r"Coverage is the fraction of test samples where the NN softmax probability "
            r"in the predicted class exceeds $\tau_{\mathrm{high}}$. "
            r"Values are mean $\pm$ std over " + str(N_RUNS) + r" independent runs.",
            "tab:threshold-nn-" + f"{tau_hi:.2f}".replace(".", "-"),
        ),
        (
            rows_bd, "patas_b_gt_d",
            r"Selective prediction: PaTAS belief-over-disbelief filter ($b > d$). "
            r"Coverage is the fraction of test samples where PaTAS belief exceeds "
            r"disbelief in the NN's predicted class (net positive subjective evidence). "
            r"Values are mean $\pm$ std over " + str(N_RUNS) + r" independent runs.",
            "tab:threshold-b-gt-d",
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
                    help="Belief threshold τ: report selective-prediction accuracy "
                         "on test samples where PaTAS belief ≥ τ (requires saved models). "
                         "Can be combined with --plots-only to skip retraining.")
    ap.add_argument("--latency", action="store_true",
                    help="Run latency benchmark (inference + training overhead).")
    args = ap.parse_args()

    if args.plots_only or args.threshold is not None or args.latency:
        # Load dataset (needed for threshold eval and effectiveness plots)
        try:
            _data_dir = args.data_dir
            if _data_dir is None:
                _data_dir = make_synthetic_5g(n_bs=20, n_hours=72,
                                              cells_per_bs=2, seed=0)
            _ds = load_5g_dataset(_data_dir, n_classes=3, test_frac=0.2, seed=0)
            _nc = int(_ds.y_train.max()) + 1
        except Exception:
            _ds, _nc = None, 3

        if args.plots_only:
            print(f"Loading cached results from {args.output} ...")
            all_results = load_cached(args.output)
            if not all_results:
                print("No cached results found. Run without --plots-only first.")
            else:
                generate_outputs(all_results, args.output, ds=_ds, n_classes=_nc, run_latency=args.latency)

        if args.threshold is not None:
            if _ds is None:
                print("Dataset unavailable — cannot run threshold evaluation.")
            else:
                eval_belief_threshold(_ds, _nc, args.output, args.threshold)
    else:
        all_results, ds, n_classes = run_all(args.data_dir, args.output,
                                             force=args.force)
        generate_outputs(all_results, args.output, ds=ds, n_classes=n_classes, run_latency=args.latency)

    print("\nDone.")