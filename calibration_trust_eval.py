"""calibration_trust_eval.py

Algorithm 5 — Calibration-based Trust Evaluation using Subjective Logic.

Reference: Chapter (dissertation), "Overall Process for Calibration Trust Evaluation".

For a trained neural network the algorithm asks: *how well do the model's
predicted probabilities agree with empirical accuracy?*  Calibration error
is converted to a subjective-logic binomial opinion (b, d, u) that can be
compared across noise conditions.

Steps (Algorithm 5)
-------------------
1. Cluster predicted probabilities into M uniform bins over [0, 1].
2. For each class c and each cluster i:
       n_i  = # samples whose predicted prob for c falls in bin i
       t_i  = # of those samples that actually belong to class c
               ("good classifications")
       RP_i = mid-point of bin i  (representative value)
       r    = t_i                 (positive evidence — model was confident AND right)
       s    = |t_i − n_i * RP_i| (calibration error — negative evidence)
       omega_i = BPQ(r, s, W=2)
3. Fuse cluster opinions per class with cumulative belief fusion.
4. Fuse class opinions with cumulative belief fusion → final trust opinion.

Standalone usage
----------------
    python calibration_trust_eval.py results_noise/

It will scan the cached nn_weights.pkl files produced by eval_5g_noise.py,
assess each trained model, and write:

    results_noise/dissertation/plots/calibration_trust.pdf/.png
    results_noise/dissertation/tables/calibration_trust.tex
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from subjective_logic import Opinion, bpq, fuse_many

# ---------------------------------------------------------------------------
# Plot style (matches eval_5g_noise.py)
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

_C = {"b": "#2b7bba", "d": "#c0392b", "u": "#e07b39"}

FEATURE_SIGMAS = [0, 0.1, 0.3, 0.5]
LABEL_FLIPS    = [0, 0.05, 0.15, 0.30]
COMBINED       = [(0.1, 0.05), (0.3, 0.15), (0.5, 0.30)]
N_RUNS         = 5
N_HIDDEN       = 32


# ---------------------------------------------------------------------------
# Algorithm 5: Calibration Trust
# ---------------------------------------------------------------------------

def calibration_trust(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_clusters: int = 10,
    W: float = 2.0,
) -> Opinion:
    """Algorithm 5: assess NN trust from predicted-probability calibration.

    Parameters
    ----------
    y_true     : (n,) integer class labels
    y_prob     : (n, K) softmax probability matrix
    n_clusters : number of uniform bins M over [0, 1]
    W          : prior weight for BPQ baseline-prior quantification (W=2)

    Returns
    -------
    Opinion(b, d, u) — fused trust opinion for the neural network
    """
    if y_prob.ndim != 2:
        raise ValueError("y_prob must be (n_samples, n_classes)")
    n_samples, n_classes = y_prob.shape

    class_opinions: List[Opinion] = []

    for c in range(n_classes):
        p_c  = y_prob[:, c]       # predicted probability for class c
        is_c = (y_true == c)      # ground-truth membership

        cluster_opinions: List[Opinion] = []
        for i in range(n_clusters):
            lo   = i / n_clusters
            hi   = (i + 1) / n_clusters
            RP_i = (lo + hi) / 2   # representative probability

            # Last bin is closed on the right so p=1.0 is included
            mask = (p_c >= lo) & (p_c < hi) if i < n_clusters - 1 else (p_c >= lo) & (p_c <= hi)

            n_i = int(mask.sum())
            if n_i == 0:
                continue

            t_i = float(is_c[mask].sum())        # "good classifications"
            r   = t_i                             # positive evidence
            s   = abs(t_i - n_i * RP_i)          # calibration deviation

            cluster_opinions.append(bpq(r, s, W=W))

        if cluster_opinions:
            class_opinions.append(fuse_many(cluster_opinions, how="cumulative"))

    if not class_opinions:
        return Opinion(0.0, 0.0, 1.0)

    return fuse_many(class_opinions, how="cumulative")


# ---------------------------------------------------------------------------
# Helpers: NN forward pass from cached weights
# ---------------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _softmax(x: np.ndarray) -> np.ndarray:
    ex = np.exp(x - x.max(axis=1, keepdims=True))
    return ex / ex.sum(axis=1, keepdims=True)


def _nn_softmax(nn_pkl_path: str, X_test: np.ndarray) -> Optional[np.ndarray]:
    """Load saved NN weights and return softmax probabilities on X_test."""
    if not os.path.exists(nn_pkl_path):
        return None
    with open(nn_pkl_path, "rb") as fh:
        w = pickle.load(fh)
    a1 = _relu(X_test @ w["W1"] + w["b1"])
    a2 = _softmax(a1 @ w["W2"] + w["b2"])
    return a2


def _assess_run(nn_pkl_path: str, X_test: np.ndarray,
                y_test: np.ndarray, n_clusters: int = 10) -> Optional[Opinion]:
    """Compute calibration trust for one cached run."""
    y_prob = _nn_softmax(nn_pkl_path, X_test)
    if y_prob is None:
        return None
    return calibration_trust(y_test, y_prob, n_clusters=n_clusters)


def _aggregate_runs(
    base_dir: str,
    label_template: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_runs: int = N_RUNS,
    n_clusters: int = 10,
) -> Tuple[float, float, float, float, float, float]:
    """Return mean±std of (b, d, u) across N_RUNS for a given label template.

    label_template must contain a '{run}' placeholder, e.g.
        'fn_0.10_ptas-cal-fb_r{run}'
    """
    bs, ds, us = [], [], []
    for run in range(n_runs):
        label = label_template.format(run=run)
        pkl   = os.path.join(base_dir, label, "nn_weights.pkl")
        op    = _assess_run(pkl, X_test, y_test, n_clusters)
        if op is not None:
            bs.append(op.b); ds.append(op.d); us.append(op.u)
    if not bs:
        nan = float("nan")
        return nan, nan, nan, nan, nan, nan
    return (float(np.mean(bs)), float(np.std(bs)),
            float(np.mean(ds)), float(np.std(ds)),
            float(np.mean(us)), float(np.std(us)))


# ---------------------------------------------------------------------------
# Panel plot helper
# ---------------------------------------------------------------------------

def _plot_trust_panel(
    ax_top: plt.Axes,
    ax_bot: plt.Axes,
    x_vals,
    b_means, b_stds,
    d_means, d_stds,
    u_means, u_stds,
    xlabel: str,
    tick_labels=None,
    show_ylabel: bool = True,
):
    xs     = np.array(x_vals, dtype=float)
    bw_bar = min(np.diff(xs).min() if len(xs) > 1 else 1.0,
                 0.04 if tick_labels is None else 0.25) * 0.9
    xlim   = (xs[0] - 2.5 * bw_bar, xs[-1] + 2.5 * bw_bar)

    # Top: lines with shaded std bands
    for means, stds, color, lbl in [
        (b_means, b_stds, _C["b"], "Belief $b$"),
        (d_means, d_stds, _C["d"], "Disbelief $d$"),
        (u_means, u_stds, _C["u"], "Uncertainty $u$"),
    ]:
        lo = np.array(means) - np.array(stds)
        hi = np.array(means) + np.array(stds)
        ax_top.plot(x_vals, means, color=color, lw=2, marker="o", ms=6, label=lbl)
        ax_top.fill_between(x_vals, lo, hi, alpha=0.18, color=color)

    if show_ylabel:
        ax_top.set_ylabel("Calibration trust opinion mass")
    ax_top.set_xlim(*xlim)
    ax_top.set_ylim(-0.02, 1.02)
    ax_top.legend(fontsize=8)
    if tick_labels is not None:
        ax_top.set_xticks(x_vals)
        ax_top.set_xticklabels(tick_labels, fontsize=7)

    # Bottom: bar chart of mean values with std error bars
    for off, means, stds, color, lbl in zip(
        [-bw_bar, 0.0, bw_bar],
        [b_means, d_means, u_means],
        [b_stds,  d_stds,  u_stds],
        [_C["b"],  _C["d"],  _C["u"]],
        ["b", "d", "u"],
    ):
        bars = ax_bot.bar(xs + off, means, width=bw_bar,
                          color=color, label=lbl,
                          yerr=stds, capsize=3,
                          error_kw={"elinewidth": 1, "ecolor": "black"},
                          zorder=3)
        for bar, m in zip(bars, means):
            if np.isnan(m) or abs(m) < 1e-6:
                continue
            ax_bot.text(bar.get_x() + bar.get_width() / 2,
                        m + (max(stds) * 0.5 if stds else 0.01) + 0.01,
                        f"{m:.3f}", ha="center", va="bottom",
                        fontsize=5.5, color=color)

    ax_bot.axhline(0, color="black", lw=0.8)
    ax_bot.set_xlabel(xlabel)
    if show_ylabel:
        ax_bot.set_ylabel("Mean trust mass")
    ax_bot.set_xlim(*xlim)
    ax_bot.set_ylim(-0.02, 1.02)
    ax_bot.legend(fontsize=7, ncol=3)
    ax_bot.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    if tick_labels is not None:
        ax_bot.set_xticks(x_vals)
        ax_bot.set_xticklabels(tick_labels, fontsize=7)


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------

def calibration_trust_analysis(ds, output_dir: str, n_clusters: int = 10):
    """Assess calibration trust for all trained models in the experiment grid.

    Scans cached nn_weights.pkl files under output_dir/data/, applies
    Algorithm 5 to each, averages across N_RUNS, and writes:

        plots/calibration_trust.pdf / .png
        tables/calibration_trust.tex

    Parameters
    ----------
    ds         : dataset object with .X_test and .y_test attributes
    output_dir : root results directory (e.g. 'results_noise')
    n_clusters : number of probability bins M (default 10)
    """
    dis_dir  = os.path.join(output_dir, "dissertation")
    plots_dir  = os.path.join(dis_dir, "plots")
    tables_dir = os.path.join(dis_dir, "tables")
    os.makedirs(plots_dir,  exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    data_dir  = os.path.join(output_dir, "data")
    X_test    = ds.X_test
    y_test    = ds.y_test

    # ------------------------------------------------------------------
    # Feature noise
    # ------------------------------------------------------------------
    fn_b_means, fn_b_stds = [], []
    fn_d_means, fn_d_stds = [], []
    fn_u_means, fn_u_stds = [], []

    fn_tex_rows: list = []

    for sigma in FEATURE_SIGMAS:
        tmpl = f"fn_{sigma:.2f}_ptas-cal-fb_r{{run}}"
        bm, bs, dm, ds_, um, us = _aggregate_runs(data_dir, tmpl, X_test, y_test,
                                                   n_clusters=n_clusters)
        fn_b_means.append(bm); fn_b_stds.append(bs)
        fn_d_means.append(dm); fn_d_stds.append(ds_)
        fn_u_means.append(um); fn_u_stds.append(us)

        def _fmt(m, s):
            return f"{m:.4f}$\\pm${s:.4f}" if not np.isnan(m) else "—"

        fn_tex_rows.append([
            f"{sigma:.2f}",
            _fmt(bm, bs), _fmt(dm, ds_), _fmt(um, us),
            "b" if (not np.isnan(bm) and bm == max(bm, dm, um)) else
            ("u" if (not np.isnan(um) and um == max(bm, dm, um)) else "d"),
        ])

    # ------------------------------------------------------------------
    # Label noise
    # ------------------------------------------------------------------
    ln_b_means, ln_b_stds = [], []
    ln_d_means, ln_d_stds = [], []
    ln_u_means, ln_u_stds = [], []

    ln_tex_rows: list = []

    for flip in LABEL_FLIPS:
        tmpl = f"ln_{flip:.2f}_ptas-cal-fb_r{{run}}"
        bm, bs, dm, ds_, um, us = _aggregate_runs(data_dir, tmpl, X_test, y_test,
                                                   n_clusters=n_clusters)
        ln_b_means.append(bm); ln_b_stds.append(bs)
        ln_d_means.append(dm); ln_d_stds.append(ds_)
        ln_u_means.append(um); ln_u_stds.append(us)
        ln_tex_rows.append([
            f"{flip:.2f}",
            _fmt(bm, bs), _fmt(dm, ds_), _fmt(um, us),
            "b" if (not np.isnan(bm) and bm == max(bm, dm, um)) else
            ("u" if (not np.isnan(um) and um == max(bm, dm, um)) else "d"),
        ])

    # ------------------------------------------------------------------
    # Combined noise
    # ------------------------------------------------------------------
    cb_b_means, cb_b_stds = [], []
    cb_d_means, cb_d_stds = [], []
    cb_u_means, cb_u_stds = [], []

    cb_tex_rows: list = []
    cb_x      = list(range(len(COMBINED)))
    cb_ticks  = [f"$\\sigma$={s:.2f}\n$p$={p:.2f}" for s, p in COMBINED]

    for sigma, flip in COMBINED:
        tmpl = f"comb_{sigma:.2f}_{flip:.2f}_ptas-cal-fb_r{{run}}"
        bm, bs, dm, ds_, um, us = _aggregate_runs(data_dir, tmpl, X_test, y_test,
                                                   n_clusters=n_clusters)
        cb_b_means.append(bm); cb_b_stds.append(bs)
        cb_d_means.append(dm); cb_d_stds.append(ds_)
        cb_u_means.append(um); cb_u_stds.append(us)
        cb_tex_rows.append([
            f"({sigma:.2f},{flip:.2f})",
            _fmt(bm, bs), _fmt(dm, ds_), _fmt(um, us),
            "b" if (not np.isnan(bm) and bm == max(bm, dm, um)) else
            ("u" if (not np.isnan(um) and um == max(bm, dm, um)) else "d"),
        ])

    # ------------------------------------------------------------------
    # Plot — 2×3 grid: (top/bot rows) × (feature / label / combined cols)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 7),
                             gridspec_kw={"height_ratios": [2, 1]})

    _plot_trust_panel(
        axes[0, 0], axes[1, 0],
        FEATURE_SIGMAS,
        fn_b_means, fn_b_stds,
        fn_d_means, fn_d_stds,
        fn_u_means, fn_u_stds,
        xlabel=r"Feature noise $\sigma_{rel}$",
        show_ylabel=True,
    )
    axes[0, 0].set_title(r"Feature noise ($\sigma_{rel}$-calibrated $T_x$)", fontsize=10)

    # _plot_trust_panel(
    #     axes[0, 1], axes[1, 1],
    #     LABEL_FLIPS,
    #     ln_b_means, ln_b_stds,
    #     ln_d_means, ln_d_stds,
    #     ln_u_means, ln_u_stds,
    #     xlabel=r"Label flip rate $p$",
    #     show_ylabel=False,
    # )
    # axes[0, 1].set_title("Label noise (trusted $T_x$)", fontsize=10)

    _plot_trust_panel(
        axes[0, 1], axes[1, 1],
        cb_x,
        cb_b_means, cb_b_stds,
        cb_d_means, cb_d_stds,
        cb_u_means, cb_u_stds,
        xlabel="Condition",
        tick_labels=cb_ticks,
        show_ylabel=False,
    )
    axes[0, 1].set_title("Combined noise ($T_x$ calibrated)", fontsize=10)

    fig.suptitle(
        "Calibration Trust of trained NNs across noise conditions\n"
        f"(mean ± std over {N_RUNS} runs, M={n_clusters} bins, W=2, BPQ + cumulative fusion)",
        fontsize=11,
    )
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(plots_dir, f"calibration_trust.{ext}")
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.join(plots_dir, 'calibration_trust.pdf')}")

    # ------------------------------------------------------------------
    # LaTeX table
    # ------------------------------------------------------------------
    header = [
        "Condition",
        r"Belief $b$ (mean$\pm$std)",
        r"Disbelief $d$ (mean$\pm$std)",
        r"Uncertainty $u$ (mean$\pm$std)",
        "Dominant",
    ]

    rows = []
    rows.append([r"\multicolumn{5}{l}{\textit{Feature noise (calibrated $T_x$)}}"])
    rows.extend(fn_tex_rows)
    rows.append([r"\midrule"])
    rows.append([r"\multicolumn{5}{l}{\textit{Label noise (trusted $T_x$)}}"])
    rows.extend(ln_tex_rows)
    rows.append([r"\midrule"])
    rows.append([r"\multicolumn{5}{l}{\textit{Combined noise}}"])
    rows.extend(cb_tex_rows)

    col_fmt = "l" + "c" * (len(header) - 1)
    tex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Algorithm~5 calibration-trust opinions for trained NNs across noise "
        r"conditions. Opinions averaged over " + str(N_RUNS) + r" independent runs. "
        r"High belief $b$ indicates a well-calibrated, trustworthy model; "
        r"high uncertainty $u$ indicates insufficient evidence.}",
        r"\label{tab:calibration-trust}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        tex_lines.append(" & ".join(str(c) for c in row) + r" \\")
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex_path = os.path.join(tables_dir, "calibration_trust.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(tex_lines))
    print(f"  Saved {tex_path}")

    return {
        "fn": list(zip(FEATURE_SIGMAS, fn_b_means, fn_d_means, fn_u_means)),
        "ln": list(zip(LABEL_FLIPS,    ln_b_means, ln_d_means, ln_u_means)),
        "cb": list(zip(COMBINED,       cb_b_means, cb_d_means, cb_u_means)),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from data_loader import load_5g_dataset, make_synthetic_5g

    parser = argparse.ArgumentParser(
        description="Algorithm 5 calibration trust evaluation for 5G noise experiments."
    )
    parser.add_argument("output_dir", nargs="?", default="results_noise",
                        help="Root results directory (default: results_noise)")
    parser.add_argument("--data-dir", default=None,
                        help="5G dataset directory (synthetic data used if omitted)")
    parser.add_argument("--bins", type=int, default=10,
                        help="Number of probability bins M (default: 10)")
    args = parser.parse_args()

    if args.data_dir is None:
        print("No data dir — using synthetic 5G dataset ...")
        data_dir = make_synthetic_5g(n_bs=20, n_hours=72, cells_per_bs=2, seed=0)
    else:
        data_dir = args.data_dir

    ds = load_5g_dataset(data_dir, n_classes=3, test_frac=0.2, seed=0)
    print(f"\nRunning Algorithm 5 calibration trust analysis (M={args.bins} bins) ...")
    calibration_trust_analysis(ds, args.output_dir, n_clusters=args.bins)
    print("Done.")
