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
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sl_calibration"))
import pandas as pd
from analysis import compute_global_opinion, build_cluster_lookup, get_per_prediction_trust
from patas_module.subjective_logic import Opinion

# ---------------------------------------------------------------------------
# Plot style (matches eval_5g_noise.py)
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         16,
    "axes.titlesize":    16,
    "axes.labelsize":    16,
    "legend.fontsize":   16,
    "xtick.labelsize":   16,
    "ytick.labelsize":   16,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "legend.fontsize":   14,
})

_C = {"b": "#2b7bba", "d": "#c0392b", "u": "#e07b39"}

from hyperparams import FEATURE_SIGMAS, LABEL_FLIPS, COMBINED, N_RUNS
# ---------------------------------------------------------------------------
# Algorithm 5: Calibration Trust
# ---------------------------------------------------------------------------

def calibration_trust(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_clusters: int = 10,
) -> Opinion:
    """Algorithm 5: assess NN trust from predicted-probability calibration.

    Parameters
    ----------
    y_true     : (n,) integer class labels
    y_prob     : (n, K) softmax probability matrix
    n_clusters : number of uniform bins M over [0, 1]

    Returns
    -------
    Opinion(b, d, u) — fused trust opinion for the neural network
    """
    if y_prob.ndim != 2:
        raise ValueError("y_prob must be (n_samples, n_classes)")
    cols = [f"Class_{i}_Probability" for i in range(y_prob.shape[1])]
    df = pd.DataFrame(y_prob, columns=cols)
    df["True Label"] = y_true
    op = compute_global_opinion(df, n_clusters=n_clusters)
    return Opinion(op.t, op.d, op.u)


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
# Dynamic trust assessment and calibration coverage helpers
# ---------------------------------------------------------------------------

_CAL_COLORS = {"nn": "#555555", "cal_pi": _C["u"]}
_CAL_LABELS = {"nn": r"NN softmax $\hat{p}$",
               "cal_pi": r"Cal. trust $\pi = b + u/2$"}


def _get_per_prediction(
    nn_pkl_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_clusters: int = 10,
) -> Optional[dict]:
    """Load NN, build cluster lookup, return per-prediction trust dict."""
    y_prob = _nn_softmax(nn_pkl_path, X_test)
    if y_prob is None:
        return None
    n_cls = y_prob.shape[1]
    df = pd.DataFrame(y_prob,
                      columns=[f"Class_{i}_Probability" for i in range(n_cls)])
    df["True Label"] = y_test
    lookup = build_cluster_lookup(df, n_clusters=n_clusters)
    return get_per_prediction_trust(df, lookup, n_clusters=n_clusters)


def _coverage_sweep(
    scores: np.ndarray, correct: np.ndarray, n_test: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Sweep threshold τ over scores; return (coverage%, accuracy%) arrays."""
    finite = scores[np.isfinite(scores)]
    if len(finite) == 0:
        return np.array([100.0]), np.array([np.nan])
    taus = np.unique(np.quantile(finite, np.linspace(0, 1, 400)))
    covs = np.empty(len(taus))
    accs = np.full(len(taus), np.nan)
    for i, tau in enumerate(taus):
        mask  = scores >= tau
        n_cov = int(mask.sum())
        covs[i] = 100.0 * n_cov / n_test
        if n_cov > 0:
            accs[i] = 100.0 * float(correct[mask].mean())
    return covs, accs


def plot_dynamic_trust(
    X_test: np.ndarray,
    y_test: np.ndarray,
    conditions: list,
    plots_dir: str,
    axis_name: str,
    axis_title: str,
    n_clusters: int = 10,
) -> None:
    """3-panel dynamic trust assessment figure (one row per noise condition).

    Panels: confidence distribution | belief distribution |
            mean b/d/u per confidence bin.
    """
    valid = [(lbl, pkl) for lbl, pkl in conditions if os.path.exists(pkl)]
    if not valid:
        return

    bin_edges   = np.linspace(0, 1, n_clusters + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bw          = (bin_edges[1] - bin_edges[0]) * 0.85

    n_rows = len(valid)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 3.5 * n_rows),
                             gridspec_kw={"wspace": 0.35, "hspace": 0.5})
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, (lbl, pkl) in enumerate(valid):
        res = _get_per_prediction(pkl, X_test, y_test, n_clusters)
        if res is None:
            for c in range(3):
                axes[row, c].set_visible(False)
            continue

        confidence = res["confidence"]
        belief     = res["belief"]
        correct    = res["correct"]

        def _stacked(ax, vals, xlabel, title, log_weighted=False, _correct=correct):
            tc, _ = np.histogram(vals, bins=bin_edges)
            cc, _ = np.histogram(vals[_correct], bins=bin_edges)
            safe    = np.maximum(tc, 1)
            heights = np.log1p(tc.astype(float)) if log_weighted else np.where(tc > 0, 1.0, 0.0)
            h_c = heights * cc / safe
            h_i = heights * (tc - cc) / safe
            ax.bar(bin_centres, h_c, width=bw,
                   color=_C["b"], alpha=0.75, label="Correct")
            ax.bar(bin_centres, h_i, width=bw,
                   color=_C["d"], alpha=0.75, label="Incorrect", bottom=h_c)
            y_top = heights.max() if heights.max() > 0 else 1.0
            for cx, tot, h in zip(bin_centres, tc, heights):
                if tot > 0:
                    ax.text(cx, h + y_top * 0.02, str(int(tot)),
                            ha="center", va="bottom", fontsize=5, rotation=90)
            ax.set_xlabel(xlabel, fontsize=9)
            ylabel = (r"$\log(1+n)$" if log_weighted else "Proportion")
            ax.set_ylabel(ylabel if row == 0 else "", fontsize=9)
            ax.set_title(title if row == 0 else "", fontsize=9)
            ax.legend(fontsize=7)
            ax.set_ylim(0, y_top * 1.35)
            ax.grid(axis="y", linestyle=":", alpha=0.3)

        _stacked(axes[row, 0], confidence,
                 "Predicted confidence", "Confidence Distribution", log_weighted=True)
        _stacked(axes[row, 1], belief,
                 r"Trust belief $b$", "Belief Distribution")

        ax = axes[row, 2]
        mb, md, mu = [], [], []
        _W = 2.0
        for i in range(n_clusters):
            m = (confidence > bin_edges[i]) & (confidence <= bin_edges[i + 1])
            n_bin = int(m.sum())
            if n_bin == 0:
                mb.append(np.nan); md.append(np.nan); mu.append(np.nan)
                continue
            # BPQ directly on this argmax-confidence bin so that u reflects
            # the actual sample count, not the larger per-class lookup pool.
            n_correct = int(correct[m].sum())
            rp        = bin_centres[i]
            r_bpq     = float(n_correct)
            s_bpq     = abs(n_correct - n_bin * rp)
            denom     = r_bpq + s_bpq + _W
            mb.append(r_bpq / denom)
            md.append(s_bpq / denom)
            mu.append(_W    / denom)
        ax.plot(bin_centres, mb, "o-",  color=_C["b"], lw=1.8, ms=4,
                label=r"Belief $b$")
        ax.plot(bin_centres, md, "s--", color=_C["d"], lw=1.8, ms=4,
                label=r"Disbelief $d$")
        ax.plot(bin_centres, mu, "^:",  color=_C["u"], lw=1.8, ms=4,
                label=r"Uncertainty $u$")
        ax.set_xlabel("Predicted confidence", fontsize=9)
        ax.set_ylabel("Mean opinion" if row == 0 else "", fontsize=9)
        ax.set_title("Mean Trust per Confidence Bin" if row == 0 else "",
                     fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 1)
        ax.grid(linestyle=":", alpha=0.3)

        axes[row, 0].annotate(lbl, xy=(-0.35, 0.5), xycoords="axes fraction",
                              rotation=90, va="center", ha="center",
                              fontsize=8, fontweight="bold")

    fig.suptitle(
        f"Dynamic Trust Assessment — {axis_title} (M={n_clusters} bins)",
        fontsize=11,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(plots_dir, f"dynamic_trust_{axis_name}.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.join(plots_dir, f'dynamic_trust_{axis_name}.pdf')}")


def plot_calibration_coverage(
    X_test: np.ndarray,
    y_test: np.ndarray,
    conditions: list,
    plots_dir: str,
    axis_name: str,
    axis_title: str,
    n_clusters: int = 10,
) -> None:
    """Coverage-accuracy curves and ROC curves: calibration π = b + u/2 vs. NN softmax."""
    try:
        from sklearn.metrics import roc_curve as _sk_roc, auc as _sk_auc
        _has_sklearn = True
    except ImportError:
        _has_sklearn = False

    valid = [(lbl, pkl) for lbl, pkl in conditions if os.path.exists(pkl)]
    if not valid:
        return

    n_cond = len(valid)
    ncols  = min(n_cond, 2)
    nrows_per_block = 2  # row 0: coverage-accuracy; row 1: ROC
    nblocks = (n_cond + ncols - 1) // ncols
    nrows   = nblocks * nrows_per_block

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows),
                             squeeze=False)

    n_test = len(y_test)
    for idx, (lbl, pkl) in enumerate(valid):
        block  = idx // ncols
        col    = idx % ncols
        ax_cov = axes[block * nrows_per_block][col]
        ax_roc = axes[block * nrows_per_block + 1][col]

        y_prob = _nn_softmax(pkl, X_test)
        if y_prob is None:
            ax_cov.set_visible(False)
            ax_roc.set_visible(False)
            continue

        nn_pred = y_prob.argmax(axis=1)
        correct = (nn_pred == y_test)
        correct_bin = correct.astype(int)

        n_cls = y_prob.shape[1]
        df = pd.DataFrame(y_prob,
                          columns=[f"Class_{i}_Probability" for i in range(n_cls)])
        df["True Label"] = y_test
        lookup = build_cluster_lookup(df, n_clusters=n_clusters)
        res    = get_per_prediction_trust(df, lookup, n_clusters=n_clusters)
        pi_cal = res["belief"] + 0.5 * res["uncertainty"]

        score_map = [("nn", y_prob.max(axis=1)), ("cal_pi", pi_cal)]

        # ---- coverage-accuracy ----
        for key, scores in score_map:
            covs, accs = _coverage_sweep(scores, correct, n_test)
            valid_mask = ~np.isnan(accs)
            if not valid_mask.any():
                continue
            ax_cov.plot(covs[valid_mask], accs[valid_mask],
                        color=_CAL_COLORS[key], lw=2, label=_CAL_LABELS[key])
            if not np.isnan(accs[0]):
                ax_cov.scatter([covs[0]], [accs[0]],
                               color=_CAL_COLORS[key], s=35, zorder=5)

        ax_cov.set_title(lbl, fontsize=9)
        ax_cov.set_xlabel("Coverage (%)")
        ax_cov.set_ylabel("Accuracy on covered (%)")
        ax_cov.set_xlim(-2, 105)
        ax_cov.legend(fontsize=7, loc="lower left")
        ax_cov.grid(linestyle=":", alpha=0.4)

        # ---- ROC curves ----
        if _has_sklearn:
            for key, scores in score_map:
                if not np.isfinite(scores).any():
                    continue
                _fpr, _tpr, _ = _sk_roc(correct_bin, scores)
                _auc_val = _sk_auc(_fpr, _tpr)
                ax_roc.plot(_fpr, _tpr,
                            color=_CAL_COLORS[key], lw=2,
                            label=f"{_CAL_LABELS[key]}  AUC={_auc_val:.3f}")
            ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title(f"ROC — {lbl}", fontsize=9)
            ax_roc.legend(fontsize=7, loc="lower right")
            ax_roc.grid(linestyle=":", alpha=0.4)
            ax_roc.set_xlim(-0.02, 1.02)
            ax_roc.set_ylim(-0.02, 1.02)
        else:
            ax_roc.set_visible(False)

    for idx in range(n_cond, nblocks * ncols):
        block = idx // ncols
        col   = idx % ncols
        axes[block * nrows_per_block][col].set_visible(False)
        axes[block * nrows_per_block + 1][col].set_visible(False)

    fig.suptitle(
        f"Calibration-based Coverage-Accuracy & ROC — {axis_title}\n"
        r"($\pi = b + u/2$ threshold vs.\ NN softmax)",
        fontsize=11,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(plots_dir,
                         f"coverage_accuracy_calibration_{axis_name}.{ext}"),
            bbox_inches="tight",
        )
    plt.close(fig)
    print(f"  Saved coverage_accuracy_calibration_{axis_name}.pdf")


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
    fig, axes = plt.subplots(2, 3, figsize=(15, 7),
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

    _plot_trust_panel(
        axes[0, 1], axes[1, 1],
        LABEL_FLIPS,
        ln_b_means, ln_b_stds,
        ln_d_means, ln_d_stds,
        ln_u_means, ln_u_stds,
        xlabel=r"Label flip rate $p$",
        show_ylabel=False,
    )
    axes[0, 2].set_title("Label noise (trusted $T_x$)", fontsize=10)

    _plot_trust_panel(
        axes[0, 2], axes[1, 2],
        cb_x,
        cb_b_means, cb_b_stds,
        cb_d_means, cb_d_stds,
        cb_u_means, cb_u_stds,
        xlabel="Condition",
        tick_labels=cb_ticks,
        show_ylabel=False,
    )
    axes[0, 2].set_title("Combined noise ($T_x$ calibrated)", fontsize=10)

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

    # ------------------------------------------------------------------
    # Dynamic trust assessment and calibration coverage plots
    # ------------------------------------------------------------------
    _axes_spec = [
        ("feature",  r"Feature noise",
         [(f"$\\sigma_{{rel}}={s:.2f}$",
           os.path.join(data_dir, f"fn_{s:.2f}_ptas-cal-fb_r0", "nn_weights.pkl"))
          for s in FEATURE_SIGMAS[:4]]),
        ("label",    "Label noise",
         [(f"$p={f:.2f}$",
           os.path.join(data_dir, f"ln_{f:.2f}_ptas-cal-fb_r0", "nn_weights.pkl"))
          for f in LABEL_FLIPS[:4]]),
        ("combined", "Combined noise",
         [(f"$\\sigma={s:.2f},\\,p={f:.2f}$",
           os.path.join(data_dir, f"comb_{s:.2f}_{f:.2f}_ptas-cal-fb_r0",
                        "nn_weights.pkl"))
          for s, f in COMBINED[:4]]),
    ]
    print("\nGenerating dynamic trust assessment plots ...")
    for _axis_name, _axis_title, _conds in _axes_spec:
        plot_dynamic_trust(X_test, y_test, _conds, plots_dir,
                           _axis_name, _axis_title, n_clusters)
    print("\nGenerating calibration coverage plots ...")
    for _axis_name, _axis_title, _conds in _axes_spec:
        plot_calibration_coverage(X_test, y_test, _conds, plots_dir,
                                  _axis_name, _axis_title, n_clusters)

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
