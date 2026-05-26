"""latency_eval.py

PaTAS latency benchmark for the 5G use-case evaluation.

Two components are measured independently:

(A) Inference latency
---------------------
    NN forward pass vs PaTAS apply_feedforward, varying batch size.
    No socket involved — both run in-process via numpy.
    Repeated N_INF_REPEATS times; median is reported.

(B) Training latency
--------------------
    Wall-clock time for one full training run (N_BENCH_EPOCHS epochs)
    without PaTAS vs with PaTAS (PTAS + NN in separate processes).
    Repeated N_TRAIN_TRIALS times; mean ± std is reported.
    Three noise conditions are benchmarked:
        clean   (sigma=0.0, flip=0.0)
        feature (sigma=0.3, flip=0.0)
        combined(sigma=0.3, flip=0.15)

Outputs
-------
    <output_dir>/dissertation/plots/latency.pdf / .png
    <output_dir>/dissertation/tables/latency.tex

Standalone usage
----------------
    python latency_eval.py [results_noise] [--data-dir PATH]

Can also be imported and called from eval_5g_noise.py:
    from latency_eval import latency_analysis
    latency_analysis(ds, output_dir)
"""
from __future__ import annotations

import os
import sys
import time
import tempfile
import shutil
import pickle
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

_C_NN   = "#555555"
_C_PTAS = "#8e44ad"

# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------

INF_BATCH_SIZES   = [1, 8, 32, 64, 128, 256, 512, 1024]
N_INF_REPEATS     = 200      # timing repetitions per batch size (median taken)
N_BENCH_EPOCHS    = 5        # epochs for training benchmark
N_TRAIN_TRIALS    = 3        # independent timing trials for training
BENCH_PORT_BASE   = 7200     # starting port for benchmark processes
N_HIDDEN          = 32
BATCH             = 64
LR                = 0.05
EPS_LOW           = 0.05

TRAIN_CONDITIONS = [
    ("clean",    0.0, 0.0),
    ("feature",  0.3, 0.0),
    ("combined", 0.3, 0.15),
]


# ---------------------------------------------------------------------------
# Part A — Inference latency
# ---------------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _softmax(x: np.ndarray) -> np.ndarray:
    ex = np.exp(x - x.max(axis=1, keepdims=True))
    return ex / ex.sum(axis=1, keepdims=True)


def _nn_forward(W1, b1, W2, b2, X: np.ndarray) -> np.ndarray:
    return _softmax(_relu(X @ W1 + b1) @ W2 + b2)


def _time_inference(
    W1, b1, W2, b2,
    X_test: np.ndarray,
    ptas_class,
    TensorArrayTO,
    n_repeats: int = N_INF_REPEATS,
) -> Tuple[List[float], List[float]]:
    """Return (nn_ms_list, ptas_ms_list) for the given test batch."""
    dim = X_test.shape[1]

    # Pre-build PaTAS state (trusted opinion — same as used in effectiveness analysis)
    from noise_utils import feature_noise_to_trust
    trust_op = feature_noise_to_trust(0.0)  # fully trusted = b=1, d=0, u=0
    bv, dv, uv = float(trust_op.t), float(trust_op.d), float(trust_op.u)
    n = len(X_test)
    tens = np.empty((n, dim, 3), dtype=np.float32)
    tens[..., 0] = bv; tens[..., 1] = dv; tens[..., 2] = uv

    # Load omega_thetas from weights (dummy vacuous)
    n_hidden  = W1.shape[1]
    n_classes = W2.shape[1]
    from patas_module.concrete.TrustOpinion import TrustOpinion
    omega_thetas = [
        TensorArrayTO(np.full((dim + 1, n_hidden,  3), [1.0, 0.0, 0.0], dtype=np.float32)),
        TensorArrayTO(np.full((n_hidden + 1, n_classes, 3), [1.0, 0.0, 0.0], dtype=np.float32)),
    ]
    ptas = ptas_class(
        omega_thetas=omega_thetas,
        operator_mapping=None,
        nn_interface=None,
        trust_assessment_func=None,
        structure=[dim, n_hidden, n_classes],
        use_tensor=True,
    )
    ta = TensorArrayTO(tens)

    # Warm-up
    _ = _nn_forward(W1, b1, W2, b2, X_test)
    _ = ptas.apply_feedforward(ta, tmp=False)

    nn_times, ptas_times = [], []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _nn_forward(W1, b1, W2, b2, X_test)
        nn_times.append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        ptas.apply_feedforward(ta, tmp=False)
        ptas_times.append((time.perf_counter() - t0) * 1e3)

    return nn_times, ptas_times


def run_inference_benchmark(
    X_test: np.ndarray,
    nn_pkl_path: Optional[str],
) -> dict:
    """Run Part A — inference latency across batch sizes.

    Returns dict mapping batch_size -> (nn_median_ms, ptas_median_ms).
    """
    try:
        from patas_module.concrete.TensorTO import TensorArrayTO
        from patas_module.NN.PTAStemplate import PTAS as PTASClass
    except ImportError:
        print("  [skip] patas_module not available — skipping inference benchmark")
        return {}

    # Load or create weights
    if nn_pkl_path and os.path.exists(nn_pkl_path):
        with open(nn_pkl_path, "rb") as fh:
            w = pickle.load(fh)
        W1, b1, W2, b2 = w["W1"], w["b1"], w["W2"], w["b2"]
    else:
        # Random weights for timing (accuracy irrelevant here)
        rng  = np.random.default_rng(0)
        dim  = X_test.shape[1]
        W1   = rng.standard_normal((dim, N_HIDDEN)).astype(np.float32) * np.sqrt(2.0 / dim)
        b1   = np.zeros((1, N_HIDDEN), dtype=np.float32)
        W2   = rng.standard_normal((N_HIDDEN, 3)).astype(np.float32) * np.sqrt(2.0 / N_HIDDEN)
        b2   = np.zeros((1, 3), dtype=np.float32)

    results: dict = {}
    for bs in INF_BATCH_SIZES:
        X_batch = X_test[:bs] if bs <= len(X_test) else np.tile(X_test, (bs // len(X_test) + 1, 1))[:bs]
        X_batch = X_batch.astype(np.float32)
        nn_times, ptas_times = _time_inference(
            W1, b1, W2, b2, X_batch, PTASClass, TensorArrayTO,
        )
        results[bs] = (float(np.median(nn_times)), float(np.median(ptas_times)))
        nn_m, pt_m = results[bs]
        print(f"    batch={bs:4d}  NN={nn_m:.3f} ms  PTAS={pt_m:.3f} ms  "
              f"overhead={pt_m/max(nn_m, 1e-9):.1f}×")

    return results


# ---------------------------------------------------------------------------
# Part B — Training latency
# ---------------------------------------------------------------------------

def _run_timed(
    X_train, y_train_oh, X_test, y_test_oh,
    n_classes: int,
    sigma: float, flip: float,
    use_ptas: bool,
    port: int,
    tmp_dir: str,
    trial: int,
) -> float:
    """Run one training trial and return wall-clock seconds."""
    from external_bridge import run_with_external_implementation
    from noise_utils import feature_noise_to_trust, label_noise_to_trust

    x_trust = feature_noise_to_trust(sigma) if use_ptas else "trusted"
    y_trust = label_noise_to_trust(flip)    if use_ptas else "trusted"

    label  = f"bench_{'ptas' if use_ptas else 'nn'}_{sigma:.2f}_{flip:.2f}_t{trial}"
    run_lbl = os.path.join(tmp_dir, label)

    t0 = time.perf_counter()
    run_with_external_implementation(
        data_dir=None,
        dataset="5g",
        n_hidden=N_HIDDEN,
        epochs=N_BENCH_EPOCHS,
        batch=BATCH,
        lr=LR,
        eps_low=EPS_LOW,
        x_trust=x_trust if use_ptas else "trusted",
        y_trust=y_trust if use_ptas else "trusted",
        use_ptas=use_ptas,
        port=port,
        run_label=run_lbl,
        X_train=X_train,
        y_train_oh=y_train_oh,
        X_test=X_test,
        y_test_oh=y_test_oh,
        # trust_feedback=use_ptas,
    )
    return time.perf_counter() - t0


def run_training_benchmark(ds) -> dict:
    """Run Part B — training wall-clock comparison across noise conditions.

    Returns dict mapping condition_name -> {
        'nn':   (mean_s, std_s, per_epoch_mean_s),
        'ptas': (mean_s, std_s, per_epoch_mean_s),
    }
    """
    from noise_utils import add_feature_noise, add_label_noise

    n_classes = int(ds.y_train.max()) + 1
    results   = {}
    port      = BENCH_PORT_BASE

    tmp_dir = tempfile.mkdtemp(prefix="latency_bench_")
    try:
        for cond_name, sigma, flip in TRAIN_CONDITIONS:
            print(f"\n  Condition: {cond_name}  (sigma={sigma}, flip={flip})")
            rng = np.random.default_rng(42)
            X_tr = add_feature_noise(ds.X_train, sigma, rng=rng)
            y_tr_int = add_label_noise(ds.y_train, flip, n_classes, rng=rng)
            y_tr_oh  = np.eye(n_classes, dtype=np.float32)[y_tr_int]
            X_te     = ds.X_test
            y_te_oh  = np.eye(n_classes, dtype=np.float32)[ds.y_test]

            nn_times, ptas_times = [], []

            for trial in range(N_TRAIN_TRIALS):
                print(f"    trial {trial+1}/{N_TRAIN_TRIALS}  [NN] ...", end=" ", flush=True)
                t_nn = _run_timed(X_tr, y_tr_oh, X_te, y_te_oh,
                                   n_classes, sigma, flip,
                                   use_ptas=False, port=port,
                                   tmp_dir=tmp_dir, trial=trial)
                nn_times.append(t_nn)
                print(f"{t_nn:.1f}s", end="   ", flush=True)

                print(f"[PTAS] ...", end=" ", flush=True)
                port += 1
                t_ptas = _run_timed(X_tr, y_tr_oh, X_te, y_te_oh,
                                     n_classes, sigma, flip,
                                     use_ptas=True, port=port,
                                     tmp_dir=tmp_dir, trial=trial)
                ptas_times.append(t_ptas)
                print(f"{t_ptas:.1f}s")
                port += 1

            results[cond_name] = {
                "sigma": sigma, "flip": flip,
                "nn":   (np.mean(nn_times),   np.std(nn_times),
                         np.mean(nn_times)   / N_BENCH_EPOCHS),
                "ptas": (np.mean(ptas_times), np.std(ptas_times),
                         np.mean(ptas_times) / N_BENCH_EPOCHS),
            }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_inference(ax_top: plt.Axes, ax_bot: plt.Axes, inf_results: dict):
    if not inf_results:
        ax_top.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax_top.transAxes)
        return

    xs      = sorted(inf_results.keys())
    nn_ms   = [inf_results[b][0] for b in xs]
    ptas_ms = [inf_results[b][1] for b in xs]
    ratio   = [p / max(n, 1e-9) for n, p in zip(nn_ms, ptas_ms)]

    ax_top.plot(xs, nn_ms,   color=_C_NN,   marker="o", lw=2, ms=6, label="NN only")
    ax_top.plot(xs, ptas_ms, color=_C_PTAS, marker="s", lw=2, ms=6, label="NN + PaTAS")
    ax_top.set_xscale("log", base=2)
    ax_top.set_yscale("log")
    ax_top.set_ylabel("Latency (ms, median)")
    ax_top.set_title("Inference latency vs batch size", fontsize=10)
    ax_top.legend()
    ax_top.grid(True, which="both", linestyle=":", alpha=0.4)

    ax_bot.plot(xs, ratio, color=_C_PTAS, marker="^", lw=2, ms=6)
    ax_bot.axhline(1.0, color="black", lw=0.8, linestyle="--")
    ax_bot.set_xscale("log", base=2)
    ax_bot.set_xlabel("Batch size")
    ax_bot.set_ylabel("Overhead ratio\n(PTAS / NN)")
    ax_bot.set_xticks(xs)
    ax_bot.set_xticklabels([str(b) for b in xs], fontsize=8)
    ax_bot.grid(True, which="both", linestyle=":", alpha=0.4)
    for x, r in zip(xs, ratio):
        ax_bot.text(x, r + 0.05, f"{r:.1f}×", ha="center", va="bottom",
                    fontsize=7, color=_C_PTAS)


def _plot_training(ax_top: plt.Axes, ax_bot: plt.Axes, train_results: dict):
    if not train_results:
        ax_top.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax_top.transAxes)
        return

    cond_names = list(train_results.keys())
    x = np.arange(len(cond_names))
    w = 0.32

    for off, key, color, lbl in [
        (-w / 2, "nn",   _C_NN,   "NN only"),
        ( w / 2, "ptas", _C_PTAS, "NN + PaTAS"),
    ]:
        means = [train_results[c][key][0] for c in cond_names]
        stds  = [train_results[c][key][1] for c in cond_names]
        bars  = ax_top.bar(x + off, means, w, color=color, label=lbl,
                           yerr=stds, capsize=4,
                           error_kw={"elinewidth": 1.2, "ecolor": "black"},
                           zorder=3)
        for bar, m in zip(bars, means):
            ax_top.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f"{m:.1f}s", ha="center", va="bottom", fontsize=7,
                        color=color)

    ax_top.set_xticks(x)
    ax_top.set_xticklabels(cond_names, fontsize=9)
    ax_top.set_ylabel(f"Total wall-clock ({N_BENCH_EPOCHS} epochs, s)")
    ax_top.set_title("Training latency across noise conditions", fontsize=10)
    ax_top.legend()
    ax_top.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)

    # Per-epoch overhead bars
    overhead = [
        (train_results[c]["ptas"][2] - train_results[c]["nn"][2]) * 1e3
        for c in cond_names
    ]
    bars2 = ax_bot.bar(x, overhead, 0.55, color=_C_PTAS, zorder=3)
    for bar, ov in zip(bars2, overhead):
        ax_bot.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{ov:.0f} ms", ha="center", va="bottom",
                    fontsize=7, color=_C_PTAS)
    ax_bot.axhline(0, color="black", lw=0.8)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(cond_names, fontsize=9)
    ax_bot.set_ylabel("PaTAS overhead\nper epoch (ms)")
    ax_bot.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def _latex_table(header, rows, caption, label) -> str:
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


def _write_tables(inf_results: dict, train_results: dict, tables_dir: str):
    # Inference table
    header_inf = [
        "Batch size",
        "NN latency (ms)",
        "PaTAS latency (ms)",
        r"Overhead ratio ($\times$)",
    ]
    rows_inf = []
    for bs in sorted(inf_results.keys()):
        nn_m, pt_m = inf_results[bs]
        rows_inf.append([
            str(bs),
            f"{nn_m:.3f}",
            f"{pt_m:.3f}",
            f"{pt_m / max(nn_m, 1e-9):.2f}",
        ])
    tex_inf = _latex_table(
        header_inf, rows_inf,
        caption=(
            r"Inference latency (median over " + str(N_INF_REPEATS) +
            r" repetitions) for plain NN forward pass vs PaTAS \texttt{apply\_feedforward}. "
            r"Overhead ratio = PaTAS / NN."
        ),
        label="tab:latency-inference",
    )

    # Training table
    header_tr = [
        "Condition",
        r"NN total (s)",
        r"PaTAS total (s)",
        r"NN / epoch (s)",
        r"PaTAS / epoch (s)",
        r"Overhead / epoch (ms)",
        r"Ratio ($\times$)",
    ]
    rows_tr = []
    for cond, v in train_results.items():
        nn_tot, nn_std, nn_ep = v["nn"]
        pt_tot, pt_std, pt_ep = v["ptas"]
        overhead_ms = (pt_ep - nn_ep) * 1e3
        ratio = pt_tot / max(nn_tot, 1e-9)
        rows_tr.append([
            cond,
            f"{nn_tot:.1f}$\\pm${nn_std:.1f}",
            f"{pt_tot:.1f}$\\pm${pt_std:.1f}",
            f"{nn_ep:.2f}",
            f"{pt_ep:.2f}",
            f"{overhead_ms:.0f}",
            f"{ratio:.2f}",
        ])
    tex_tr = _latex_table(
        header_tr, rows_tr,
        caption=(
            r"Training latency (mean $\pm$ std over " + str(N_TRAIN_TRIALS) +
            r" trials, " + str(N_BENCH_EPOCHS) +
            r" epochs) for plain NN vs PaTAS-augmented NN training. "
            r"Overhead / epoch = per-epoch wall-clock difference."
        ),
        label="tab:latency-training",
    )

    path_inf = os.path.join(tables_dir, "latency_inference.tex")
    path_tr  = os.path.join(tables_dir, "latency_training.tex")
    with open(path_inf, "w", encoding="utf-8") as fh:
        fh.write(tex_inf)
    with open(path_tr,  "w", encoding="utf-8") as fh:
        fh.write(tex_tr)
    print(f"  Saved {path_inf}")
    print(f"  Saved {path_tr}")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def latency_analysis(ds, output_dir: str):
    """Full latency analysis: inference + training benchmarks.

    Parameters
    ----------
    ds         : dataset object with .X_test, .y_test, .X_train, .y_train
    output_dir : root results directory (e.g. 'results_noise')
    """
    dis_dir    = os.path.join(output_dir, "dissertation")
    plots_dir  = os.path.join(dis_dir, "plots")
    tables_dir = os.path.join(dis_dir, "tables")
    os.makedirs(plots_dir,  exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Try to find a cached nn_weights.pkl (fn_0.00 clean run)
    nn_pkl = os.path.join(output_dir, "data",
                          "fn_0.00_ptas-cal-fb_r0", "nn_weights.pkl")

    # ---- Part A: Inference ----
    print("\n[A] Inference latency benchmark ...")
    inf_results = run_inference_benchmark(ds.X_test.astype(np.float32), nn_pkl)

    # ---- Part B: Training ----
    print("\n[B] Training latency benchmark ...")
    train_results = run_training_benchmark(ds)

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 7),
                             gridspec_kw={"height_ratios": [2, 1]})
    _plot_inference(axes[0, 0], axes[1, 0], inf_results)
    _plot_training( axes[0, 1], axes[1, 1], train_results)

    fig.suptitle(
        f"PaTAS latency overhead — inference (left) and training (right, {N_BENCH_EPOCHS} epochs)",
        fontsize=12,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(plots_dir, f"latency.{ext}")
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved {os.path.join(plots_dir, 'latency.pdf')}")

    # ---- Tables ----
    _write_tables(inf_results, train_results, tables_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from data_loader import load_5g_dataset, make_synthetic_5g

    parser = argparse.ArgumentParser(
        description="PaTAS latency benchmark for the 5G noise evaluation."
    )
    parser.add_argument("output_dir", nargs="?", default="results_noise",
                        help="Root results directory (default: results_noise)")
    parser.add_argument("--data-dir", default=None,
                        help="5G dataset directory (synthetic data used if omitted)")
    parser.add_argument("--inference-only", action="store_true",
                        help="Skip training benchmark (faster)")
    parser.add_argument("--training-only", action="store_true",
                        help="Skip inference benchmark")
    args = parser.parse_args()

    if args.data_dir is None:
        print("No data dir — using synthetic 5G dataset ...")
        data_dir = make_synthetic_5g(n_bs=20, n_hours=72, cells_per_bs=2, seed=0)
    else:
        data_dir = args.data_dir

    ds = load_5g_dataset(data_dir, n_classes=3, test_frac=0.2, seed=0)

    dis_dir    = os.path.join(args.output_dir, "dissertation")
    plots_dir  = os.path.join(dis_dir, "plots")
    tables_dir = os.path.join(dis_dir, "tables")
    os.makedirs(plots_dir,  exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    nn_pkl = os.path.join(args.output_dir, "data",
                          "fn_0.00_ptas-cal-fb_r0", "nn_weights.pkl")

    inf_results   = {}
    train_results = {}

    if not args.training_only:
        print("\n[A] Inference latency benchmark ...")
        inf_results = run_inference_benchmark(ds.X_test.astype(np.float32), nn_pkl)

    if not args.inference_only:
        print("\n[B] Training latency benchmark ...")
        train_results = run_training_benchmark(ds)

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 7),
                             gridspec_kw={"height_ratios": [2, 1]})
    _plot_inference(axes[0, 0], axes[1, 0], inf_results)
    _plot_training( axes[0, 1], axes[1, 1], train_results)
    fig.suptitle(
        f"PaTAS latency overhead — inference (left) and training (right, {N_BENCH_EPOCHS} epochs)",
        fontsize=12,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(plots_dir, f"latency.{ext}")
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved {os.path.join(plots_dir, 'latency.pdf')}")

    _write_tables(inf_results, train_results, tables_dir)
    print("Done.")
