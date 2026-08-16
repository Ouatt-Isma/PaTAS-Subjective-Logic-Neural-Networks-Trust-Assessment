"""eval_model_trust.py — PaTAS model-side trust as a label-free training audit.

The complement to run_uq_comparison.py's per-sample filter: this script
evaluates the claim that PaTAS's *model-side* trust — the propagated opinion
of the trained weight-opinion network under a fully-trusted input — tracks
the quality of the data the network was trained on, without needing any
labelled test set.

For a sweep of train-time label-flip rates p (clean features, labels flipped
with probability p), it trains/loads the base network and its PaTAS mirror
per rate and reports, per rate:

    * PaTAS feedforward trust       projected probability b + a·u of the
                                    aggregated output opinion under a fully
                                    trusted input (a = 0.5)
    * PaTAS depth-normalized trust  same, after undoing the geometric
                                    depth decay (comparable across depths)
    * mean IPTA path trust          per-sample activation-path trust,
                                    averaged over a test subset
    * mean softmax confidence       the label-free baseline: the network's
                                    own average max-softmax on the same
                                    subset
    * test accuracy                 ground truth (uses labels — the quantity
                                    the label-free signals should track)

The headline result is the detection margin, not a regression fit: PaTAS
trust acts as a corruption tripwire — it drops by a large margin at ANY
non-zero flip rate, including rates so low that accuracy and confidence
barely move — and it reads this from the training dynamics alone (the
feedforward trust uses no test data whatsoever).  Mean softmax confidence,
by contrast, tracks the flip rate almost linearly (cross-entropy on
p-flipped labels converges to confidence ≈ label purity), so it *measures*
corruption severity but only given a clean labelled test set to read it on,
and its moderate values are indistinguishable from a merely-hard task.
Both views (per-rate margins and correlations with accuracy) are reported.

Every label-noise rate gets its own cache directories (suffix _nl<p>), so
rates never collide; the p=0 row reuses the standard clean-condition caches
unchanged.

Usage
-----
    python eval_model_trust.py --dataset mnist --train-missing
    python eval_model_trust.py --dataset mnist --arch 512 --rates 0 0.15 0.3 0.45 --train-missing
    python eval_model_trust.py --dataset gtsrb --train-missing
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import multiprocessing

# ── Path bootstrap ────────────────────────────────────────────────────────────
_v2_dir = os.path.dirname(os.path.abspath(__file__))
_patas_dir = os.path.join(_v2_dir, "patas_module")
_tests_dir = os.path.join(_v2_dir, "tests")
for _p in (_v2_dir, _patas_dir, _tests_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import patas_module  # noqa: F401 — path bootstrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_uq_comparison import (
    DATASET_CFG, _arch_str, _DEFAULT_EPS,
    CLEAN_CONDITION, label_noise_condition,
    get_base_mlp, ensure_patas_cache, load_offline_ptas, patas_ipta_scores,
)


# ---------------------------------------------------------------------------
# Correlation helpers (scipy-free)
# ---------------------------------------------------------------------------

def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or x[ok].std() == 0 or y[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    return pearson(rx, ry)


# ---------------------------------------------------------------------------
# Per-rate evaluation
# ---------------------------------------------------------------------------

def model_feedforward_trust(ptas, input_dim: int) -> tuple[float, float]:
    """(raw, depth-normalized) projected probability b + 0.5·u of the
    aggregated output opinion under a fully trusted input — the model-side
    trust of the trained PaTAS network, no test data involved."""
    from NN.PTAStemplate import PTAS
    from concrete.TensorTO import TensorArrayTO, fill as tfill, to_numpy

    a = ptas.apply_feedforward(TensorArrayTO(tfill((1, input_dim), method="trust")))
    agg = to_numpy(PTAS.aggregation(a))
    norm = to_numpy(ptas.depth_normalized_aggregation(a))
    return (float(agg[0] + 0.5 * agg[2]), float(norm[0] + 0.5 * norm[2]))


def eval_rate(dataset: str, rate: float, args) -> dict:
    from NN.datasets import load_data
    from main import TRUST_TO_DATASET

    cond = CLEAN_CONDITION if rate == 0 else label_noise_condition(rate)
    cfgd = DATASET_CFG[dataset]
    arch = cfgd["arch"]

    print(f"\n{'='*70}\n  Model-trust audit — {dataset} ({_arch_str(arch)})  "
          f"label-flip rate p={rate:g}  ({cond.label})\n{'='*70}")

    x_how = TRUST_TO_DATASET.get(cond.x_trust, "clean")
    y_how = TRUST_TO_DATASET.get(cond.y_trust, "clean")
    _load_kwargs = {} if cond.noise_level is None else {"noise_level": cond.noise_level}
    X_train, X_test, y_train, y_test, _ = load_data(dataset, x_how, y_how,
                                                     **_load_kwargs)
    y_test_lbl = y_test.argmax(1)

    rng = np.random.default_rng(args.seed)
    n_sub = min(args.subset, len(X_test))
    idx = np.sort(rng.choice(len(X_test), size=n_sub, replace=False))
    Xs, ys = X_test[idx], y_test_lbl[idx]

    base = get_base_mlp(dataset, arch, X_train, y_train, X_test, y_test,
                        args.epochs, args.train_missing,
                        cond.x_trust, cond.y_trust,
                        noise_level=cond.noise_level,
                        force_retrain=args.force_retrain_all)

    row = {"rate": rate, "tag": cond.tag or "clean"}

    probs = base.forward(Xs)
    row["test_acc"] = float(np.mean(probs.argmax(1) == ys))
    row["mean_conf"] = float(probs.max(1).mean())
    row["conf_acc_gap"] = row["mean_conf"] - row["test_acc"]

    ptas = None
    if ensure_patas_cache(dataset, arch, args.eps, args.epochs,
                          args.train_missing, fuse_method=args.fuse_method,
                          x_trust=cond.x_trust, y_trust=cond.y_trust,
                          noise_level=cond.noise_level,
                          force_retrain=args.force_retrain_all):
        ptas = load_offline_ptas(dataset, arch, args.eps,
                                 fuse_method=args.fuse_method,
                                 x_trust=cond.x_trust, y_trust=cond.y_trust,
                                 noise_level=cond.noise_level)
    if ptas is None:
        print("  [ModelTrust] PaTAS cache unavailable — trust columns stay NaN.")
        row.update({"ff_trust": float("nan"), "ff_trust_norm": float("nan"),
                    "mean_path_trust": float("nan")})
        return row

    row["ff_trust"], row["ff_trust_norm"] = model_feedforward_trust(
        ptas, cfgd["input_dim"])

    n_ipta = min(args.ipta_subset, len(Xs))
    sc = patas_ipta_scores(ptas, base, Xs[:n_ipta], cfgd["input_dim"], itm=None)
    row["mean_path_trust"] = float(np.nanmean(sc["trust"]))

    print(f"  p={rate:g}: acc={row['test_acc']*100:.2f}%  "
          f"conf={row['mean_conf']:.4f}  ff_trust={row['ff_trust']:.4f}  "
          f"ff_norm={row['ff_trust_norm']:.4f}  "
          f"path_trust={row['mean_path_trust']:.4f}")
    return row


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

SIGNALS = [
    # key, plot label, colour, linestyle  (colours match uq_methods.METHOD_STYLE)
    ("ff_trust_norm",   "PaTAS trust (depth-norm.)", "#0072B2", "-"),
    ("ff_trust",        "PaTAS trust (raw)",         "#56B4E9", "-"),
    ("mean_path_trust", "PaTAS mean IPTA path trust","#004466", "--"),
    ("mean_conf",       "Mean softmax confidence",   "#555555", "-."),
    ("test_acc",        "Test accuracy",             "#000000", ":"),
]


def plot_rows(rows: list[dict], out_base: str, title: str) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 13,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150})
    rates = [r["rate"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for key, label, color, ls in SIGNALS:
        vals = [r.get(key, float("nan")) for r in rows]
        ax.plot(rates, vals, marker="o", ms=5, lw=2, color=color, ls=ls,
                label=label)
    ax.set_xlabel("Train-time label-flip rate $p$")
    ax.set_ylabel("Trust / confidence / accuracy")
    ax.set_xticks(rates)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(linestyle=":", alpha=0.35)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title(title)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_base}.pdf")


def add_margins(rows: list[dict]) -> None:
    """Relative drops vs. the clean (p=0) row, in place: how far each
    label-free signal falls below its own clean-model baseline."""
    base = next((r for r in rows if r["rate"] == 0), None)
    for r in rows:
        if base is None or r["rate"] == 0:
            r["trust_drop_rel"] = 0.0 if r["rate"] == 0 else float("nan")
            r["conf_drop_rel"] = 0.0 if r["rate"] == 0 else float("nan")
            continue
        t0, c0 = base.get("ff_trust"), base.get("mean_conf")
        r["trust_drop_rel"] = ((t0 - r["ff_trust"]) / t0
                               if t0 and np.isfinite(r.get("ff_trust", np.nan))
                               else float("nan"))
        r["conf_drop_rel"] = ((c0 - r["mean_conf"]) / c0
                              if c0 else float("nan"))


def write_table(rows: list[dict], correlations: dict, dataset: str,
                out_path: str) -> None:
    def _f(v, fmt="{:.3f}"):
        return fmt.format(v) if v is not None and np.isfinite(v) else "—"

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Training-quality audit on " + dataset.upper() +
        r": models trained under increasing train-time label-flip rates $p$ "
        r"(clean features). PaTAS model-side trust — the opinion of the "
        r"trained weight-opinion network under a fully trusted input — is "
        r"read from the training dynamics alone (no test data at all) and "
        r"drops by a large margin at any non-zero flip rate, acting as a "
        r"corruption tripwire even where accuracy barely moves. Mean "
        r"softmax confidence tracks the flip rate because cross-entropy on "
        r"flipped labels converges to the label purity, but reading it "
        r"requires a clean labelled test set, and a moderate value is "
        r"indistinguishable from a merely difficult task. "
        r"$\Delta_{\mathrm{rel}}$: relative drop vs.\ the $p{=}0$ model; "
        r"$r$/$\rho$: Pearson/Spearman correlation with test accuracy.}",
        rf"\label{{tab:model-trust-{dataset}}}",
        r"\begin{tabular}{cccccccc}",
        r"\toprule",
        r"$p$ & Test acc.\ (\%) & Mean conf. & PaTAS trust & "
        r"PaTAS trust (norm.) & Mean path trust & "
        r"$\Delta_{\mathrm{rel}}$ trust (\%) & "
        r"$\Delta_{\mathrm{rel}}$ conf.\ (\%) \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['rate']:g} & {_f(r['test_acc']*100, '{:.2f}')} & "
            f"{_f(r['mean_conf'])} & {_f(r['ff_trust'])} & "
            f"{_f(r['ff_trust_norm'])} & {_f(r['mean_path_trust'])} & "
            f"{_f(r['trust_drop_rel']*100, '{:.1f}')} & "
            f"{_f(r['conf_drop_rel']*100, '{:.1f}')} \\\\")
    c = correlations
    lines += [
        r"\midrule",
        rf"$r$ (vs.\ acc.) & — & {_f(c['mean_conf']['pearson'])} & "
        rf"{_f(c['ff_trust']['pearson'])} & {_f(c['ff_trust_norm']['pearson'])} & "
        rf"{_f(c['mean_path_trust']['pearson'])} & — & — \\",
        rf"$\rho$ (vs.\ acc.) & — & {_f(c['mean_conf']['spearman'])} & "
        rf"{_f(c['ff_trust']['spearman'])} & {_f(c['ff_trust_norm']['spearman'])} & "
        rf"{_f(c['mean_path_trust']['spearman'])} & — & — \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", choices=["mnist", "gtsrb"], default="mnist")
    p.add_argument("--arch", type=int, nargs="+", default=None,
                   help="Override MLP hidden dims, e.g. --arch 512")
    p.add_argument("--rates", type=float, nargs="+",
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                   help="Train-time label-flip rates (default 0..0.5)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--eps", type=float, default=_DEFAULT_EPS)
    p.add_argument("--fuse-method",
                   choices=["average", "cumulative", "weighted", "compromise", "constraint"],
                   default="average")
    p.add_argument("--subset", type=int, default=2000,
                   help="Test samples for accuracy/confidence (default 2000)")
    p.add_argument("--ipta-subset", type=int, default=500,
                   help="Samples for the per-sample IPTA path-trust mean "
                        "(default 500)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-missing", action="store_true")
    p.add_argument("--force-retrain-all", action="store_true",
                   help="Retrain base NN and PaTAS caches even if present")
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: 2 epochs, rates {0, 0.3}, small subsets")
    return p.parse_args()


def main():
    from uq_methods import print_device_info
    print_device_info()
    args = parse_args()
    if args.force_retrain_all:
        args.train_missing = True
    if args.quick:
        args.epochs = 2
        args.rates = [0.0, 0.3]
        args.subset = 300
        args.ipta_subset = 100
    dataset = args.dataset
    if args.arch:
        DATASET_CFG[dataset]["arch"] = tuple(args.arch)
    arch = DATASET_CFG[dataset]["arch"]

    fuse_suffix = "" if args.fuse_method == "average" else f"_fuse_{args.fuse_method}"
    out_dir = f"results/ModelTrust_{dataset}_{_arch_str(arch)}{fuse_suffix}"
    os.makedirs(out_dir, exist_ok=True)

    rows = [eval_rate(dataset, float(r), args) for r in sorted(set(args.rates))]
    add_margins(rows)

    acc = [r["test_acc"] for r in rows]
    correlations = {
        key: {"pearson": pearson([r.get(key) for r in rows], acc),
              "spearman": spearman([r.get(key) for r in rows], acc)}
        for key in ("ff_trust", "ff_trust_norm", "mean_path_trust", "mean_conf")
    }

    print(f"\n  {'p':>5} {'acc':>8} {'conf':>8} {'ff_trust':>9} "
          f"{'ff_norm':>8} {'path':>8} {'Δtrust%':>8} {'Δconf%':>8}")
    print("  " + "-" * 70)
    for r in rows:
        print(f"  {r['rate']:>5g} {r['test_acc']*100:7.2f}% "
              f"{r['mean_conf']:8.4f} {r['ff_trust']:9.4f} "
              f"{r['ff_trust_norm']:8.4f} {r['mean_path_trust']:8.4f} "
              f"{r['trust_drop_rel']*100:7.1f}% {r['conf_drop_rel']*100:7.1f}%")
    print("\n  Detection margin: PaTAS trust reads from training dynamics "
          "alone (no test data);\n  mean confidence needs a clean labelled "
          "test set. Δ% = relative drop vs. the p=0 model.")
    print("\n  Correlation with test accuracy across rates "
          "(Pearson / Spearman):")
    for key, c in correlations.items():
        print(f"    {key:<16} r={c['pearson']:+.3f}  ρ={c['spearman']:+.3f}")

    title = {"mnist": "MNIST", "gtsrb": "GTSRB"}[dataset]
    plot_rows(rows, os.path.join(out_dir, "model_trust_vs_labelnoise"),
              f"{title} — PaTAS model trust vs. training-label corruption")
    write_table(rows, correlations, dataset, os.path.join(out_dir, "table.tex"))
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"dataset": dataset, "arch": _arch_str(arch),
                   "eps": args.eps, "fuse_method": args.fuse_method,
                   "epochs": args.epochs, "subset": args.subset,
                   "ipta_subset": args.ipta_subset,
                   "rows": rows, "correlations": correlations}, fh, indent=2)
    print(f"  Saved {out_dir}/summary.json")
    print("\n=== Model-trust audit complete ===\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    # PaTAS scenario training spawns a PTAS-server + NN-client process pair;
    # 'fork' is incompatible with an initialized CUDA context (see
    # run_uq_comparison.py) — force 'spawn' everywhere.
    multiprocessing.set_start_method("spawn", force=True)
    main()
