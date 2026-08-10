"""ablation_fusion.py — Ablation study (b): the fusion operator.

The PaTAS trust revision fuses the gradient-derived evidence opinion into
each weight opinion (op_theta) and applies the learning-rate-discounted
update (update = binomial multiplication + fusion).  Both call sites use a
two-source belief-fusion operator; this ablation swaps it between:

    average    — Averaging Belief Fusion (ABF):  u = 2·u1·u2 / (u1+u2)
                 (dependent sources; the default used in the dissertation)
    cumulative — aleatory Cumulative Belief Fusion (CBF):
                 u = u1·u2 / (u1+u2−u1·u2)  (independent-evidence semantics;
                 accumulates evidence, so uncertainty vanishes faster)
    weighted   — Weighted Belief Fusion (WBF): confidence-weighted average

Each variant retrains PaTAS (the NN and its gradient stream are identical —
the model cache is shared — only the trust propagation differs) and is
cached under results/PTAS_Eval_..._fuse_<method>.  We report, per operator:

    aggregated (t,d,u) under trusted/vacuous/distrusted input (raw + depth-
    normalised), separation Δ, and per-sample IPTA filter AUROC / ECE.

Outputs (results/Ablation_fusion_<dataset>_<arch>/):
    ablation_fusion.pdf/.png     grouped bars (masses) + filter metrics
    ablation_fusion.tex          LaTeX table
    ablation_fusion.json         raw numbers

Usage
-----
    python ablation_fusion.py --dataset mnist --epochs 20
    python ablation_fusion.py --dataset mnist --quick     # smoke test
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import multiprocessing

_v2_dir = os.path.dirname(os.path.abspath(__file__))
for _p in (_v2_dir, os.path.join(_v2_dir, "patas_module"), os.path.join(_v2_dir, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import patas_module  # noqa: F401

from uq_methods import (plt, ece_from_confidence, aurc, roc_correctness)
from run_uq_comparison import (
    DATASET_CFG, _arch_str,
    ensure_patas_cache, load_offline_ptas, patas_ipta_scores, get_base_mlp,
)
from ablation_epsilon import aggregated_masses

FUSION_METHODS = ["average", "cumulative", "weighted"]

_FUSE_LABEL = {"average": "Averaging (ABF)",
               "cumulative": "Cumulative (CBF)",
               "weighted": "Weighted (WBF)"}
_MASS_COLORS = {"t": "#0072B2", "d": "#D55E00", "u": "#E69F00"}


def run(args) -> None:
    from NN.datasets import load_data

    dataset = args.dataset
    cfgd = DATASET_CFG[dataset]
    arch = cfgd["arch"]
    if isinstance(arch, str):
        raise SystemExit("fusion ablation uses the MLP datasets (mnist / gtsrb).")
    out_dir = f"results/Ablation_fusion_{dataset}_{_arch_str(arch)}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}\n  Fusion-operator ablation — {dataset} ({_arch_str(arch)})  "
          f"eps={args.eps}  epochs={args.epochs}\n{'='*70}\n")

    X_train, X_test, y_train, y_test, _ = load_data(dataset, "clean", "clean")
    y_test_lbl = y_test.argmax(1)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(X_test), size=min(args.subset, len(X_test)),
                             replace=False))
    Xs, ys = X_test[idx], y_test_lbl[idx]

    base = get_base_mlp(dataset, arch, X_train, y_train, X_test, y_test,
                        args.epochs, train_missing=True)
    # Two test conditions: clean, and feature-noised (labels stay clean)
    from uq_methods import apply_feature_noise
    Xs_noisy = apply_feature_noise(Xs, args.test_noise, seed=args.seed)
    conditions = [("clean", Xs, base.forward(Xs).argmax(1) == ys),
                  ("noisy", Xs_noisy, base.forward(Xs_noisy).argmax(1) == ys)]

    rows = []
    for fuse in args.fusions:
        print(f"\n──── fusion = {fuse} " + "─" * 45)
        ok = ensure_patas_cache(dataset, arch, args.eps, args.epochs,
                                train_missing=True, fuse_method=fuse)
        if not ok:
            print(f"  [{fuse}] PaTAS cache unavailable — skipped.")
            continue
        ptas = load_offline_ptas(dataset, arch, args.eps, fuse_method=fuse)
        masses = aggregated_masses(ptas, cfgd["input_dim"])
        row = {
            "fusion": fuse,
            "masses": masses,
            "separation": masses["trust"]["raw"][0] - masses["distrust"]["raw"][0],
        }
        for tag, Xc, correct_c in conditions:
            res = patas_ipta_scores(ptas, base, Xc, cfgd["input_dim"])
            scores = res["score"]        # trust-discounted confidence
            _, _, auroc = roc_correctness(scores, correct_c)
            row[f"auroc_{tag}"] = auroc
            row[f"aurc_{tag}"] = aurc(scores, correct_c)
            row[f"ece_{tag}"] = ece_from_confidence(scores, correct_c)
            row[f"mean_trust_{tag}"] = float(np.nanmean(res["trust"]))
        rows.append(row)
        print(f"  t(trusted)={masses['trust']['raw'][0]:.4f}  "
              f"u(vacuous)={masses['vacuous']['raw'][2]:.4f}  "
              f"mean path trust={row['mean_trust_clean']:.4f}  "
              f"ECE clean={row['ece_clean']:.3f}  "
              f"noisy(p={args.test_noise:g})={row['ece_noisy']:.3f}  "
              f"AUROC clean={row['auroc_clean']:.3f}")

    if not rows:
        raise SystemExit("No fusion variant produced results.")

    # ---- Figure: grouped bars (masses, trusted input) + filter metrics -----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(rows))
    labels = [_FUSE_LABEL[r["fusion"]] for r in rows]

    ax = axes[0]
    w = 0.26
    for k, (comp, ci) in enumerate((("t", 0), ("d", 1), ("u", 2))):
        vals = [r["masses"]["trust"]["raw"][ci] for r in rows]
        bars = ax.bar(x + (k - 1) * w, vals, width=w - 0.02,
                      color=_MASS_COLORS[comp], label=f"${comp}$")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8,
                    color=_MASS_COLORS[comp])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Aggregated opinion mass (trusted input)")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=3)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_title("Output assessment under fully trusted input")

    ax = axes[1]
    ax.plot(x, [r["auroc_clean"] for r in rows], "o-", color="#0072B2",
            lw=2, ms=7, label="AUROC clean $\\uparrow$")
    ax.plot(x, [r["auroc_noisy"] for r in rows], "o--", color="#0072B2",
            lw=1.6, ms=7, markerfacecolor="none",
            label=f"AUROC noise $p={args.test_noise:g}$")
    ax.plot(x, [r["ece_clean"] for r in rows], "s-", color="#D55E00",
            lw=2, ms=7, label="ECE clean $\\downarrow$")
    ax.plot(x, [r["ece_noisy"] for r in rows], "s--", color="#D55E00",
            lw=1.6, ms=7, markerfacecolor="none",
            label=f"ECE noise $p={args.test_noise:g}$")
    for xi, r in zip(x, rows):
        ax.annotate(f"{r['auroc_clean']:.3f}", xy=(xi, r["auroc_clean"]),
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=9, color="#0072B2")
        ax.annotate(f"{r['ece_clean']:.3f}", xy=(xi, r["ece_clean"]),
                    xytext=(0, -14), textcoords="offset points", ha="center",
                    fontsize=9, color="#D55E00")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_title("Per-sample IPTA filter quality")

    fig.suptitle(f"Fusion-operator ablation — {dataset.upper()} "
                 f"({_arch_str(arch)}, $\\epsilon={args.eps}$)", fontsize=13)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"ablation_fusion.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved {out_dir}/ablation_fusion.pdf")

    # ---- LaTeX table --------------------------------------------------------
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Ablation over the belief-fusion operator used in the PaTAS "
        r"trust revision (" + dataset.upper() + r", "
        + _arch_str(arch).replace("_", "--")
        + r" MLP, trust/trust, $\epsilon=" + str(args.eps) + r"$). "
        r"$(t,d,u)$ is the aggregated output opinion under fully trusted "
        r"input; $\Delta = t_{\mathrm{tr}} - t_{\mathrm{dis}}$; AUROC/ECE "
        r"refer to the PaTAS filter (trust-discounted confidence "
        r"$t_{\mathrm{path}}\cdot\hat p$), evaluated on clean test data "
        r"and under test-time feature noise ($p="
        + f"{args.test_noise:g}" + r"$).}",
        rf"\label{{tab:ablation-fusion-{dataset}}}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"Fusion operator & $t$ & $d$ & $u$ & $\Delta$ & "
        r"AUROC$_{\mathrm{clean}}$ & AUROC$_{\mathrm{noise}}$ & "
        r"ECE$_{\mathrm{clean}}$ & ECE$_{\mathrm{noise}}$ \\",
        r"\midrule",
    ]
    for r in rows:
        t_, d_, u_ = r["masses"]["trust"]["raw"]
        lines.append(
            f"{_FUSE_LABEL[r['fusion']]} & {t_:.4f} & {d_:.4f} & {u_:.4f} & "
            f"{r['separation']:.4f} & {r['auroc_clean']:.3f} & "
            f"{r['auroc_noisy']:.3f} & {r['ece_clean']:.3f} & "
            f"{r['ece_noisy']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = os.path.join(out_dir, "ablation_fusion.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  Saved {tex_path}")

    with open(os.path.join(out_dir, "ablation_fusion.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"dataset": dataset, "arch": _arch_str(arch), "eps": args.eps,
                   "epochs": args.epochs, "subset": int(len(Xs)),
                   "test_noise": args.test_noise,
                   "rows": rows}, fh, indent=2)
    print(f"  Saved {out_dir}/ablation_fusion.json")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", choices=["mnist", "gtsrb"], default="mnist")
    p.add_argument("--fusions", nargs="+", choices=FUSION_METHODS,
                   default=FUSION_METHODS)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--subset", type=int, default=1000)
    p.add_argument("--test-noise", type=float, default=0.3,
                   help="Feature-noise probability for the noised test "
                        "condition (default 0.3)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: 2 epochs, 200-sample subset")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.epochs = 2
        args.subset = 200
    run(args)
    print("\n=== Fusion ablation complete ===\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
