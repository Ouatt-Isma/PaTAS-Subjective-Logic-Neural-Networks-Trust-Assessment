"""ablation_epsilon.py — Ablation study (a): the ε threshold.

ε (epsilon_low) is the gradient-stability threshold in the PaTAS trust
revision: a weight update with |Δw| < ε counts as positive evidence
("the weight has converged / is stable") in theta_given_y, |Δw| ≥ ε as
negative evidence.  The ablation makes the choice of ε systematic:

* ε → 0   : every update looks unstable → evidence is all negative →
            the assessment collapses towards distrust; the trust-discounted
            confidence (PaTAS filter score t_path · p_softmax) becomes
            grossly UNDER-confident → large ECE.
* ε → ∞   : every update looks stable → uniform positive evidence →
            trust saturates at 1 and the filter degenerates to the raw
            softmax confidence, inheriting its OVER-confidence → ECE rises
            again (visibly so under test-time noise).
* in-between: the trust level matches the network's empirical reliability;
            we select ε that minimises the mean ECE of the trust-discounted
            confidence over the clean AND the feature-noised test condition,
            and report the aggregate masses to show the two saturation
            regimes.  (AUROC is reported for completeness but barely depends
            on ε: the path trust is nearly a per-model constant, so the *ranking*
            of the discounted confidence is essentially that of the softmax.)

For each ε the PaTAS scenario (trust/trust) is trained — cached per-ε under
results/PTAS_Eval_<ds>_<arch>_trust_trust_eps_<ε>_PathSize_None — and we
record:

    t / d / u aggregated output opinion for trusted, vacuous and distrusted
    input (raw and depth-normalised), and AUROC / AURC / ECE of the
    per-sample IPTA filter on a fixed test subset.

Outputs (results/Ablation_epsilon_<dataset>_<arch>/):
    ablation_epsilon.pdf/.png    masses + filter AUROC vs ε (log axis)
    ablation_epsilon.tex         LaTeX table
    ablation_epsilon.json        raw numbers

Usage
-----
    python ablation_epsilon.py --dataset mnist --epochs 20
    python ablation_epsilon.py --dataset mnist --eps 0.001 0.01 0.05 0.1 0.5 1.0
    python ablation_epsilon.py --dataset mnist --quick     # smoke test
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
    DATASET_CFG, _lr_for, _arch_str,
    ensure_patas_cache, load_offline_ptas, patas_ipta_scores, get_base_mlp,
)

_DEFAULT_EPS_GRID = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

_MASS_COLORS = {"t": "#0072B2", "d": "#D55E00", "u": "#E69F00"}


def aggregated_masses(ptas, input_dim: int) -> dict:
    """Aggregated (t,d,u) for trusted / vacuous / distrusted input, raw + norm."""
    from concrete.TensorTO import TensorArrayTO, fill as tfill, to_numpy
    from NN.PTAStemplate import PTAS as PTASClass

    out = {}
    for profile in ("trust", "vacuous", "distrust"):
        a = ptas.apply_feedforward(TensorArrayTO(tfill((1, input_dim), method=profile)))
        raw = to_numpy(PTASClass.aggregation(a)).tolist()
        norm = to_numpy(ptas.depth_normalized_aggregation(a)).tolist()
        out[profile] = {"raw": raw, "norm": norm}
    return out


def run(args) -> None:
    from NN.datasets import load_data

    dataset = args.dataset
    cfgd = DATASET_CFG[dataset]
    arch = cfgd["arch"]
    if isinstance(arch, str):
        raise SystemExit("epsilon ablation uses the MLP datasets (mnist / gtsrb); "
                         "IPTA is not defined for the conv architecture.")
    out_dir = f"results/Ablation_epsilon_{dataset}_{_arch_str(arch)}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}\n  ε ablation — {dataset} ({_arch_str(arch)})  "
          f"grid={args.eps}  epochs={args.epochs}\n{'='*70}\n")

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
    for eps in args.eps:
        print(f"\n──── ε = {eps} " + "─" * 50)
        ok = ensure_patas_cache(dataset, arch, eps, args.epochs, train_missing=True)
        if not ok:
            print(f"  [ε={eps}] PaTAS cache unavailable — skipped.")
            continue
        ptas = load_offline_ptas(dataset, arch, eps)
        masses = aggregated_masses(ptas, cfgd["input_dim"])
        row = {
            "eps": eps,
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
        row["ece_mean"] = float(np.nanmean([row["ece_clean"],
                                            row["ece_noisy"]]))
        rows.append(row)
        print(f"  t(trusted)={masses['trust']['raw'][0]:.4f}  "
              f"d(distrusted)={masses['distrust']['raw'][1]:.4f}  "
              f"mean path trust={row['mean_trust_clean']:.4f}  "
              f"ECE clean={row['ece_clean']:.3f}  "
              f"noisy(p={args.test_noise:g})={row['ece_noisy']:.3f}  "
              f"AUROC clean={row['auroc_clean']:.3f}")

    if not rows:
        raise SystemExit("No ε value produced results.")

    best = min(rows, key=lambda r: (r["ece_mean"]
                                    if np.isfinite(r["ece_mean"]) else np.inf))
    print(f"\n  ► Recommended ε = {best['eps']}  (min mean ECE of the "
          f"trust-discounted confidence over clean+noisy = {best['ece_mean']:.3f})")

    # ---- Figure: masses (left) + filter quality (right) --------------------
    eps_v = [r["eps"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    # distinct markers/linestyles: t and d often coincide numerically
    # (symmetric propagation), so pure overplotting would hide one of them
    for comp, ci, fmt, ms, lw, lbl in (
            ("t", 0, "o-",  9, 2.6, "Trust $t$ (trusted input)"),
            ("d", 1, "s--", 6, 1.8, "Distrust $d$ (distrusted input)"),
            ("u", 2, "^:",  6, 1.8, "Uncertainty $u$ (vacuous input)")):
        profile = {"t": "trust", "d": "distrust", "u": "vacuous"}[comp]
        vals = [r["masses"][profile]["raw"][ci] for r in rows]
        ax.plot(eps_v, vals, fmt, color=_MASS_COLORS[comp], lw=lw, ms=ms,
                markerfacecolor="none" if comp == "t" else _MASS_COLORS[comp],
                label=lbl)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\epsilon$ (gradient-stability threshold)")
    ax.set_ylabel("Aggregated opinion mass")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(linestyle=":", alpha=0.35)
    ax.set_title("Aggregated assessment vs. $\\epsilon$")

    ax = axes[1]
    ax.plot(eps_v, [r["ece_clean"] for r in rows], "o-", color="#0072B2",
            lw=2, ms=5, label="ECE (clean test)")
    ax.plot(eps_v, [r["ece_noisy"] for r in rows], "s--", color="#D55E00",
            lw=2, ms=5, label=f"ECE (feature noise $p={args.test_noise:g}$)")
    ax.axvline(best["eps"], color="#555555", lw=1, ls="--")
    ax.annotate(f"$\\epsilon^*={best['eps']}$",
                xy=(best["eps"], best["ece_mean"]), xytext=(5, -12),
                textcoords="offset points", fontsize=10, color="#555555")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\epsilon$ (gradient-stability threshold)")
    ax.set_ylabel("ECE of trust-discounted confidence $\\downarrow$")
    ax.grid(linestyle=":", alpha=0.35)
    ax.legend()
    ax.set_title("Filter calibration vs. $\\epsilon$")

    fig.suptitle(f"$\\epsilon$ ablation — {dataset.upper()} ({_arch_str(arch)})",
                 fontsize=13)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"ablation_epsilon.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir}/ablation_epsilon.pdf")

    # ---- LaTeX table --------------------------------------------------------
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Ablation over the gradient-stability threshold $\epsilon$ ("
        + dataset.upper() + r", " + _arch_str(arch).replace("_", "--")
        + r" MLP, trust/trust). $t_{\mathrm{tr}}$ = aggregated trust under fully "
        r"trusted input, $d_{\mathrm{dis}}$ = aggregated distrust under fully "
        r"distrusted input, $\Delta$ = separation $t_{\mathrm{tr}} - "
        r"t_{\mathrm{dis}}$. AUROC/ECE refer to the PaTAS filter "
        r"(trust-discounted confidence $t_{\mathrm{path}}\cdot\hat p$), "
        r"evaluated on clean test data and under test-time feature noise "
        r"($p=" + f"{args.test_noise:g}" + r"$). "
        r"Small $\epsilon$ collapses trust and makes the filter "
        r"under-confident; large $\epsilon$ saturates trust and reduces the "
        r"filter to the over-confident softmax; $\epsilon^{*}="
        + str(best["eps"]) + r"$ minimises the mean ECE over both conditions.}",
        rf"\label{{tab:ablation-epsilon-{dataset}}}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"$\epsilon$ & $t_{\mathrm{tr}}$ & $d_{\mathrm{dis}}$ & $\Delta$ & "
        r"AUROC$_{\mathrm{clean}}$ & AUROC$_{\mathrm{noise}}$ & "
        r"ECE$_{\mathrm{clean}}$ & ECE$_{\mathrm{noise}}$ \\",
        r"\midrule",
    ]
    for r in rows:
        star = r"$^{*}$" if r["eps"] == best["eps"] else ""
        lines.append(
            f"{r['eps']}{star} & {r['masses']['trust']['raw'][0]:.4f} & "
            f"{r['masses']['distrust']['raw'][1]:.4f} & {r['separation']:.4f} & "
            f"{r['auroc_clean']:.3f} & {r['auroc_noisy']:.3f} & "
            f"{r['ece_clean']:.3f} & {r['ece_noisy']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = os.path.join(out_dir, "ablation_epsilon.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  Saved {tex_path}")

    with open(os.path.join(out_dir, "ablation_epsilon.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"dataset": dataset, "arch": _arch_str(arch),
                   "epochs": args.epochs, "subset": int(len(Xs)),
                   "test_noise": args.test_noise,
                   "recommended_eps": best["eps"], "rows": rows}, fh, indent=2)
    print(f"  Saved {out_dir}/ablation_epsilon.json")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", choices=["mnist", "gtsrb"], default="mnist")
    p.add_argument("--eps", type=float, nargs="+", default=_DEFAULT_EPS_GRID)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--subset", type=int, default=1000)
    p.add_argument("--test-noise", type=float, default=0.3,
                   help="Feature-noise probability for the noised test "
                        "condition (default 0.3)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: 2 epochs, 200-sample subset, 3 ε values")
    return p.parse_args()


def main():
    from uq_methods import print_device_info
    print_device_info()
    args = parse_args()
    if args.quick:
        args.epochs = 2
        args.subset = 200
        if len(args.eps) == len(_DEFAULT_EPS_GRID):
            args.eps = [0.001, 0.05, 1.0]
    run(args)
    print("\n=== ε ablation complete ===\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
