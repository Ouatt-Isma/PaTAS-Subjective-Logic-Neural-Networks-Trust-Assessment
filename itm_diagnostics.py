"""itm_diagnostics.py — a-priori applicability diagnostics for input trust.

Fits the InputTrustModel on each dataset's own (clean) training features and
reports the statistics that predict whether input-conformity trust will be
informative there — BEFORE any model is trained or scored:

    * span            robust feature range (0.5th–99.5th percentile)
    * median σ_eff    typical effective per-feature deviation
    * max weight      largest reliability weight w = (σ_ref/σ)^α (capped)
    * top-decile      evidence share of the 10% most reliable features
    * strong frac.    fraction of features with w ≥ 4 (strong witnesses)

A dataset with spatially registered content (MNIST, Fashion-MNIST: dead
borders, tight marginals) concentrates evidence in reliable witnesses —
corruption and OOD inputs violate them and are detected.  Unregistered
imagery (GTSRB, CIFAR-10) has uniformly wide marginals: no reliable
witnesses exist, the weights stay flat, and marginal conformity is
uninformative — the empirical boundary case.  The fitted model therefore
self-diagnoses its applicability per dataset.

Outputs: console table + results/itm_diagnostics.{tex,json}.

Usage:  python itm_diagnostics.py            # all datasets with a data/ cache
        python itm_diagnostics.py --datasets mnist fashion
"""
from __future__ import annotations

import os
import sys
import json
import argparse

_v2_dir = os.path.dirname(os.path.abspath(__file__))
for _p in (_v2_dir, os.path.join(_v2_dir, "patas_module")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import patas_module  # noqa: F401 — path bootstrap

from input_trust import InputTrustModel

DATASETS = ["mnist", "fashion", "gtsrb", "cifar10gray"]
LABELS = {"mnist": "MNIST", "fashion": "Fashion-MNIST",
          "gtsrb": "GTSRB", "cifar10gray": "CIFAR-10 (grayscale)"}


def _load_train(name: str) -> np.ndarray:
    from NN import datasets as ds
    if name == "mnist":
        return ds.load_mnist()[0]
    if name == "fashion":
        return ds.load_fashion()[0]
    if name == "gtsrb":
        return ds.load_gtsrb()[0]
    if name == "cifar10gray":
        X = ds.load_cifar10()[0]
        return X.reshape(-1, 3, 32, 32).mean(axis=1).reshape(-1, 32 * 32)
    raise ValueError(name)


def diagnose(name: str, **itm_kwargs) -> dict:
    X = _load_train(name)
    itm = InputTrustModel(**itm_kwargs).fit(X)
    w = itm.weights
    return {
        "dataset": name,
        "n_features": int(X.shape[1]),
        "span": round(itm.span, 3),
        "median_sigma_eff": round(float(np.median(itm.sigma_eff)), 4),
        "max_weight": round(float(w.max()), 1),
        "top_decile_share": round(float(np.sort(w)[-len(w) // 10:].sum() / w.sum()), 3),
        "strong_witness_frac": round(float(np.mean(w >= 4.0)), 3),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    rows = []
    for name in args.datasets:
        try:
            rows.append(diagnose(name))
        except Exception as exc:                            # noqa: BLE001
            print(f"[itm-diag] {name}: skipped ({exc})")
    if not rows:
        raise SystemExit("No dataset could be loaded — warm the data/ caches first.")

    hdr = f"{'Dataset':<22} {'span':>7} {'med σ':>7} {'w_max':>6} {'top10%':>7} {'w≥4':>6}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in rows:
        print(f"{LABELS[r['dataset']]:<22} {r['span']:>7.3f} "
              f"{r['median_sigma_eff']:>7.4f} {r['max_weight']:>6.1f} "
              f"{r['top_decile_share']:>7.3f} {r['strong_witness_frac']:>6.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    lines = [
        r"\begin{table}[ht]", r"\centering",
        r"\caption{A-priori applicability diagnostics of input-conformity "
        r"trust: statistics of the InputTrustModel fitted on each dataset's "
        r"clean training features, before any model is trained. Registered "
        r"domains (MNIST, Fashion-MNIST) concentrate evidence in a minority "
        r"of tight-marginal ``reliable witness'' features (large maximum "
        r"weight, high top-decile evidence share), which corruption and "
        r"out-of-distribution inputs violate detectably; unregistered "
        r"imagery (GTSRB, CIFAR-10) has uniformly wide marginals — no "
        r"reliable witnesses — so marginal conformity is predicted (and, "
        r"for GTSRB, empirically confirmed) to be uninformative.}",
        r"\label{tab:itm-diagnostics}",
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"Dataset & Span & Median $\sigma_{\mathrm{eff}}$ & $w_{\max}$ & "
        r"Top-decile share & Frac.\ $w \geq 4$ \\", r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{LABELS[r['dataset']]} & {r['span']:.2f} & "
            f"{r['median_sigma_eff']:.3f} & {r['max_weight']:.1f} & "
            f"{r['top_decile_share']:.2f} & {r['strong_witness_frac']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = os.path.join(args.out_dir, "itm_diagnostics.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    with open(os.path.join(args.out_dir, "itm_diagnostics.json"), "w",
              encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nSaved {tex_path} and results/itm_diagnostics.json")


if __name__ == "__main__":
    main()
