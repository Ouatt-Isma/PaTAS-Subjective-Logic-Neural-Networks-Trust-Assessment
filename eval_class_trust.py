"""eval_class_trust.py — per-class output trust from cached poisoned runs.

Reviewer question (round 3, minors): the poisoned-MNIST experiment reports
the trust drop for class 6; is the drop specific to the poisoned pair, and
is it outside the natural variability of the clean classes?  Both flipped
classes (the 6<->9 backdoor pair) should sit below the clean-class band.

Reads the cached at.pkl (per-class output opinions under a fully trusted
input) of every results/PTAS_Eval_*_PathSize_* directory (all patch sizes,
including the pm<mode> single-channel controls) plus the clean-model
reference (PathSize_None), and reports per run:

    per-class projected trust  p_c = b_c + 0.5 u_c
    clean-class band           mean +/- std over classes not in the flipped
                               pair, and the minimum clean-class trust
    z-scores                   (p_6 - mean_clean) / std_clean, same for 9

Writes results/ClassTrust_poisoned/summary.json and table.tex.

Usage:  python eval_class_trust.py [--glob "results/PTAS_Eval_mnist_*"]
"""
from __future__ import annotations

import os
import sys
import json
import glob
import pickle
import argparse

_v2_dir = os.path.dirname(os.path.abspath(__file__))
_patas_dir = os.path.join(_v2_dir, "patas_module")
for _p in (_v2_dir, _patas_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import patas_module  # noqa: F401 — path bootstrap

POIS_PAIR = (6, 9)      # the flipped backdoor pair (flip_map {6: 9, 9: 6})


def read_class_trusts(at_path: str) -> np.ndarray | None:
    """Per-class projected trust b + 0.5*u from a cached at.pkl."""
    try:
        with open(at_path, "rb") as fh:
            at = pickle.load(fh)
    except Exception as e:                                  # noqa: BLE001
        print(f"  [skip] {at_path}: {e}")
        return None
    v = np.asarray(at.to_numpy() if hasattr(at, "to_numpy") else at.value,
                   dtype=np.float64)
    v = v.reshape(-1, v.shape[-2], 3)[0]                    # (classes, 3)
    return v[:, 0] + 0.5 * v[:, 2]


def analyze(name: str, p: np.ndarray) -> dict:
    n_classes = len(p)
    clean = np.array([c for c in range(n_classes) if c not in POIS_PAIR])
    mu, sd = float(p[clean].mean()), float(p[clean].std())
    row = {
        "run": name,
        "per_class": p.round(4).tolist(),
        "clean_mean": mu, "clean_std": sd,
        "clean_min": float(p[clean].min()),
        "trust_6": float(p[POIS_PAIR[0]]), "trust_9": float(p[POIS_PAIR[1]]),
        "z_6": (p[POIS_PAIR[0]] - mu) / sd if sd > 0 else float("nan"),
        "z_9": (p[POIS_PAIR[1]] - mu) / sd if sd > 0 else float("nan"),
    }
    row["below_clean_min_6"] = bool(row["trust_6"] < row["clean_min"])
    row["below_clean_min_9"] = bool(row["trust_9"] < row["clean_min"])
    return row


def write_table(rows: list[dict], out_path: str) -> None:
    def _f(v, fmt="{:.3f}"):
        return fmt.format(v) if v is not None and np.isfinite(v) else "--"

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Per-class output trust (projected probability of the "
        r"class's aggregated output opinion under a fully trusted input) "
        r"for the poisoned-MNIST runs: both flipped classes of the "
        r"$6\leftrightarrow 9$ backdoor pair against the band of the eight "
        r"clean classes (mean $\pm$ std and minimum). $z$: distance from "
        r"the clean-class mean in clean-class standard deviations. "
        r"pm-suffixed runs are the single-channel controls (label flip "
        r"only / trigger patch only).}",
        r"\label{tab:classtrust}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Run & Clean mean $\pm$ std & Clean min & Trust$_6$ & $z_6$ & "
        r"Trust$_9$ & $z_9$ \\",
        r"\midrule",
    ]
    for r in rows:
        run_tex = r["run"].replace("_", "\\_")
        lines.append(
            f"{run_tex} & "
            f"{_f(r['clean_mean'])} $\\pm$ {_f(r['clean_std'])} & "
            f"{_f(r['clean_min'])} & {_f(r['trust_6'])} & "
            f"{_f(r['z_6'], '{:+.1f}')} & {_f(r['trust_9'])} & "
            f"{_f(r['z_9'], '{:+.1f}')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  Saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--glob", default="results/PTAS_Eval_mnist_*",
                    help="Cache directories to scan for at.pkl")
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(args.glob)):
        at_path = os.path.join(d, "at.pkl")
        if not os.path.exists(at_path):
            continue
        p = read_class_trusts(at_path)
        if p is None or len(p) != 10:
            continue
        name = os.path.basename(d).replace("PTAS_Eval_mnist_", "")
        rows.append(analyze(name, p))

    if not rows:
        print("No at.pkl found under", args.glob)
        return

    print(f"\n  {'run':<44} {'clean mean±std':>16} {'min':>7} "
          f"{'t6':>7} {'z6':>6} {'t9':>7} {'z9':>6}")
    print("  " + "-" * 96)
    for r in rows:
        print(f"  {r['run']:<44} {r['clean_mean']:.3f} ± {r['clean_std']:.3f}"
              f"   {r['clean_min']:7.3f} {r['trust_6']:7.3f} "
              f"{r['z_6']:+6.1f} {r['trust_9']:7.3f} {r['z_9']:+6.1f}")

    out_dir = "results/ClassTrust_poisoned"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"pois_pair": POIS_PAIR, "rows": rows}, fh, indent=2)
    write_table(rows, os.path.join(out_dir, "table.tex"))
    print(f"  Saved {out_dir}/summary.json")


if __name__ == "__main__":
    main()
