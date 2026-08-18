"""eval_poisoned_conformity.py — backdoor detection WITHOUT oracle patch
knowledge (reviewer W7).

The paper's Table IV shows that IPTA separates patched from clean inputs
when the patch pixels are explicitly distrusted, which requires knowing the
trigger. This experiment closes the loop between the poisoning study and
the comparative evaluation: the conformity-based input opinions of
Experiment 4 (fit on clean training features, no knowledge of the trigger)
are fed through the poisoned model's IPTA. Because the trigger patch sits
on near-constant border pixels, exactly the reliable witnesses the
conformity model weights hardest, patched inputs should be flagged
automatically.

Per sample set (clean 3, clean 6, patched 3, patched 6) and per input-
opinion mode (fully trusted, conformity, oracle patch-distrust), reports
the mean IPTA opinion of the true-class output neuron and the per-sample
input trust, plus the AUROC of patched-vs-clean detection from the
conformity-based scores alone.

Usage:
    python eval_poisoned_conformity.py                # patch 4, arch 128
    python eval_poisoned_conformity.py --patch-size 4 --subset 400
"""
from __future__ import annotations

import os
import sys
import io
import json
import argparse
import contextlib

_v2_dir = os.path.dirname(os.path.abspath(__file__))
for _p in (_v2_dir, os.path.join(_v2_dir, "patas_module"), os.path.join(_v2_dir, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import patas_module  # noqa: F401

from input_trust import InputTrustModel


def load_poisoned_artifacts(patch_size: int, hidden: int, eps: float):
    import pickle
    from main import nn_cache_dir, ptas_cache_dir
    from NN.primaryNN import NeuralNetwork
    from NN.PTAStemplate import PTAS
    from concrete.TensorTO import TensorArrayTO

    nn_dir = nn_cache_dir("mnist", str(hidden), "trust", "trust", patch=patch_size)
    ptas_dir = ptas_cache_dir("mnist", str(hidden), "trust", "trust", eps,
                              patch=patch_size)
    nn_path = os.path.join(nn_dir, "nn_model.pkl")
    omega_path = os.path.join(ptas_dir, "omega_arrays.pkl")
    for p in (nn_path, omega_path):
        if not os.path.exists(p):
            raise SystemExit(f"Missing cache {p} — run the poisoned experiment "
                             f"(tests/test_mnist_poisoned.py) first.")
    nn = NeuralNetwork(28 * 28, hidden_sizes=[hidden], output_size=10,
                       ptas=False, operation=True)
    nn.load_model(nn_path)
    with open(omega_path, "rb") as fh:
        omegas = pickle.load(fh)
    ptas = PTAS([TensorArrayTO(a) for a in omegas], operator_mapping=None,
                nn_interface=None, trust_assessment_func=None,
                structure=[28 * 28, hidden, 10], epsilon_low=eps, eval=False)
    return nn, ptas


def ipta_class_opinions(ptas, nn, X: np.ndarray, cls: int, itm=None,
                        oracle_patch: int | None = None) -> np.ndarray:
    """Per-sample IPTA opinion of the class-``cls`` output neuron under the
    selected input-opinion mode. Returns (n, 3)."""
    from concrete.TensorTO import TensorArrayTO, fill as tfill

    if itm is not None:
        ops = itm.opinions(X)
    out = np.full((len(X), 3), np.nan)
    silent = io.StringIO()
    for i in range(len(X)):
        with contextlib.redirect_stdout(silent):
            _, path = nn.forward(X[i:i + 1], getactivated=True)
            ipta = ptas.GenIPTA(path)
            if itm is not None:
                Tx = TensorArrayTO(ops[i:i + 1])
            else:
                v = tfill((1, 28 * 28), method="trust")
                if oracle_patch:
                    for r in range(oracle_patch):
                        for c in range(oracle_patch):
                            v[0, 28 * r + c] = (0.0, 1.0, 0.0)
                Tx = TensorArrayTO(v)
            Ty = ipta(Tx)
        out[i] = np.asarray(Ty.to_numpy())[0, cls]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--patch-size", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--subset", type=int, default=400,
                    help="Samples per class subset (default 400)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from NN.datasets import load_mnist, add_trigger_patch, mnist_get_scaling
    from sklearn.metrics import roc_auc_score

    X_train, X_test, y_train, y_test = load_mnist()
    itm = InputTrustModel().fit(X_train)
    print(f"[W7] {itm.describe()}")
    nn, ptas = load_poisoned_artifacts(args.patch_size, args.hidden, args.eps)

    rng = np.random.default_rng(args.seed)
    scaled = mnist_get_scaling(1.0)
    sets = {}
    for cls in (3, 6):
        idx = np.where(y_test == cls)[0]
        idx = idx[rng.permutation(len(idx))[:args.subset]]
        clean = X_test[idx]
        patched = np.stack([add_trigger_patch(x, patch_value=scaled,
                                              patch_size=args.patch_size)
                            for x in clean])
        sets[f"clean{cls}"] = (clean, cls)
        sets[f"patched{cls}"] = (patched, cls)

    results = {}
    for name, (X, cls) in sets.items():
        row = {"n": len(X),
               "acc": float(np.mean(nn.predict(X) == cls)),
               "input_trust": float(itm.sample_trust(X).mean())}
        for mode, kw in (("trusted", {}), ("conformity", {"itm": itm}),
                         ("oracle", {"oracle_patch": args.patch_size})):
            ops = ipta_class_opinions(ptas, nn, X, cls, **kw)
            pp = ops[:, 0] + 0.5 * ops[:, 2]
            row[mode] = {"b": round(float(ops[:, 0].mean()), 4),
                         "d": round(float(ops[:, 1].mean()), 4),
                         "u": round(float(ops[:, 2].mean()), 4),
                         "pp_mean": round(float(pp.mean()), 4)}
            row[f"_pp_{mode}"] = pp
        results[name] = row
        print(f"{name:10s} acc={row['acc']*100:6.2f}%  "
              f"input_trust={row['input_trust']:.4f}  "
              + "  ".join(f"{m}: ({row[m]['b']:.3f},{row[m]['d']:.3f},"
                          f"{row[m]['u']:.3f})" for m in
                          ("trusted", "conformity", "oracle")))

    # Detection: patched vs clean per class, from conformity-based signals
    # only (no oracle knowledge anywhere).
    det = {}
    for cls in (3, 6):
        c, p = results[f"clean{cls}"], results[f"patched{cls}"]
        lab = np.r_[np.ones(c["n"]), np.zeros(p["n"])]
        det[f"ipta_conformity_{cls}"] = round(float(roc_auc_score(
            lab, np.r_[c["_pp_conformity"], p["_pp_conformity"]])), 4)
        Xc, Xp = sets[f"clean{cls}"][0], sets[f"patched{cls}"][0]
        det[f"input_trust_{cls}"] = round(float(roc_auc_score(
            lab, np.r_[itm.sample_trust(Xc), itm.sample_trust(Xp)])), 4)
    print("\nPatched-vs-clean detection AUROC (no oracle):")
    for k, v in det.items():
        print(f"  {k:24s} {v:.4f}")

    out_dir = "results/PoisonedConformity_mnist"
    os.makedirs(out_dir, exist_ok=True)
    payload = {"patch_size": args.patch_size, "hidden": args.hidden,
               "eps": args.eps, "subset": args.subset,
               "itm": itm.describe(), "detection": det,
               "sets": {k: {kk: vv for kk, vv in v.items()
                            if not kk.startswith("_")}
                        for k, v in results.items()}}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved {out_dir}/summary.json")


if __name__ == "__main__":
    main()
