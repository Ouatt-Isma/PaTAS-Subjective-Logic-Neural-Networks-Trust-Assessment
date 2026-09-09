"""influence_baseline.py — what a practitioner would do instead.

Provenance-guided parameter pruning has to beat the two things a defender
with the same knowledge would actually try, and one standard method from the
literature:

  drop-source     discard the untrusted source and retrain.  The obvious
                  move.  It costs the benign data that source also carried,
                  and it costs a full retraining run.
  influence       rank training samples by self-influence at the converged
                  model (TracIn's diagonal term, exactly
                  ||a^(l)||^2 * ||dz^(l)||^2 summed over layers), drop the
                  most influential, retrain.  This is the method a reviewer
                  will assume should have been used.
  random          drop the same number of samples at random, retrain.  The
                  control that separates a real ranking from the effect of
                  simply training on less data.
  prune           this work: prune the parameters whose attributed trust is
                  lowest.  No retraining, and the untrusted source's benign
                  data is kept.

Every arm gets the same knowledge: which source is untrusted, and nothing
about the trigger, the poisoned samples or the targeted classes.

Usage
-----
    python influence_baseline.py --dataset mnist --arch 128 --poisoned-patch 4
"""
from __future__ import annotations

import os
import sys
import json
import pickle
import argparse

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.join(_here, "patas_module"), os.path.join(_here, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np


def self_influence(Ws, bs, X, Y, batch=512):
    """TracIn self-influence at the converged model.

    For a dense layer the per-sample gradient is the outer product
    a_prev (x) dz, whose squared Frobenius norm is ||a_prev||^2 ||dz||^2, so
    the score is exact without ever materialising a per-sample gradient.
    """
    L = len(Ws)
    out = np.zeros(len(X), np.float64)
    for s0 in range(0, len(X), batch):
        xb, yb = X[s0:s0 + batch], Y[s0:s0 + batch]
        acts, zs, a = [xb], [], xb
        for l in range(L):
            z = a @ Ws[l] + bs[l]
            zs.append(z)
            if l == L - 1:
                e = np.exp(z - z.max(1, keepdims=True)); a = e / e.sum(1, keepdims=True)
            else:
                a = np.maximum(z, 0.0)
            acts.append(a)
        dz = acts[-1] - yb
        acc = np.zeros(len(xb), np.float64)
        for l in range(L - 1, -1, -1):
            an = (acts[l] ** 2).sum(1)          # ||a_prev||^2
            dn = (dz ** 2).sum(1)               # ||dz||^2
            acc += an * dn + dn                 # weights + bias
            if l > 0:
                dz = (dz @ Ws[l].T) * (zs[l - 1] > 0)
        out[s0:s0 + batch] = acc
    return out


def train_model(X, Y, X_test, Y_test, arch, epochs, seed, path):
    """Train (or reuse) one network with the framework's own trainer."""
    if os.path.exists(path):
        with open(path, "rb") as fh:
            wd = pickle.load(fh)
    else:
        os.environ["PATAS_SEED"] = str(seed)
        from NN.primaryNN import NeuralNetwork
        from main import get_lr_mnist
        nn = NeuralNetwork(X.shape[1], hidden_sizes=list(arch), output_size=Y.shape[1],
                           ptas=False, operation=False)
        nn.train(X, Y, X_test, Y_test, epochs=epochs, batch_size=128,
                 shuffle=True, lr_scheduler=get_lr_mnist)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        nn.save_model(path)
        with open(path, "rb") as fh:
            wd = pickle.load(fh)
    Ws, bs, i = [], [], 1
    while f"W{i}" in wd:
        Ws.append(np.asarray(wd[f"W{i}"], np.float32))
        bs.append(np.asarray(wd[f"b{i}"], np.float32).reshape(-1)); i += 1
    return Ws, bs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "gtsrb"])
    ap.add_argument("--arch", type=int, nargs="+", default=[128])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--poisoned-patch", type=int, default=4)
    ap.add_argument("--untrusted-tail", type=float, default=1.0 / 3.0)
    ap.add_argument("--drop-budgets", type=float, nargs="+", default=[0.05, 0.10, 0.15],
                    help="Fraction of TRAINING SAMPLES the sample-level arms drop")
    ap.add_argument("--k", type=float, default=8.0,
                    help="Robust deviations below the median at which a "
                         "feature is flagged; distribution-relative, so it "
                         "needs no per-dataset or per-run calibration")
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="Attributed-trust threshold for the pruning arm")
    ap.add_argument("--evidence", type=float, default=50.0)
    args = ap.parse_args()

    import eval_repair as ER
    from NN.datasets import load_data
    from main import DATASET_META
    from attribution import accumulate, ProvenanceSource, flag_features
    from subjective_logic import bpq_vec

    meta = DATASET_META[args.dataset]
    ER.PATCH_VALUE = meta["scale_patch"](1.0)
    img, pois_pair = meta["img_size"], tuple(meta["pois_pair"])
    patch_idx = np.array([img * r + c for r in range(args.poisoned_patch)
                          for c in range(args.poisoned_patch)])

    X, X_test, Y, Y_test, _ = load_data(args.dataset, "clean", "clean",
                                        poisoned_patch=args.poisoned_patch)
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.float32)
    X_test = np.asarray(X_test, np.float32); Y_test = np.asarray(Y_test, np.float32)
    y_test = Y_test.argmax(1)
    n = len(X)
    untrusted = np.zeros(n, bool)
    untrusted[int(round((1.0 - args.untrusted_tail) * n)):] = True
    lit = X[:, patch_idx].min(1) > (ER.PATCH_VALUE - 1e-3)
    print(f"[infl] {args.dataset}: {n} train, untrusted source {untrusted.mean()*100:.0f}%, "
          f"{int(lit.sum())} samples actually carry the trigger ({lit.mean()*100:.1f}%)")

    cache = f"results/Influence_{args.dataset}_{'_'.join(map(str,args.arch))}_p{args.poisoned_patch}"
    src = ProvenanceSource(n, args.untrusted_tail, (0.0, 1.0, 0.0))
    rows = []

    for seed in args.seeds:
        Ws, bs = train_model(X, Y, X_test, Y_test, args.arch, args.epochs, seed,
                             f"{cache}/models/full_seed{seed}.pkl")
        acc, _, asr = ER.evaluate(Ws, bs, X_test, y_test, patch_idx, pois_pair)
        rows.append(dict(seed=seed, arm="undefended", dropped=0, retrained=False,
                         clean_acc=acc, asr=asr, poison_recall=float("nan")))
        print(f"\n[infl] seed {seed}: undefended clean {acc*100:.2f}%  attack {asr*100:.2f}%")

        # --- this work: prune parameters, no retraining ---------------------
        mass, R, S = accumulate(Ws, bs, X, Y, src, verbose=False)
        m0 = mass[0][:-1]
        live = m0.sum(1) > np.percentile(m0.sum(1), 20)
        r = np.divide(args.evidence * R[0][:-1], m0, out=np.zeros_like(m0), where=m0 > 1e-12)
        s_ = np.divide(args.evidence * S[0][:-1], m0, out=np.zeros_like(m0), where=m0 > 1e-12)
        om = bpq_vec(r, s_, W=2.0)
        score = (om[..., 0] + 0.5 * om[..., 2]).min(1)
        sel, _ = flag_features(score, live, args.k)
        if len(sel):
            W_, b_ = ER.prune_features(Ws, bs, sel)
            a2, _, z2 = ER.evaluate(W_, b_, X_test, y_test, patch_idx, pois_pair)
        else:
            a2, z2 = acc, asr
        rows.append(dict(seed=seed, arm="prune (ours)", dropped=int(len(sel)),
                         retrained=False, clean_acc=a2, asr=z2,
                         poison_recall=float(np.isin(patch_idx, sel).mean())))
        print(f"[infl]   prune (ours): {len(sel)} features, no retrain -> "
              f"clean {a2*100:.2f}%  attack {z2*100:.2f}%")

        # --- drop the whole untrusted source and retrain --------------------
        keep = ~untrusted
        W2, b2 = train_model(X[keep], Y[keep], X_test, Y_test, args.arch,
                             args.epochs, seed, f"{cache}/models/dropsrc_seed{seed}.pkl")
        a3, _, z3 = ER.evaluate(W2, b2, X_test, y_test, patch_idx, pois_pair)
        rows.append(dict(seed=seed, arm="drop-source", dropped=int(untrusted.sum()),
                         retrained=True, clean_acc=a3, asr=z3,
                         poison_recall=float(lit[untrusted].sum() / max(lit.sum(), 1))))
        print(f"[infl]   drop-source: {int(untrusted.sum())} samples, retrained -> "
              f"clean {a3*100:.2f}%  attack {z3*100:.2f}%")

        # --- influence-ranked and random sample removal, both retrained -----
        infl = self_influence(Ws, bs, X, Y)
        order = np.argsort(-infl)
        rng = np.random.default_rng(seed)
        for frac in args.drop_budgets:
            k = int(round(frac * n))
            for arm, drop in (("influence", order[:k]),
                              ("random-samples", rng.choice(n, k, replace=False))):
                mask = np.ones(n, bool); mask[drop] = False
                tag = f"{arm.split('-')[0]}{int(frac*100)}_seed{seed}"
                W3, b3 = train_model(X[mask], Y[mask], X_test, Y_test, args.arch,
                                     args.epochs, seed, f"{cache}/models/{tag}.pkl")
                a4, _, z4 = ER.evaluate(W3, b3, X_test, y_test, patch_idx, pois_pair)
                rows.append(dict(seed=seed, arm=f"{arm} {int(frac*100)}%", dropped=k,
                                 retrained=True, clean_acc=a4, asr=z4,
                                 poison_recall=float(lit[drop].sum() / max(lit.sum(), 1))))
                print(f"[infl]   {arm} {int(frac*100)}%: retrained -> clean {a4*100:.2f}%"
                      f"  attack {z4*100:.2f}%  poison caught "
                      f"{lit[drop].sum()/max(lit.sum(),1)*100:.1f}%")

    from statistics import mean, pstdev
    arms = []
    for d in rows:
        if d["arm"] not in arms:
            arms.append(d["arm"])
    print(f"\n{'arm':<20}{'retrain':>8}{'dropped':>9}{'clean acc':>17}"
          f"{'attack success':>19}{'poison caught':>15}")
    for arm in arms:
        v = [d for d in rows if d["arm"] == arm]
        f = lambda k: (mean([d[k] for d in v]),
                       pstdev([d[k] for d in v]) if len(v) > 1 else 0.0)
        a, ad = f("clean_acc"); z, zd = f("asr")
        pr = [d["poison_recall"] for d in v if d["poison_recall"] == d["poison_recall"]]
        print(f"{arm:<20}{'yes' if v[0]['retrained'] else 'no':>8}{v[0]['dropped']:>9}"
              f"{a*100:>11.2f}±{ad*100:<5.2f}{z*100:>13.2f}±{zd*100:<5.2f}"
              f"{(mean(pr)*100 if pr else float('nan')):>14.1f}%")
    os.makedirs(cache, exist_ok=True)
    with open(os.path.join(cache, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(dataset=args.dataset, arch=args.arch, patch=args.poisoned_patch,
                       untrusted_tail=args.untrusted_tail, epochs=args.epochs,
                       seeds=args.seeds, threshold=args.threshold, rows=rows), fh, indent=2)
    print(f"\n[infl] saved {cache}/summary.json")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
