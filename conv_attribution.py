"""conv_attribution.py — provenance attribution for convolutional models.

For a dense layer a parameter belongs to one input feature, so attributing
trust to parameters and removing the untrusted ones is well defined. Weight
sharing breaks that: a convolutional filter is applied at every position and
is tied to none of them, so "these parameters carry the trigger" has no
spatial meaning and pruning a filter removes it everywhere at once.

We therefore attribute to input POSITIONS rather than to parameters, using
the gradient of the loss with respect to the input:

    mass_p = sum_n |x_{n,p} - mu_p| * |dL_n / dx_{n,p}|
    r_p    = N * sum_n |x_{n,p} - mu_p| * |dL_n / dx_{n,p}| * b_n / mass_p
    s_p    = N * sum_n ... * d_n / mass_p

This is the same quantity the dense method computes, since for a dense first
layer dL/dx_j is exactly the weighted sum of the errors the parameters of
feature j receive, but it is defined for any architecture. The output is a
spatial trust map, and positions many robust deviations below its median are
the ones untrusted data disproportionately drove.

The defence changes with it. Parameters cannot be removed per position, so
instead the flagged positions are neutralised at inference by replacing them
with the trusted-data mean. Nothing is retrained and the model is untouched.

Usage
-----
    python conv_attribution.py --dataset mnist --poisoned-patch 4 --seeds 0 1 2
"""
from __future__ import annotations

import os
import sys
import json
import argparse

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.join(_here, "patas_module"), os.path.join(_here, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """Two convolutional stages and a linear head. Deliberately ordinary: the
    point is weight sharing, not the architecture."""

    def __init__(self, img, n_classes, in_ch=1, width=16):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, width, 3, padding=1)
        self.c2 = nn.Conv2d(width, width * 2, 3, padding=1)
        self.fc = nn.Linear(width * 2 * (img // 4) * (img // 4), n_classes)
        self.img, self.in_ch = img, in_ch

    def forward(self, x):
        x = x.view(-1, self.in_ch, self.img, self.img)
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))


def train(model, X, Y, epochs, bs, seed, dev):
    torch.manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    Xt = torch.as_tensor(X, device=dev); Yt = torch.as_tensor(Y.argmax(1), device=dev)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xt[idx]), Yt[idx])
            loss.backward(); opt.step()
    return model


@torch.no_grad()
def accuracy(model, X, y, dev, bs=1024):
    ok = 0
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i:i + bs], device=dev)
        ok += (model(xb).argmax(1).cpu().numpy() == y[i:i + bs]).sum()
    return float(ok) / len(X)


def input_attribution(model, X, Y, b_n, d_n, mu, dev, bs=256):
    """Spatial trust map: influence of each input position, split into the
    belief and disbelief carried by the samples that produced it."""
    d = X.shape[1]
    mass = np.zeros(d); infl = np.zeros(d); R = np.zeros(d); S = np.zeros(d)
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i:i + bs], device=dev).clone().requires_grad_(True)
        yb = torch.as_tensor(Y[i:i + bs].argmax(1), device=dev)
        loss = F.cross_entropy(model(xb), yb, reduction="sum")
        g, = torch.autograd.grad(loss, xb)
        gr = np.abs(g.detach().cpu().numpy())
        w = np.abs(X[i:i + bs] - mu) * gr
        infl += (np.abs(X[i:i + bs]) * gr).sum(0)   # true influence, for the live filter
        mass += w.sum(0)
        R += (w * b_n[i:i + bs, None]).sum(0)
        S += (w * d_n[i:i + bs, None]).sum(0)
    return infl, mass, R, S


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "fashion", "gtsrb"])
    ap.add_argument("--poisoned-patch", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--untrusted-tail", type=float, default=1.0 / 3.0)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--rule", choices=["mad", "otsu"], default="mad",
                    help="Decision rule. 'mad' (default here) thresholds at k "
                         "robust deviations; 'otsu' is the bimodality rule the "
                         "dense experiments use.")
    ap.add_argument("--evidence", type=float, default=50.0)
    ap.add_argument("--width", type=int, default=16)
    args = ap.parse_args()

    from NN.datasets import load_data
    from main import DATASET_META
    from attribution import flag_features
    from subjective_logic import bpq_vec

    dev = "cpu"
    meta = DATASET_META[args.dataset]
    img, pois = meta["img_size"], tuple(meta["pois_pair"])
    pv = meta["scale_patch"](1.0)
    pidx = np.array([img * r + c for r in range(args.poisoned_patch)
                     for c in range(args.poisoned_patch)])
    X, X_test, Y, Y_test, _ = load_data(args.dataset, "clean", "clean",
                                        poisoned_patch=args.poisoned_patch)
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.float32)
    X_test = np.asarray(X_test, np.float32); y_test = np.asarray(Y_test).argmax(1)
    n = len(X)
    untrusted = np.zeros(n, bool)
    untrusted[int(round((1 - args.untrusted_tail) * n)):] = True
    b_n = np.where(untrusted, 0.0, 1.0).astype(np.float32)
    d_n = np.where(untrusted, 1.0, 0.0).astype(np.float32)
    mu = X[~untrusted].mean(0)
    print(f"[conv] {args.dataset}: {n} train, untrusted {untrusted.mean()*100:.0f}%, "
          f"trigger {args.poisoned_patch}x{args.poisoned_patch}")

    def asr(model, mask=None, patched=True):
        """Attack success. With patched=False this is the floor: how often a
        CLEAN input of a targeted class is already classified as the other,
        which masking cannot go below."""
        out = []
        for src, dst in ((pois[0], pois[1]), (pois[1], pois[0])):
            Xs = X_test[y_test == src]
            if not len(Xs):
                continue
            Xq = Xs.copy()
            if patched:
                Xq[:, pidx] = pv
            if mask is not None and len(mask):
                Xq[:, mask] = mu[mask]
            out.append(float(np.mean(
                np.concatenate([model(torch.as_tensor(Xq[i:i+1024], device=dev))
                                .argmax(1).cpu().numpy() for i in range(0, len(Xq), 1024)]) == dst)))
        return float(np.mean(out))

    rows = []
    for seed in args.seeds:
        np.random.seed(seed)
        model = SmallCNN(img, Y.shape[1], 1, args.width).to(dev)
        train(model, X, Y, args.epochs, 128, seed, dev)
        model.eval()
        a0 = accuracy(model, X_test, y_test, dev); z0 = asr(model)
        floor = asr(model, patched=False)
        infl, mass, R, S = input_attribution(model, X, Y, b_n, d_n, mu, dev)
        # the live filter must use true influence: a trigger position carries
        # little centred mass while being highly influential
        live = infl > np.percentile(infl, 20)
        r = np.divide(args.evidence * R, mass, out=np.zeros_like(mass), where=mass > 1e-12)
        s = np.divide(args.evidence * S, mass, out=np.zeros_like(mass), where=mass > 1e-12)
        om = bpq_vec(r, s, W=2.0)
        score = om[..., 0] + 0.5 * om[..., 2]
        # The bimodality rule that helps the dense case fires on only one of
        # three convolutional seeds, so the fixed-deviation rule is used here.
        # No single decision rule was best in both settings; see the paper.
        sel, _ = flag_features(score, live, args.k, rule=args.rule)
        share = np.divide(S, mass, out=np.zeros_like(mass), where=mass > 1e-12)
        order = np.argsort(np.where(live, score, np.inf))
        ranks = [int(np.where(order == p)[0][0]) for p in pidx if live[p]]
        zm = asr(model, sel) if len(sel) else z0
        # masking also has to leave clean inputs alone
        Xc = X_test.copy()
        if len(sel):
            Xc[:, sel] = mu[sel]
        am = accuracy(model, Xc, y_test, dev)
        print(f"[conv] seed {seed}: clean {a0*100:.2f}% attack {z0*100:.2f}%  |  "
              f"flagged {len(sel)} ({int(np.isin(pidx, sel).sum())}/{len(pidx)} trigger, "
              f"ranks {min(ranks) if ranks else -1}-{max(ranks) if ranks else -1})  ->  "
              f"masked clean {am*100:.2f}% attack {zm*100:.2f}%")
        rows.append(dict(seed=seed, clean=a0, asr=z0, floor=floor,
                         flagged=int(len(sel)),
                         trigger_found=int(np.isin(pidx, sel).sum()),
                         masked_clean=am, masked_asr=zm,
                         share_trigger=float(share[pidx].mean()),
                         share_live=float(share[live].mean())))
    from statistics import mean, pstdev
    f = lambda k: (mean([r[k] for r in rows]),
                   pstdev([r[k] for r in rows]) if len(rows) > 1 else 0.0)
    print(f"\n  {'':22}{'clean acc':>16}{'attack success':>18}")
    for lbl, ka, kz in (("undefended", "clean", "asr"),
                        ("masked (ours)", "masked_clean", "masked_asr"),
                        ("floor (clean inputs)", "clean", "floor")):
        a, ad = f(ka); z, zd = f(kz)
        print(f"  {lbl:<22}{a*100:>10.2f}±{ad*100:<5.2f}{z*100:>12.2f}±{zd*100:<5.2f}")
    a, _ = f("share_trigger"); b, _ = f("share_live")
    t, _ = f("trigger_found")
    print(f"  untrusted share: trigger {a:.3f} vs live {b:.3f}   "
          f"trigger positions found {t:.1f}/{len(pidx)}")
    out = f"results/ConvAttr_{args.dataset}_p{args.poisoned_patch}"
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(dataset=args.dataset, patch=args.poisoned_patch,
                       epochs=args.epochs, width=args.width, k=args.k,
                       rows=rows), fh, indent=2)
    print(f"\n[conv] saved {out}/summary.json")


if __name__ == "__main__":
    main()
