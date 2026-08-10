"""uq_methods.py — Uncertainty-quantification baselines and metrics.

Shared building blocks for the quantitative comparison of selective-prediction
filters (run_uq_comparison.py) and the ablation studies:

Baselines
---------
* **MC Dropout** (Gal & Ghahramani, 2016): the base architecture retrained
  with dropout; predictive distribution = mean softmax over T stochastic
  forward passes.
* **Evidential Deep Learning** (Sensoy, Kaplan & Kandemir, NeurIPS 2018):
  the base architecture retrained with the Dirichlet MSE loss (Eq. 5 of the
  paper) plus the annealed KL regulariser (Eq. 10).  Evidence head =
  softplus.  NOTE: the pypi package ``evidential-deep-learning`` is
  TensorFlow/Keras and targets evidential *regression* (Amini et al. 2020),
  so the classification loss is implemented here directly in torch,
  following Sensoy et al.
* **Softmax filter**: max softmax probability of the (already trained) base
  network — no extra training.
* **PaTAS filter**: per-sample IPTA trust score (built in
  run_uq_comparison.py from the cached omega_arrays.pkl).

Metrics
-------
* ``ece_from_confidence`` — Expected Calibration Error from per-sample
  (confidence, correctness) pairs; works for softmax confidences and for
  trust scores alike.
* ``coverage_accuracy_curve`` — selective-prediction sweep.
* ``roc_correctness`` — ROC of a confidence score as a detector of correct
  vs. incorrect predictions (+ AUC).

Plot helpers use one fixed, colourblind-safe (Okabe–Ito) colour per method
so that every figure in the chapter encodes methods identically.
"""
from __future__ import annotations

import os
import sys

# Path bootstrap so this module works from the repo root and from tests/
_v2_dir = os.path.dirname(os.path.abspath(__file__))
_patas_dir = os.path.join(_v2_dir, "patas_module")
for _p in (_v2_dir, _patas_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Plot style — serif, matches calibration_trust_eval.py / eval_5g_noise.py
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         13,
    "axes.titlesize":    13,
    "axes.labelsize":    13,
    "legend.fontsize":   11,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

#: Fixed method → (colour, linestyle, label).  Okabe–Ito palette (CVD-safe);
#: identity is additionally encoded by linestyle for print/grayscale.
METHOD_STYLE: Dict[str, dict] = {
    "patas":      {"color": "#0072B2", "ls": "-",  "label": "PaTAS filter"},
    "softmax":    {"color": "#555555", "ls": "--", "label": "Softmax filter"},
    "mc_dropout": {"color": "#D55E00", "ls": "-.", "label": "MC Dropout"},
    "edl":        {"color": "#009E73", "ls": ":",  "label": "EDL (Sensoy et al.)"},
    "deeptrust":  {"color": "#CC79A7", "ls": "-",  "label": "DeepTrust (Cheng et al.)"},
}


def _style(method: str) -> dict:
    return METHOD_STYLE.get(method, {"color": "black", "ls": "-", "label": method})


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TorchMLP(nn.Module):
    """ReLU MLP matching NN.primaryNN.NeuralNetwork's topology.

    ``dropout_p > 0`` inserts dropout after every hidden activation — kept
    active at inference time for MC Dropout via ``mc_forward``.
    """

    def __init__(self, input_size: int, hidden_sizes: list[int],
                 output_size: int, dropout_p: float = 0.0):
        super().__init__()
        self.dropout_p = dropout_p
        dims = [input_size] + list(hidden_sizes) + [output_size]
        self.layers = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                if self.dropout_p > 0:
                    x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x  # logits


class TorchResNetLite(nn.Module):
    """torch.nn mirror of NN.convNN.ConvNet (spec-driven ResNet-style CNN).

    Accepts the same ``specs`` produced by ``default_resnet_lite_specs`` /
    ``cifar10_resnet_specs`` so MC-Dropout/EDL variants share the exact
    topology of the PaTAS-assessed base model.  Dropout (2D, on feature
    maps) is applied after each pooled block when ``dropout_p > 0``.
    """

    def __init__(self, specs: list[dict], img_size: int, in_channels: int,
                 dropout_p: float = 0.0):
        super().__init__()
        self.specs = specs
        self.img_size = img_size
        self.in_channels = in_channels
        self.dropout_p = dropout_p
        mods = []
        for sp in specs:
            kind = sp["kind"]
            if kind == "conv":
                mods.append(nn.ModuleList([
                    nn.Conv2d(sp["cin"], sp["cout"], sp["k"], padding=sp["k"] // 2)]))
            elif kind == "resconv":
                mods.append(nn.ModuleList([
                    nn.Conv2d(sp["cin"], sp["cout"], sp["k"], padding=sp["k"] // 2),
                    nn.Conv2d(sp["cout"], sp["cout"], sp["k"], padding=sp["k"] // 2)]))
            elif kind == "dense":
                mods.append(nn.ModuleList([
                    nn.Linear(sp["cin"] * sp["spatial"], sp["out"])]))
            else:
                raise ValueError(f"Unknown spec kind: {kind}")
        self.blocks = nn.ModuleList(mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.view(-1, self.in_channels, self.img_size, self.img_size)
        for sp, blk in zip(self.specs, self.blocks):
            kind = sp["kind"]
            if kind == "conv":
                h = F.relu(blk[0](h))
                if sp.get("pool", True):
                    h = F.avg_pool2d(h, 2)
                if self.dropout_p > 0:
                    h = F.dropout2d(h, p=self.dropout_p, training=self.training)
            elif kind == "resconv":
                r = F.relu(blk[0](h))
                r = blk[1](r)
                h = F.relu(h + r)
                if sp.get("pool", True):
                    h = F.avg_pool2d(h, 2)
                if self.dropout_p > 0:
                    h = F.dropout2d(h, p=self.dropout_p, training=self.training)
            else:  # dense
                h = blk[0](h.reshape(h.shape[0], -1))
        return h  # logits


# ---------------------------------------------------------------------------
# EDL loss (Sensoy et al. 2018)
# ---------------------------------------------------------------------------

def _kl_dirichlet_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL( Dir(alpha) || Dir(1,...,1) ), per sample.  alpha: (n, K)."""
    K = alpha.shape[1]
    S = alpha.sum(dim=1)
    ln_gamma_S = torch.lgamma(S)
    ln_gamma_a = torch.lgamma(alpha).sum(dim=1)
    ln_gamma_K = torch.lgamma(torch.tensor(float(K), device=alpha.device))
    dg = torch.digamma(alpha) - torch.digamma(S).unsqueeze(1)
    return ln_gamma_S - ln_gamma_a - ln_gamma_K + ((alpha - 1.0) * dg).sum(dim=1)


def edl_mse_loss(logits: torch.Tensor, y_onehot: torch.Tensor,
                 epoch: int, annealing_epochs: int = 10) -> torch.Tensor:
    """Dirichlet MSE loss (Sensoy et al., Eq. 5) + annealed KL (Eq. 10).

    evidence = softplus(logits);  alpha = evidence + 1.
    """
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    err = ((y_onehot - p) ** 2).sum(dim=1)
    var = (p * (1.0 - p) / (S + 1.0)).sum(dim=1)
    # KL only on the misleading evidence: alpha_tilde = y + (1-y)*alpha
    alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
    lam = min(1.0, float(epoch) / float(max(annealing_epochs, 1)))
    return (err + var).mean() + lam * _kl_dirichlet_uniform(alpha_tilde).mean()


# ---------------------------------------------------------------------------
# Training (shared by MC-Dropout and EDL variants)
# ---------------------------------------------------------------------------

def train_torch_model(
    model: nn.Module,
    X_train: np.ndarray, y_train_onehot: np.ndarray,
    X_test: Optional[np.ndarray] = None, y_test_onehot: Optional[np.ndarray] = None,
    epochs: int = 20, batch_size: int = 128,
    lr_scheduler: Callable[[int], float] = lambda e: 0.05,
    loss_type: str = "ce",           # "ce" | "edl"
    annealing_epochs: int = 10,
    device: Optional[str] = None,
    seed: int = 42,
    verbose: bool = True,
) -> nn.Module:
    """Mini-batch SGD trainer mirroring NeuralNetwork.train's regime
    (plain SGD + per-epoch LR schedule + shuffling), so the MC-Dropout and
    EDL variants differ from the base model only where the method requires.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    Xt = torch.as_tensor(X_train, dtype=torch.float32)
    Yt = torch.as_tensor(y_train_onehot, dtype=torch.float32)
    n = Xt.shape[0]

    for epoch in range(epochs):
        model.train()
        lr = float(lr_scheduler(epoch))
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        perm = torch.randperm(n)
        tot_loss, nb = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xt[idx].to(dev)
            yb = Yt[idx].to(dev)
            logits = model(xb)
            if loss_type == "edl":
                loss = edl_mse_loss(logits, yb, epoch, annealing_epochs)
            else:
                loss = F.cross_entropy(logits, yb.argmax(dim=1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += float(loss.item())
            nb += 1
        if verbose:
            msg = f"  [{loss_type}] epoch {epoch+1}/{epochs}  loss={tot_loss/max(nb,1):.4f}"
            if X_test is not None and y_test_onehot is not None:
                acc = float(np.mean(
                    predict_probs(model, X_test).argmax(1)
                    == y_test_onehot.argmax(1)))
                msg += f"  test_acc={acc*100:.2f}%"
            print(msg)
    return model


@torch.no_grad()
def predict_probs(model: nn.Module, X: np.ndarray, chunk: int = 4096,
                  edl: bool = False) -> np.ndarray:
    """Deterministic predictive probabilities.

    edl=False: softmax(logits).  edl=True: alpha / S (Dirichlet mean).
    """
    model.eval()
    dev = next(model.parameters()).device
    outs = []
    for i in range(0, len(X), chunk):
        logits = model(torch.as_tensor(X[i:i + chunk], dtype=torch.float32, device=dev))
        if edl:
            alpha = F.softplus(logits) + 1.0
            outs.append((alpha / alpha.sum(dim=1, keepdim=True)).cpu().numpy())
        else:
            outs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, X: np.ndarray, T: int = 30,
                       chunk: int = 4096, seed: int = 0) -> dict:
    """T stochastic forward passes with dropout active.

    Returns dict with:
        probs   : (n, K) mean softmax
        score   : (n,)  max of mean softmax (confidence for filtering)
        entropy : (n,)  predictive entropy of the mean
        mi      : (n,)  mutual information (BALD)
    """
    model.train()  # dropout active; no grads (no_grad context)
    torch.manual_seed(seed)
    dev = next(model.parameters()).device
    sum_p = None
    sum_ent = None
    for _ in range(T):
        outs = []
        for i in range(0, len(X), chunk):
            logits = model(torch.as_tensor(X[i:i + chunk], dtype=torch.float32, device=dev))
            outs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        p = np.concatenate(outs, axis=0)
        ent = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1)
        sum_p = p if sum_p is None else sum_p + p
        sum_ent = ent if sum_ent is None else sum_ent + ent
    model.eval()
    probs = sum_p / T
    pred_ent = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)
    mi = pred_ent - sum_ent / T
    return {"probs": probs, "score": probs.max(axis=1),
            "entropy": pred_ent, "mi": mi}


@torch.no_grad()
def edl_predict(model: nn.Module, X: np.ndarray, chunk: int = 4096) -> dict:
    """EDL predictive quantities (Sensoy et al.).

    Returns dict with:
        probs : (n, K) alpha / S
        u     : (n,)  vacuity K / S
        score : (n,)  1 - u  (total evidence mass — confidence for filtering)
        bmax  : (n,)  belief mass of the predicted class
    """
    model.eval()
    dev = next(model.parameters()).device
    alphas = []
    for i in range(0, len(X), chunk):
        logits = model(torch.as_tensor(X[i:i + chunk], dtype=torch.float32, device=dev))
        alphas.append((F.softplus(logits) + 1.0).cpu().numpy())
    alpha = np.concatenate(alphas, axis=0)
    S = alpha.sum(axis=1, keepdims=True)
    K = alpha.shape[1]
    probs = alpha / S
    u = (K / S).ravel()
    belief = (alpha - 1.0) / S
    return {"probs": probs, "u": u, "score": 1.0 - u,
            "bmax": belief.max(axis=1)}


# ---------------------------------------------------------------------------
# Model cache helpers
# ---------------------------------------------------------------------------

def load_or_train(model_factory: Callable[[], nn.Module], cache_path: str,
                  train_fn: Callable[[nn.Module], nn.Module],
                  force_retrain: bool = False) -> nn.Module:
    """state_dict-level cache: load cache_path if present, else train + save."""
    model = model_factory()
    if not force_retrain and os.path.exists(cache_path):
        print(f"[UQ] Loading cached model from {cache_path}")
        model.load_state_dict(torch.load(cache_path, map_location="cpu"))
        return model
    model = train_fn(model)
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    torch.save(model.state_dict(), cache_path)
    print(f"[UQ] Model cached to {cache_path}")
    return model


# ---------------------------------------------------------------------------
# Test-time corruption (matches NN.datasets.noised_features / load_X "noise")
# ---------------------------------------------------------------------------

def apply_feature_noise(X: np.ndarray, noise_prob: float,
                        noise_scale: float = 0.3, seed: int = 0) -> np.ndarray:
    """Vectorised, seeded version of NN.datasets.noised_features.

    Each feature is corrupted independently with probability ``noise_prob``
    by adding uniform noise in [-noise_scale, +noise_scale]; values are
    clipped to [0, 1].  The input is not modified.
    """
    if noise_prob <= 0:
        return X
    rng = np.random.default_rng(seed)
    Xn = X.astype(np.float32, copy=True)
    mask = rng.random(Xn.shape) < noise_prob
    Xn[mask] += ((rng.random(int(mask.sum())) * 2 - 1) * noise_scale).astype(np.float32)
    return np.clip(Xn, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ece_from_confidence(confidence: np.ndarray, correct: np.ndarray,
                        n_bins: int = 15) -> float:
    """Expected Calibration Error over equal-width confidence bins.

    ECE = sum_b (n_b / n) * | acc(b) - conf(b) |

    Works for softmax max-probabilities and for trust-based confidence
    scores alike (any score in [0, 1] interpreted as claimed accuracy).
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    ok = np.isfinite(confidence)
    confidence, correct = confidence[ok], correct[ok]
    n = len(confidence)
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (confidence > lo) & (confidence <= hi) if i > 0 else \
            (confidence >= lo) & (confidence <= hi)
        nb = int(m.sum())
        if nb == 0:
            continue
        ece += (nb / n) * abs(correct[m].mean() - confidence[m].mean())
    return float(ece)


def coverage_accuracy_curve(scores: np.ndarray, correct: np.ndarray,
                            n_taus: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """Sweep threshold τ over score quantiles; return (coverage %, accuracy %)."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    finite = scores[np.isfinite(scores)]
    n = len(scores)
    if len(finite) == 0 or n == 0:
        return np.array([100.0]), np.array([np.nan])
    taus = np.unique(np.quantile(finite, np.linspace(0, 1, n_taus)))
    covs, accs = [], []
    for tau in taus:
        m = scores >= tau
        nc = int(m.sum())
        if nc == 0:
            continue
        covs.append(100.0 * nc / n)
        accs.append(100.0 * float(correct[m].mean()))
    return np.asarray(covs), np.asarray(accs)


def roc_correctness(scores: np.ndarray, correct: np.ndarray):
    """ROC of `scores` as a detector of correct predictions.

    Returns (fpr, tpr, auc).  Requires scikit-learn.
    """
    from sklearn.metrics import roc_curve, auc
    ok = np.isfinite(scores)
    fpr, tpr, _ = roc_curve(np.asarray(correct, dtype=int)[ok], scores[ok])
    return fpr, tpr, float(auc(fpr, tpr))


def aurc(scores: np.ndarray, correct: np.ndarray) -> float:
    """Area under the Risk–Coverage curve (lower is better).

    Risk = error rate on the covered set as coverage sweeps 1/n .. 1.
    """
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=float)
    ok = np.isfinite(scores)
    scores, correct = scores[ok], correct[ok]
    n = len(scores)
    if n == 0:
        return float("nan")
    order = np.argsort(-scores)           # most confident first
    err = 1.0 - correct[order]
    risks = np.cumsum(err) / np.arange(1, n + 1)
    return float(risks.mean())


# ---------------------------------------------------------------------------
# Figures — one colour per method, consistent across the chapter
# ---------------------------------------------------------------------------

def plot_roc_methods(results: Dict[str, dict], out_base: str, title: str) -> None:
    """results[method] must contain 'score' and 'correct'."""
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    for method, res in results.items():
        st = _style(method)
        fpr, tpr, auc_val = roc_correctness(res["score"], res["correct"])
        ax.plot(fpr, tpr, color=st["color"], ls=st["ls"], lw=2,
                label=f"{st['label']}  (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1, ls="-", zorder=0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    ax.grid(linestyle=":", alpha=0.35)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_base}.pdf")


def plot_metrics_vs_noise(metrics_by_cond: Dict[float, Dict[str, dict]],
                          out_base: str, title: str) -> None:
    """Accuracy / AUROC / ECE as a function of the test-noise level.

    metrics_by_cond: {noise_level: {method: {'test_acc','auroc','ece',...}}}
    """
    noise_levels = sorted(metrics_by_cond)
    methods = list(next(iter(metrics_by_cond.values())).keys())
    panels = (("test_acc", "Accuracy (%)", 100.0),
              ("auroc", "AUROC $\\uparrow$", 1.0),
              ("ece", "ECE $\\downarrow$", 1.0))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (key, ylabel, scale) in zip(axes, panels):
        for method in methods:
            st = _style(method)
            vals = [metrics_by_cond[nl].get(method, {}).get(key, np.nan) * scale
                    for nl in noise_levels]
            ax.plot(noise_levels, vals, marker="o", ms=5, lw=2,
                    color=st["color"], ls=st["ls"], label=st["label"])
        ax.set_xlabel("Test-time feature-noise probability $p$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(noise_levels)
        ax.grid(linestyle=":", alpha=0.35)
    axes[0].legend(fontsize=9)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_base}.pdf")


def plot_coverage_methods(results: Dict[str, dict], out_base: str, title: str) -> None:
    """results[method] must contain 'score' and 'correct'."""
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    for method, res in results.items():
        st = _style(method)
        covs, accs = coverage_accuracy_curve(res["score"], res["correct"])
        ax.plot(covs, accs, color=st["color"], ls=st["ls"], lw=2, label=st["label"])
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Accuracy on covered samples (%)")
    ax.set_title(title)
    ax.set_xlim(-2, 103)
    ax.legend(loc="lower left")
    ax.grid(linestyle=":", alpha=0.35)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_base}.pdf")
