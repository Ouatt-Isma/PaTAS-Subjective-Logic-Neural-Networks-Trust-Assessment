"""run_uq_comparison.py — Quantitative comparison of selective-prediction filters.

Compares, on the clean-data (trust/trust) model of each dataset:

    * PaTAS filter    — trust-discounted confidence: the softmax confidence
                        discounted (SL trust discounting) by the per-sample
                        IPTA path-trust from the cached PaTAS opinion
                        matrices (omega_arrays.pkl).  The raw path trust is
                        model-side (near-constant on a well-trained network)
                        and is kept as a diagnostic in scores.npz.
    * Softmax filter  — max softmax probability of the base network
    * MC Dropout      — Gal & Ghahramani (2016), T stochastic passes
    * EDL             — Sensoy et al. (NeurIPS 2018), Dirichlet MSE loss
    * DeepTrust       — Cheng et al.; no public code, so only the values
                        reported in their paper appear in the summary table
                        (fill DEEPTRUST_PAPER below).

Every method is evaluated on the clean test set AND under test-time feature
noise (--test-noise, default p ∈ {0.3, 0.6}; the same Bernoulli-uniform
corruption as the training-noise model, labels stay clean).  This whole
battery is additionally repeated for models TRAINED under label noise
(--label-noise, default flip rate 0.3: clean features, labels flipped with
probability p via NN.datasets.noised_label / y_trust='vacuous') alongside
the default clean-trained models — the case where PaTAS's model-side trust
should diverge from softmax, which stays confident regardless of how the
network was trained.

Outputs per dataset AND per training condition
(results/UQ_Compare_<dataset>_<arch>[_labelnoise<p>]/):
    roc_<cond>.pdf/.png                ROC of each score as a correct-vs-
                                       incorrect detector (cond = clean,
                                       noise0.3, ...), with AUC values
    coverage_accuracy_<cond>.pdf/.png  accuracy–coverage curves per condition
    metrics_vs_noise.pdf/.png          Accuracy / AUROC / ECE vs noise level
    ece_table.tex                      LaTeX table: Acc / AUROC / AURC / ECE /
                                       ECE-calibrated, one block per test
                                       condition. ECE-calibrated re-scores
                                       each method's own confidence through
                                       an isotonic map fit once per method on
                                       a held-out clean split (same treatment
                                       for every method, so no method is
                                       singled out for recalibration)
    summary.json                       all metrics, machine-readable
    scores.npz                         raw per-sample scores per condition

Usage
-----
    python run_uq_comparison.py --dataset mnist                   # cached models
    python run_uq_comparison.py --dataset mnist --train-missing   # train what's absent
    python run_uq_comparison.py --dataset gtsrb --epochs 20
    python run_uq_comparison.py --dataset cifar10 --train-missing
    python run_uq_comparison.py --dataset mnist --quick           # smoke test
    python run_uq_comparison.py --dataset mnist --fuse-method cumulative --train-missing
    python run_uq_comparison.py --dataset mnist --force-retrain-all     # wipe & retrain everything

Cache reuse (x_trust/y_trust = trust/trust for clean, trust/vacuous for
train-time label noise):
    base NN     results/NN_Train_<ds>_<arch>_<x_trust>_<y_trust>_PathSize_None/nn_model.pkl
    PaTAS       results/PTAS_Eval_<ds>_<arch>_<x_trust>_<y_trust>_eps_<eps>_PathSize_None[_fuse_<method>]/omega_arrays.pkl
    MC/EDL      results/UQ_Train_<ds>_<arch>[_labelnoise<p>]_{mcdropout,edl}/model.pt

--fuse-method (average | cumulative | weighted | compromise | constraint,
default average) selects the subjective-logic fusion operator GenIPTA uses
to combine the per-sample activated-path weight opinions into one trust
value, and that PTAS.run_chunk uses to revise omegas during training.
Non-default values get their own PTAS_Eval_..._fuse_<method> cache dir
(never collides with --fuse-method average runs), so pair a change with
--train-missing.

The MNIST/GTSRB PaTAS filter uses per-sample IPTA (exact path-conditioned
trust).  For CIFAR-10 the PaTAS mirror is convolutional, where IPTA is not
defined; there the PaTAS score is the softmax confidence discounted by the
projected probability of the predicted class's output-trust opinion (from
at.pkl), i.e. a class-level trust discount.
"""
from __future__ import annotations

import os
import sys
import io
import json
import time
import pickle
import shutil
import argparse
import contextlib
import multiprocessing
from dataclasses import dataclass
from typing import Optional

# ── Path bootstrap ────────────────────────────────────────────────────────────
_v2_dir = os.path.dirname(os.path.abspath(__file__))
_patas_dir = os.path.join(_v2_dir, "patas_module")
_tests_dir = os.path.join(_v2_dir, "tests")
for _p in (_v2_dir, _patas_dir, _tests_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import patas_module  # noqa: F401 — path bootstrap

from uq_methods import (
    TorchMLP, TorchResNetLite,
    train_torch_model, predict_probs, mc_dropout_predict, edl_predict,
    load_or_train,
    ece_from_confidence, aurc, roc_correctness,
    fit_calibrator, apply_calibrator,
    plot_roc_methods, plot_coverage_methods,
)

# ---------------------------------------------------------------------------
# DeepTrust (Cheng et al.) — no public implementation available.
# Fill in the values reported in the paper for the matching setup; rows stay
# "—" in the LaTeX table until provided.  Keys: dataset → {metric: value}.
# ---------------------------------------------------------------------------
DEEPTRUST_PAPER: dict = {
    # "mnist": {"test_acc": None, "auroc": None, "ece": None,
    #           "note": "values transcribed from Cheng et al., Table X"},
}

_DEFAULT_EPS = 0.05

DATASET_CFG = {
    "mnist":   {"arch": (128,),  "input_dim": 28 * 28, "output_dim": 10, "port": 5241},
    "gtsrb":   {"arch": (128,),  "input_dim": 32 * 32, "output_dim": 43, "port": 5251},
    "cifar10": {"arch": "resnet-lite", "input_dim": 3 * 32 * 32, "output_dim": 10,
                "port": 5261, "img_size": 32, "in_channels": 3, "base_channels": 16},
}


def _lr_for(dataset: str):
    from main import get_lr_mnist, get_lr_gtsrb
    return {"mnist": get_lr_mnist, "gtsrb": get_lr_gtsrb,
            "cifar10": get_lr_mnist}[dataset]


def _arch_str(arch) -> str:
    if isinstance(arch, str):
        return arch
    return "_".join(str(h) for h in arch)


def _nn_dir(dataset: str, arch, x_trust: str = "trust", y_trust: str = "trust") -> str:
    return f"results/NN_Train_{dataset}_{_arch_str(arch)}_{x_trust}_{y_trust}_PathSize_None"


def _ptas_dir(dataset: str, arch, eps: float, fuse_method: str = "average",
             x_trust: str = "trust", y_trust: str = "trust") -> str:
    suffix = "" if fuse_method == "average" else f"_fuse_{fuse_method}"
    return (f"results/PTAS_Eval_{dataset}_{_arch_str(arch)}_{x_trust}_{y_trust}"
            f"_eps_{eps}_PathSize_None{suffix}")


# ---------------------------------------------------------------------------
# Training conditions — clean data, and train-time label noise
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainCondition:
    """A training-data condition: which x_trust/y_trust dataset variant the
    base NN, PaTAS, MC-Dropout and EDL models are all trained on.

    ``tag`` names the results sub-directory / cache suffix (empty for the
    default "clean" condition, so it reuses pre-existing caches unchanged).
    """
    tag: str
    x_trust: str
    y_trust: str
    noise_level: Optional[float]
    label: str


CLEAN_CONDITION = TrainCondition("", "trust", "trust", None, "Clean training data")


def label_noise_condition(rate: float) -> TrainCondition:
    """Train-time label noise: clean features, labels flipped with
    probability ``rate`` (NN.datasets.noised_label via y_trust='vacuous')."""
    return TrainCondition(f"labelnoise{rate:g}", "trust", "vacuous", rate,
                          f"Label noise (train-time flip $p={rate:g}$)")


# ---------------------------------------------------------------------------
# Base model (softmax filter) — reuses the scenario caches
# ---------------------------------------------------------------------------

def get_base_mlp(dataset: str, arch: tuple, X_train, y_train, X_test, y_test,
                 epochs: int, train_missing: bool,
                 x_trust: str = "trust", y_trust: str = "trust",
                 force_retrain: bool = False):
    from NN.primaryNN import NeuralNetwork
    from NN.utils import writedict

    cfg = DATASET_CFG[dataset]
    nn_dir = _nn_dir(dataset, arch, x_trust, y_trust)
    model_path = os.path.join(nn_dir, "nn_model.pkl")
    nn = NeuralNetwork(cfg["input_dim"], hidden_sizes=list(arch),
                       output_size=cfg["output_dim"], ptas=False, operation=True)
    cached = os.path.exists(model_path)
    if cached and not force_retrain:
        nn.load_model(model_path)
        return nn
    if not cached and not train_missing:
        raise FileNotFoundError(
            f"Base model not found: {model_path}\n"
            f"Run the {x_trust}/{y_trust} scenario first "
            f"or pass --train-missing.")
    print(f"[BASE] Training base NN for {dataset} {arch} "
          f"({x_trust}/{y_trust}, no PaTAS attached)"
          f"{' — forced retrain' if cached else ''} ...")
    os.makedirs(nn_dir, exist_ok=True)
    nn.train(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=128,
             lr_scheduler=_lr_for(dataset), shuffle=True, plot=False, fname=nn_dir)
    nn.save_model(model_path)
    train_acc = float(np.mean(nn.predict(X_train) == y_train.argmax(1)))
    test_acc = float(np.mean(nn.predict(X_test) == y_test.argmax(1)))
    writedict({"Train": train_acc, "Test": test_acc},
              os.path.join(nn_dir, "metrics.txt"))
    return nn


def get_base_convnet(X_train, y_train, X_test, y_test, epochs: int,
                     train_missing: bool,
                     x_trust: str = "trust", y_trust: str = "trust",
                     force_retrain: bool = False):
    from NN.convNN import ConvNet, cifar10_resnet_specs
    from NN.utils import writedict

    cfg = DATASET_CFG["cifar10"]
    specs = cifar10_resnet_specs(img_size=cfg["img_size"],
                                 in_channels=cfg["in_channels"],
                                 num_classes=cfg["output_dim"],
                                 base_channels=cfg["base_channels"])
    nn_dir = _nn_dir("cifar10", "resnet-lite", x_trust, y_trust)
    model_path = os.path.join(nn_dir, "nn_model.pkl")
    net = ConvNet(img_size=cfg["img_size"], in_channels=cfg["in_channels"],
                  num_classes=cfg["output_dim"], base_channels=cfg["base_channels"],
                  ptas=False, operation=True, specs=specs)
    cached = os.path.exists(model_path)
    if cached and not force_retrain:
        net.load_model(model_path)
        return net, specs
    if not cached and not train_missing:
        raise FileNotFoundError(
            f"Base ConvNet not found: {model_path}\n"
            f"Run tests/test_cifar10_resnet.py first or pass --train-missing.")
    print(f"[BASE] Training CIFAR-10 ResNet-lite ({x_trust}/{y_trust}, "
          f"no PaTAS attached){' — forced retrain' if cached else ''} ...")
    hist = net.train(X_train, y_train, X_test, y_test, epochs=epochs,
                     batch_size=128, lr_scheduler=_lr_for("cifar10"))
    os.makedirs(nn_dir, exist_ok=True)
    net.save_model(model_path)
    writedict({"Train": hist["train_acc"][-1] if hist["train_acc"] else float("nan"),
               "Test": hist["test_acc"][-1] if hist["test_acc"] else float("nan")},
              os.path.join(nn_dir, "metrics.txt"))
    return net, specs


# ---------------------------------------------------------------------------
# PaTAS per-sample filter (IPTA) — MLP datasets
# ---------------------------------------------------------------------------

def load_offline_ptas(dataset: str, arch: tuple, eps: float,
                      fuse_method: str = "average",
                      x_trust: str = "trust", y_trust: str = "trust"):
    """Rebuild a PTAS object from cached omega_arrays.pkl (no socket)."""
    from NN.PTAStemplate import PTAS
    from concrete.TensorTO import TensorArrayTO

    cfg = DATASET_CFG[dataset]
    omega_path = os.path.join(
        _ptas_dir(dataset, arch, eps, fuse_method, x_trust, y_trust),
        "omega_arrays.pkl")
    if not os.path.exists(omega_path):
        return None
    with open(omega_path, "rb") as fh:
        omega_arrays = pickle.load(fh)
    structure = [cfg["input_dim"]] + list(arch) + [cfg["output_dim"]]
    return PTAS(
        omega_thetas=[TensorArrayTO(a) for a in omega_arrays],
        operator_mapping=None,
        nn_interface=None,
        trust_assessment_func=None,
        structure=structure,
        epsilon_low=eps,
        eval=False,
        fuse_method=fuse_method,
    )


def patas_ipta_scores(ptas, base_nn, X: np.ndarray, input_dim: int) -> dict:
    """Per-sample PaTAS filter: trust-discounted confidence.

    For each sample the IPTA (activation-path-pruned trust network, evaluated
    under a fully-trusted input) yields the trust opinion of the predicted
    class's computation path.  Its projected probability t = b + u/2 is
    *model-side* trust — nearly constant across samples on a well-trained
    network — so it is combined with the input-conditional prediction
    confidence via SL trust discounting:

        score = t_path × p_softmax(predicted class)

    Returns dict with:
        score : (n,) trust-discounted confidence (the PaTAS filter score)
        trust : (n,) raw IPTA path trust t = b + u/2 (diagnostic)
        conf  : (n,) softmax confidence of the base network
    """
    from concrete.TensorTO import TensorArrayTO, fill as tfill
    from tqdm import tqdm

    Tx = TensorArrayTO(tfill((1, input_dim), method="trust"))
    trust = np.full(len(X), np.nan)
    conf = np.full(len(X), np.nan)
    silent = io.StringIO()
    for i in tqdm(range(len(X)), desc="PaTAS IPTA", ncols=90):
        try:
            with contextlib.redirect_stdout(silent):
                probs, path = base_nn.forward(X[i:i + 1], getactivated=True)
                ipta = ptas.GenIPTA(path)
                Ty = ipta(Tx)                       # (1, K, 3)
            pred = int(np.argmax(probs, axis=1)[0])
            op = np.asarray(Ty.to_numpy())[0, pred]  # [b, d, u]
            trust[i] = float(op[0] + 0.5 * op[2])    # projected probability
            conf[i] = float(np.max(probs))
        except Exception:
            pass
    return {"score": trust * conf, "trust": trust, "conf": conf}


def patas_class_trust_scores(dataset: str, arch, eps: float,
                             probs: np.ndarray, fuse_method: str = "average",
                             x_trust: str = "trust", y_trust: str = "trust"
                             ) -> Optional[dict]:
    """Conv fallback (CIFAR-10): softmax confidence discounted by the projected
    probability of the predicted class's output-trust opinion (at.pkl).
    Same semantics as patas_ipta_scores (score / trust / conf) with the trust
    factor at class rather than activation-path granularity — IPTA is not
    defined for conv layers."""
    at_path = os.path.join(
        _ptas_dir(dataset, arch, eps, fuse_method, x_trust, y_trust), "at.pkl")
    if not os.path.exists(at_path):
        return None
    with open(at_path, "rb") as fh:
        at = pickle.load(fh)
    v = np.asarray(at.value if hasattr(at, "value") else at)  # (1, K, 3)
    v = v.reshape(-1, 3)
    class_pp = v[:, 0] + 0.5 * v[:, 2]                        # (K,)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    trust = class_pp[pred]
    return {"score": conf * trust, "trust": trust, "conf": conf}


def ensure_patas_cache(dataset: str, arch: tuple, eps: float, epochs: int,
                       train_missing: bool, fuse_method: str = "average",
                       x_trust: str = "trust", y_trust: str = "trust",
                       noise_level: Optional[float] = None,
                       force_retrain: bool = False) -> bool:
    """Make sure omega_arrays.pkl exists; optionally run the scenario."""
    ptas_dir = _ptas_dir(dataset, arch, eps, fuse_method, x_trust, y_trust)
    omega_path = os.path.join(ptas_dir, "omega_arrays.pkl")
    cached = os.path.exists(omega_path)
    if cached and not force_retrain:
        return True
    if not cached and not train_missing:
        print(f"[PaTAS] Missing {omega_path}\n"
              f"        Run the {x_trust}/{y_trust} scenario "
              f"or pass --train-missing.  Skipping PaTAS filter.")
        return False
    if cached and force_retrain:
        # start_ptas() below only retrains when omega_arrays.pkl is absent, so
        # the stale cache (and its at/av/ad.pkl, plot, log) must go first.
        shutil.rmtree(ptas_dir)
    print(f"[PaTAS] Training scenario for {dataset} {arch} eps={eps} "
          f"fuse={fuse_method} {x_trust}/{y_trust}"
          f"{' (forced retrain)' if cached else ''} ...")
    from main import TestCaseConfig
    from test_mnist import run_scenario  # generic over cfg
    cfgd = DATASET_CFG[dataset]
    cfg = TestCaseConfig(
        dataset=dataset, input_dim=cfgd["input_dim"], output_dim=cfgd["output_dim"],
        hidden_dim=arch[0], hidden_dims=tuple(arch), epochs=epochs, batch_size=128,
        learning_rate=_lr_for(dataset), epsilon_low=eps,
        x_trust=x_trust, y_trust=y_trust, port=cfgd["port"],
        mnist_patch_size=None, mnist_poisoned_soph=False,
        fuse_method=fuse_method, noise_level=noise_level,
    )
    run_scenario(cfg)
    ok = os.path.exists(omega_path)
    if not ok:
        print(f"[PaTAS] Training attempt did NOT produce {omega_path} — "
              f"the PTAS server/client scenario likely errored or was "
              f"interrupted (check for '[PTAS ERROR]' / '[CLIENT ERROR]' "
              f"above, or a port {cfgd['port']} conflict). "
              f"Skipping PaTAS filter for this condition.")
    return ok


def ensure_cifar_cache(x_trust: str, y_trust: str, eps: float, epochs: int,
                       base_channels: int, train_missing: bool,
                       noise_level: Optional[float] = None,
                       force_retrain: bool = False) -> bool:
    """Make sure the CIFAR-10 ResNet-lite PTAS+NN caches exist for the given
    (x_trust, y_trust) training condition; optionally run the scenario.

    Runs the *same* two-process scenario as tests/test_cifar10_resnet.py, so
    it populates both the PTAS_Eval (omega_arrays.pkl, at.pkl) and NN_Train
    (nn_model.pkl, metrics.txt) directories that get_base_convnet and
    patas_class_trust_scores read from — no duplicate training.
    """
    ptas_dir = _ptas_dir("cifar10", "resnet-lite", eps, "average", x_trust, y_trust)
    nn_dir = _nn_dir("cifar10", "resnet-lite", x_trust, y_trust)
    omega_path = os.path.join(ptas_dir, "omega_arrays.pkl")
    nn_model_path = os.path.join(nn_dir, "nn_model.pkl")
    cached = os.path.exists(omega_path) and os.path.exists(nn_model_path)
    if cached and not force_retrain:
        return True
    if not cached and not train_missing:
        print(f"[PaTAS/CIFAR] Missing {omega_path} or {nn_model_path}\n"
              f"        Run tests/test_cifar10_resnet.py --xtrust {x_trust} "
              f"--ytrust {y_trust} or pass --train-missing.")
        return False
    if force_retrain:
        # run_cifar_resnet_scenario() below only retrains what's absent, so
        # both stale caches must go first for a true forced retrain.
        if os.path.exists(ptas_dir):
            shutil.rmtree(ptas_dir)
        if os.path.exists(nn_dir):
            shutil.rmtree(nn_dir)
    print(f"[PaTAS/CIFAR] Training scenario for cifar10 resnet-lite eps={eps} "
          f"{x_trust}/{y_trust}{' (forced retrain)' if cached else ''} ...")
    from test_cifar10_resnet import run_cifar_resnet_scenario
    cfgd = DATASET_CFG["cifar10"]
    run_cifar_resnet_scenario(
        epochs=epochs, port=cfgd["port"], x_trust=x_trust, y_trust=y_trust,
        epsilon_low=eps, base_channels=base_channels, noise_level=noise_level)
    ok = os.path.exists(omega_path) and os.path.exists(nn_model_path)
    if not ok:
        print(f"[PaTAS/CIFAR] Training attempt did NOT produce {omega_path} "
              f"and/or {nn_model_path} — the PTAS server/client scenario "
              f"likely errored or was interrupted (check for "
              f"'[PTAS ERROR]' / '[CLIENT ERROR]' above, or a port "
              f"{cfgd['port']} conflict). Skipping PaTAS filter for this "
              f"condition.")
    return ok


# ---------------------------------------------------------------------------
# LaTeX summary table
# ---------------------------------------------------------------------------

def write_tex_table(metrics_by_cond: dict, dataset: str, out_path: str,
                    train_label: str = "Clean training data",
                    train_tag: str = "") -> None:
    """metrics_by_cond: {noise_level: {method: metric dict}} (0.0 = clean)."""
    def _f(v, fmt="{:.3f}"):
        return fmt.format(v) if v is not None and np.isfinite(v) else "—"

    label_map = {
        "patas":      "PaTAS filter (ours)",
        "softmax":    "Softmax filter",
        "mc_dropout": "MC Dropout~\\cite{gal2016dropout}",
        "edl":        "EDL~\\cite{sensoy2018evidential}",
    }
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Selective-prediction quality and calibration on "
        + dataset.upper() + r" (models trained on: " + train_label + r"), "
        r"on clean test data and under test-time feature noise "
        r"(each pixel corrupted with probability $p$ by uniform noise, "
        r"as in the training-noise model): area under the ROC curve for "
        r"correct-vs-incorrect discrimination (AUROC, higher is better), "
        r"area under the risk--coverage curve (AURC, lower is better), and "
        r"Expected Calibration Error before and after post-hoc isotonic "
        r"recalibration (ECE / ECE$_{\text{cal}}$, lower is better; the "
        r"calibrator for each method is fit once on a held-out clean split "
        r"and reused unchanged across all test-noise conditions, so no "
        r"method is treated differently). DeepTrust has "
        r"no public implementation; values are those reported by Cheng et~al.}",
        rf"\label{{tab:uq-comparison-{dataset}{'-' + train_tag if train_tag else ''}}}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Accuracy (\%) & AUROC $\uparrow$ & AURC $\downarrow$ & "
        r"ECE $\downarrow$ & ECE$_{\text{cal}}$ $\downarrow$ \\",
    ]
    for noise, metrics in metrics_by_cond.items():
        cond = (r"\textit{Clean test data}" if noise == 0 else
                rf"\textit{{Feature noise $p={noise:g}$}}")
        lines += [r"\midrule", rf"\multicolumn{{6}}{{l}}{{{cond}}} \\"]
        for method in ("patas", "softmax", "mc_dropout", "edl"):
            if method not in metrics:
                continue
            m = metrics[method]
            lines.append(
                f"{label_map[method]} & {_f(m.get('test_acc', np.nan)*100, '{:.2f}')} & "
                f"{_f(m.get('auroc'))} & {_f(m.get('aurc'))} & {_f(m.get('ece'))} & "
                f"{_f(m.get('ece_calib'))} \\\\")
        if noise == 0:
            dt = DEEPTRUST_PAPER.get(dataset)
            if dt:
                acc = dt.get("test_acc")
                lines.append(
                    f"DeepTrust~\\cite{{cheng2020deeptrust}}$^{{\\dagger}}$ & "
                    f"{_f(acc*100 if acc is not None else None, '{:.2f}')} & "
                    f"{_f(dt.get('auroc'))} & — & {_f(dt.get('ece'))} & — \\\\")
            else:
                lines.append(r"DeepTrust~\cite{cheng2020deeptrust}$^{\dagger}$ & — & — & — & — & — \\")
    lines += [
        r"\bottomrule",
        r"\multicolumn{6}{l}{\footnotesize $^{\dagger}$No public code; "
        r"values as reported in the original paper (fill \texttt{DEEPTRUST\_PAPER}).}\\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main per-dataset run
# ---------------------------------------------------------------------------

def score_all_methods(Xn, ys, dataset, arch, cfgd, args, is_conv,
                      base, ptas, mc_model, edl_model, cond) -> dict:
    """Score every available method on one (possibly noised) test batch.

    Returns {method: {'score','conf','correct','test_acc'}}.  Labels stay
    clean — only the features are corrupted — so 'correct' measures true
    accuracy under the corruption and the scores are what each filter would
    see at deployment time.  ``cond`` (TrainCondition) selects which PaTAS
    cache (trained under that x_trust/y_trust condition) provides the trust
    factor for the conv (CIFAR-10) fallback.
    """
    results: dict = {}

    # ---- Base model / softmax filter ----------------------------------------
    probs_base = base.forward(Xn)
    correct_base = probs_base.argmax(1) == ys
    results["softmax"] = {
        "score": probs_base.max(1), "conf": probs_base.max(1),
        "correct": correct_base, "test_acc": float(correct_base.mean()),
    }

    # ---- PaTAS filter: trust-discounted confidence ---------------------------
    sc = None
    if is_conv:
        sc = patas_class_trust_scores(dataset, arch, args.eps, probs_base,
                                      fuse_method=args.fuse_method,
                                      x_trust=cond.x_trust, y_trust=cond.y_trust)
    elif ptas is not None:
        t0 = time.time()
        sc = patas_ipta_scores(ptas, base, Xn, cfgd["input_dim"])
        print(f"  PaTAS IPTA scoring: {time.time()-t0:.1f}s "
              f"({np.isnan(sc['score']).sum()} failures)  "
              f"mean path trust={np.nanmean(sc['trust']):.4f}")
    if sc is not None:
        results["patas"] = {
            "score": sc["score"], "conf": sc["score"],
            "trust": sc["trust"],       # raw IPTA/class trust — diagnostic
            "correct": correct_base,
            "test_acc": float(correct_base.mean()),
        }

    # ---- MC Dropout ----------------------------------------------------------
    mc = mc_dropout_predict(mc_model, Xn, T=args.mc_passes, seed=args.seed)
    mc_correct = mc["probs"].argmax(1) == ys
    results["mc_dropout"] = {
        "score": mc["score"], "conf": mc["probs"].max(1), "correct": mc_correct,
        "test_acc": float(mc_correct.mean()),
    }

    # ---- EDL -------------------------------------------------------------------
    edl = edl_predict(edl_model, Xn)
    edl_correct = edl["probs"].argmax(1) == ys
    results["edl"] = {
        "score": edl["score"], "conf": edl["probs"].max(1), "correct": edl_correct,
        "test_acc": float(edl_correct.mean()),
    }
    return results


def compute_metrics(results: dict, calibrators: Optional[dict] = None) -> dict:
    """calibrators: optional {method: fitted isotonic calibrator}, fit once
    on a held-out clean split (see run_condition) and reused across every
    test-noise condition. When given, adds 'ece_calib' — ECE after mapping
    each method's own 'conf' through its own calibrator, so every method
    gets the same post-hoc treatment (no method is singled out)."""
    metrics: dict = {}
    for method, res in results.items():
        _, _, auroc = roc_correctness(res["score"], res["correct"])
        m = {
            "test_acc": res["test_acc"],
            "auroc": auroc,
            "aurc": aurc(res["score"], res["correct"]),
            "ece": ece_from_confidence(res["conf"], res["correct"]),
        }
        if calibrators and calibrators.get(method) is not None:
            calibrated_conf = apply_calibrator(calibrators[method], res["conf"])
            m["ece_calib"] = ece_from_confidence(calibrated_conf, res["correct"])
        if "trust" in res:
            m["mean_trust"] = float(np.nanmean(res["trust"]))
        metrics[method] = m
    return metrics


def run_condition(dataset: str, cond: TrainCondition, args) -> dict:
    """Run the full filter comparison for one *training* condition (clean
    data, or train-time label noise), evaluated across every *test* noise
    condition (clean + --test-noise feature corruption)."""
    from NN.datasets import load_data
    from main import TRUST_TO_DATASET
    from uq_methods import apply_feature_noise, plot_metrics_vs_noise

    cfgd = DATASET_CFG[dataset]
    arch = cfgd["arch"]
    out_dir = f"results/UQ_Compare_{dataset}_{_arch_str(arch)}"
    if cond.tag:
        out_dir += f"_{cond.tag}"
    os.makedirs(out_dir, exist_ok=True)

    noise_levels = sorted(set(float(p) for p in args.test_noise) | {0.0})
    print(f"\n{'='*70}\n  UQ comparison — {dataset} ({_arch_str(arch)})  "
          f"training: {cond.label}  eps={args.eps}  epochs={args.epochs}  "
          f"subset={args.subset}\n"
          f"  test conditions: clean + feature noise p ∈ "
          f"{[p for p in noise_levels if p > 0]}\n{'='*70}")

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

    # Disjoint calibration split (clean features) for post-hoc ECE recalibration —
    # drawn once here, before any noise is applied, and never scored/reported on
    # directly so it can't leak into the evaluation metrics above.
    remaining = np.setdiff1d(np.arange(len(X_test)), idx, assume_unique=True)
    n_calib = min(n_sub, len(remaining))
    calib_idx = rng.choice(remaining, size=n_calib, replace=False) if n_calib > 0 else remaining
    Xc, yc = X_test[calib_idx], y_test_lbl[calib_idx]

    is_conv = dataset == "cifar10"

    # ---- Models (trained under this condition, loaded/trained once) --------
    specs = None
    if is_conv:
        ensure_cifar_cache(cond.x_trust, cond.y_trust, args.eps, args.epochs,
                           cfgd["base_channels"], args.train_missing,
                           cond.noise_level, force_retrain=args.force_retrain_all)
        base, specs = get_base_convnet(X_train, y_train, X_test, y_test,
                                       args.epochs, args.train_missing,
                                       cond.x_trust, cond.y_trust,
                                       force_retrain=args.force_retrain_all)
    else:
        base = get_base_mlp(dataset, arch, X_train, y_train, X_test, y_test,
                            args.epochs, args.train_missing,
                            cond.x_trust, cond.y_trust,
                            force_retrain=args.force_retrain_all)

    ptas = None
    if not is_conv and ensure_patas_cache(dataset, arch, args.eps, args.epochs,
                                          args.train_missing,
                                          fuse_method=args.fuse_method,
                                          x_trust=cond.x_trust, y_trust=cond.y_trust,
                                          noise_level=cond.noise_level,
                                          force_retrain=args.force_retrain_all):
        ptas = load_offline_ptas(dataset, arch, args.eps,
                                 fuse_method=args.fuse_method,
                                 x_trust=cond.x_trust, y_trust=cond.y_trust)

    def _factory_dropout():
        if is_conv:
            return TorchResNetLite(specs, cfgd["img_size"], cfgd["in_channels"],
                                   dropout_p=args.dropout)
        return TorchMLP(cfgd["input_dim"], list(arch), cfgd["output_dim"],
                        dropout_p=args.dropout)

    uq_tag = f"_{cond.tag}" if cond.tag else ""
    mc_path = f"results/UQ_Train_{dataset}_{_arch_str(arch)}{uq_tag}_mcdropout/model.pt"
    mc_model = load_or_train(
        _factory_dropout, mc_path,
        lambda m: train_torch_model(
            m, X_train, y_train, X_test, y_test, epochs=args.epochs,
            batch_size=128, lr_scheduler=_lr_for(dataset), loss_type="ce",
            seed=args.seed),
        force_retrain=args.force_retrain)

    def _factory_edl():
        if is_conv:
            return TorchResNetLite(specs, cfgd["img_size"], cfgd["in_channels"],
                                   dropout_p=0.0)
        return TorchMLP(cfgd["input_dim"], list(arch), cfgd["output_dim"],
                        dropout_p=0.0)

    edl_path = f"results/UQ_Train_{dataset}_{_arch_str(arch)}{uq_tag}_edl/model.pt"
    edl_model = load_or_train(
        _factory_edl, edl_path,
        lambda m: train_torch_model(
            m, X_train, y_train, X_test, y_test, epochs=args.epochs,
            batch_size=128, lr_scheduler=_lr_for(dataset), loss_type="edl",
            annealing_epochs=max(args.epochs // 2, 1), seed=args.seed),
        force_retrain=args.force_retrain)

    # ---- Fit one post-hoc calibrator per method on the held-out clean split --
    # Every method (softmax, patas, mc_dropout, edl) is calibrated the same
    # way, fit once here and reused unchanged across every test-noise
    # condition below — so recalibration can't be biased toward any one
    # method, and ECE-after-calibration reflects how well each method's
    # calibration *transfers* under feature noise it wasn't fit on.
    calibrators: dict = {}
    if n_calib > 0:
        calib_results = score_all_methods(Xc, yc, dataset, arch, cfgd, args, is_conv,
                                          base, ptas, mc_model, edl_model, cond)
        calibrators = {method: fit_calibrator(res["conf"], res["correct"])
                       for method, res in calib_results.items()}

    # ---- Evaluate on every test condition -------------------------------------
    title = {"mnist": "MNIST", "gtsrb": "GTSRB", "cifar10": "CIFAR-10"}[dataset]
    full_title = f"{title} — {cond.label}" if cond.tag else title
    metrics_by_cond: dict = {}
    npz_payload = {"idx": idx, "y": ys, "calib_idx": calib_idx}

    for nl in noise_levels:
        tag = "clean" if nl == 0 else f"noise{nl:g}"
        cond_lbl = "clean" if nl == 0 else f"feature noise $p={nl:g}$"
        print(f"\n  ── test condition: {cond_lbl} " + "─" * 40)
        Xn = Xs if nl == 0 else apply_feature_noise(Xs, nl, seed=args.seed)

        results = score_all_methods(Xn, ys, dataset, arch, cfgd, args, is_conv,
                                    base, ptas, mc_model, edl_model, cond)
        metrics = compute_metrics(results, calibrators)
        metrics_by_cond[nl] = metrics

        print(f"\n  {'Method':<14} {'Acc':>8} {'AUROC':>8} {'AURC':>8} {'ECE':>8} {'ECE-cal':>8}")
        print("  " + "-" * 60)
        for method, m in metrics.items():
            print(f"  {method:<14} {m['test_acc']*100:7.2f}% {m['auroc']:8.3f} "
                  f"{m['aurc']:8.3f} {m['ece']:8.3f} {m.get('ece_calib', float('nan')):8.3f}")

        plot_roc_methods(results, os.path.join(out_dir, f"roc_{tag}"),
                         f"{full_title} ({cond_lbl}) — correct-vs-incorrect ROC")
        plot_coverage_methods(results,
                              os.path.join(out_dir, f"coverage_accuracy_{tag}"),
                              f"{full_title} ({cond_lbl}) — accuracy–coverage")
        for m, r in results.items():
            for k, v in r.items():
                if isinstance(v, np.ndarray):
                    npz_payload[f"{tag}_{m}_{k}"] = np.asarray(v)
            if calibrators.get(m) is not None:
                npz_payload[f"{tag}_{m}_conf_calib"] = apply_calibrator(
                    calibrators[m], results[m]["conf"])

    # ---- Cross-condition outputs -------------------------------------------
    if len(noise_levels) > 1:
        plot_metrics_vs_noise(metrics_by_cond,
                              os.path.join(out_dir, "metrics_vs_noise"),
                              f"{full_title} — robustness to test-time feature noise")
    write_tex_table(metrics_by_cond, dataset, os.path.join(out_dir, "ece_table.tex"),
                    train_label=cond.label, train_tag=cond.tag)

    np.savez_compressed(os.path.join(out_dir, "scores.npz"), **npz_payload)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"dataset": dataset, "arch": _arch_str(arch), "eps": args.eps,
                   "fuse_method": args.fuse_method,
                   "epochs": args.epochs, "subset": n_sub,
                   "mc_passes": args.mc_passes, "dropout": args.dropout,
                   "train_condition": {"tag": cond.tag, "x_trust": cond.x_trust,
                                       "y_trust": cond.y_trust,
                                       "noise_level": cond.noise_level,
                                       "label": cond.label},
                   "test_noise": noise_levels,
                   "metrics": {("clean" if nl == 0 else f"noise{nl:g}"): m
                               for nl, m in metrics_by_cond.items()}},
                  fh, indent=2)
    print(f"  Saved {out_dir}/summary.json")
    return metrics_by_cond


def run(dataset: str, args) -> dict:
    """Run every training condition (clean + train-time label noise) for one
    dataset and return {condition_tag: metrics_by_test_noise}."""
    conditions = [CLEAN_CONDITION] + [label_noise_condition(p)
                                      for p in args.label_noise]
    return {cond.tag or "clean": run_condition(dataset, cond, args)
            for cond in conditions}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", choices=list(DATASET_CFG) + ["all"], default="mnist")
    p.add_argument("--arch", type=int, nargs="+", default=None,
                   help="Override MLP hidden dims, e.g. --arch 1000 or --arch 500 500 "
                        "(ignored for cifar10)")
    p.add_argument("--epochs", type=int, default=20,
                   help="Training epochs for models that need training (default 20)")
    p.add_argument("--eps", type=float, default=_DEFAULT_EPS,
                   help=f"PaTAS epsilon of the cached omegas (default {_DEFAULT_EPS})")
    p.add_argument("--fuse-method",
                   choices=["average", "cumulative", "weighted", "compromise", "constraint"],
                   default="average",
                   help="Trust-revision fusion operator used both when training "
                        "the PaTAS omegas and when GenIPTA fuses the per-sample "
                        "activated-path opinions at inference (default: average). "
                        "'compromise' (consensus & compromise fusion) and "
                        "'constraint' (Dempster's-rule-style belief constraint "
                        "fusion) are more aggressive than average/cumulative at "
                        "preserving/amplifying agreement between fused opinions, "
                        "so may resist the variance-collapse-with-width effect "
                        "seen with 'average'. Changing it trains/caches a "
                        "separate PTAS_Eval directory (suffixed _fuse_<method>), "
                        "so pair with --train-missing.")
    p.add_argument("--subset", type=int, default=2000,
                   help="Test samples used for scoring (IPTA is per-sample; default 2000)")
    p.add_argument("--mc-passes", type=int, default=30,
                   help="MC-Dropout forward passes T (default 30)")
    p.add_argument("--dropout", type=float, default=0.2,
                   help="Dropout rate for the MC-Dropout model (default 0.2)")
    p.add_argument("--test-noise", type=float, nargs="+", default=[0.3, 0.6],
                   help="Test-time feature-noise probabilities to evaluate in "
                        "addition to clean data (default: 0.3 0.6)")
    p.add_argument("--label-noise", type=float, nargs="*", default=[0.3],
                   help="Train-time label-flip rates: for each value, a "
                        "second set of models is trained on label-noised "
                        "data (clean features) and run through the same "
                        "filter comparison, in addition to the clean-trained "
                        "models (default: 0.3; pass --label-noise with no "
                        "values to disable)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-missing", action="store_true",
                   help="Train any missing base/PaTAS caches instead of failing")
    p.add_argument("--force-retrain", action="store_true",
                   help="Retrain the MC-Dropout/EDL models even if cached")
    p.add_argument("--force-retrain-all", action="store_true",
                   help="Retrain every cache — base NN, PaTAS, MC-Dropout, "
                        "EDL — even if already cached, instead of just "
                        "MC-Dropout/EDL like --force-retrain. Implies "
                        "--force-retrain and --train-missing.")
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: 2 epochs, 300-sample subset, T=5")
    return p.parse_args()


def main():
    from uq_methods import print_device_info
    print_device_info()
    args = parse_args()
    if args.force_retrain_all:
        args.force_retrain = True
        args.train_missing = True
    if args.quick:
        args.epochs = 2
        args.subset = 300
        args.mc_passes = 5
        args.test_noise = [0.3]
        args.label_noise = [0.3]
    datasets = list(DATASET_CFG) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        if args.arch and ds != "cifar10":
            DATASET_CFG[ds]["arch"] = tuple(args.arch)
        run(ds, args)
    print("\n=== UQ comparison complete ===\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    # PaTAS scenario training spawns a PTAS-server + NN-client process pair.
    # On Linux, multiprocessing defaults to 'fork', which is incompatible
    # with an already-initialized CUDA context (any GPU tensor allocated in
    # this process — e.g. the base NN — poisons every forked child). Windows
    # already defaults to 'spawn'; force it everywhere so GPU runs work.
    multiprocessing.set_start_method("spawn", force=True)
    main()
