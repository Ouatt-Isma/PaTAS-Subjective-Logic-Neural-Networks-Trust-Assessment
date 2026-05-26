"""
Bridge adapter for external PTAS and NeuralNetwork implementations.

Usage:
    python train2.py --external data/
    python train2.py --external           # synthetic data
"""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import sys

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Union

from patas_module.NN.primaryNN import NeuralNetwork
from patas_module.PTASTemp.ptasInterface import PTASInterface
from patas_module.NN.PTAStemplate import PTAS
from patas_module.concrete.TrustOpinion import TrustOpinion
from patas_module.concrete.ArrayTO import ArrayTO
from patas_module.main import ptas_evaluation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trust_label(trust) -> str:
    """Compact, filename-safe label for a TrustOpinion or a string type."""
    if isinstance(trust, str):
        return trust
    return f"b{trust.t:.3f}_d{trust.d:.3f}_u{trust.u:.3f}"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExternalConfig:
    """Full configuration for one PTAS+NN experiment."""
    dataset: str
    input_dim: int
    output_dim: int
    hidden_dim: int
    epochs: int
    batch_size: int
    learning_rate: Callable[[int], float]
    epsilon_low: float

    # Trust types: accept either a named string or a TrustOpinion object.
    # Use x_trust="percal" together with sigma_per_sample to enable per-sample
    # feature trust calibration via index lookup.
    x_trust: Union[str, TrustOpinion] = "trusted"
    y_trust: Union[str, TrustOpinion] = "trusted"

    epsilon_up: float = 100.0
    port: int = 6543
    mnist_patch_size: Optional[int] = None
    mnist_poisoned_soph: bool = False
    no_round: Optional[int] = None
    binary: bool = False
    ipta: bool = False

    # Optional label used for the output directory; auto-generated if empty
    run_label: str = ""

    # Enable PTAS trust-tracking mode (captures trust mass evolution during training)
    eval: bool = False

    # Per-sample feature trust: sigma_per_sample[i] is the noise level applied
    # to training sample i.  Set alongside x_trust="percal".
    sigma_per_sample: Optional[np.ndarray] = None

    # When True, PTAS sends back the output trust scalar with each layer-1 ACK
    # and the NN scales its gradient update by that scalar.
    # trust_feedback: bool = False


# ---------------------------------------------------------------------------
# Learning-rate scheduler (must be picklable for multiprocessing)
# ---------------------------------------------------------------------------

class _ConstantLR:
    """Picklable constant learning-rate scheduler."""
    def __init__(self, base_lr: float):
        self.base_lr = base_lr

    def __call__(self, _epoch: int) -> float:
        return self.base_lr


def lr_constant(base_lr: float) -> _ConstantLR:
    return _ConstantLR(base_lr)


# ---------------------------------------------------------------------------
# Trust generators
# ---------------------------------------------------------------------------

def build_trust_generator(trust_type: Union[str, TrustOpinion]):
    """Return a callable (indices_or_n, dim) -> ArrayTO for the given trust type.

    Accepts either a named string ('trusted', 'vacuous', 'distrusted') or
    a TrustOpinion object for custom calibrated opinions.

    The first argument may be an integer n (legacy) or a numpy array of batch
    indices (sent by primaryNN in TRAINING_FEEDFORWARD messages); both are
    handled identically for uniform trust opinions.
    """
    if isinstance(trust_type, TrustOpinion):
        opinion = trust_type
        def gen(indices_or_n, dim: int) -> ArrayTO:
            n = int(indices_or_n) if isinstance(indices_or_n, (int, np.integer)) else len(indices_or_n)
            return ArrayTO(TrustOpinion.fill(shape=(n, dim), value=opinion))
        return gen

    _METHOD = {
        "trusted":    "ftrust",
        "ftrust":     "ftrust",
        "vacuous":    "vacuous",
        "distrusted": "fdistrust",
        "fdistrust":  "fdistrust",
    }
    method = _METHOD.get(trust_type)
    if method is None:
        raise ValueError(f"Unknown trust type: {trust_type!r}")

    def gen(indices_or_n, dim: int) -> ArrayTO:
        n = int(indices_or_n) if isinstance(indices_or_n, (int, np.integer)) else len(indices_or_n)
        return ArrayTO(TrustOpinion.fill(shape=(n, dim), method=method))
    return gen


def build_mnist_poisoned_soph_generator(patch_size: int):
    raise NotImplementedError(
        "Replace with your actual poisoned-MNIST trust generator."
    )


# ---------------------------------------------------------------------------
# Output path helper
# ---------------------------------------------------------------------------

def _datapath(cfg: ExternalConfig, use_ptas: bool) -> str:
    if cfg.run_label:
        # If run_label already contains a path separator, use it as a full path.
        if os.sep in cfg.run_label or '/' in cfg.run_label:
            return cfg.run_label
        return os.path.join("results", cfg.run_label)
    mode = "ptas" if use_ptas else "nn"
    xl = _trust_label(cfg.x_trust)
    yl = _trust_label(cfg.y_trust)
    return f"results/{mode}_{cfg.dataset}_{cfg.hidden_dim}_{xl}_{yl}"


# ---------------------------------------------------------------------------
# PTAS server process
# ---------------------------------------------------------------------------

def start_ptas_server(cfg: ExternalConfig, ready_event: multiprocessing.Event):
    """Build and run the PTAS server inside its own process."""
    print("=" * 70)
    print(f"PTAS Server  [{cfg.dataset}]  structure: "
          f"[{cfg.input_dim}, {cfg.hidden_dim}, {cfg.output_dim}]")
    print(f"  epsilon: ({cfg.epsilon_low}, {cfg.epsilon_up})  "
          f"epochs: {cfg.epochs}  port: {cfg.port}")
    print(f"  x_trust: {_trust_label(cfg.x_trust)}  "
          f"y_trust: {_trust_label(cfg.y_trust)}")
    print("=" * 70)

    if cfg.dataset == "mnist" and cfg.mnist_poisoned_soph:
        trust_assessment = build_mnist_poisoned_soph_generator(cfg.mnist_patch_size)
    else:
        # Per-sample feature trust: use sigma_per_sample lookup when available.
        if cfg.sigma_per_sample is not None and cfg.x_trust == "percal":
            from noise_utils import build_per_sample_feature_trust_generator
            x_gen = build_per_sample_feature_trust_generator(cfg.sigma_per_sample)
        else:
            x_gen = build_trust_generator(cfg.x_trust)
        y_gen = build_trust_generator(cfg.y_trust)

        def trust_assessment(x: np.ndarray, dim: int) -> ArrayTO:
            # x is batch indices for FEEDFORWARD, y_true for BACKPROPAGATION.
            # Pass x directly so per-sample generators can index into sigma_arr.
            n = len(x)
            if dim == cfg.input_dim:
                return x_gen(x, dim)
            if dim == cfg.output_dim:
                return y_gen(n, dim)
            raise ValueError(f"Unexpected dim {dim}")

    omega_thetas = [
        ArrayTO(TrustOpinion.fill(shape=(cfg.input_dim + 1, cfg.hidden_dim), method="vacuous")),
        ArrayTO(TrustOpinion.fill(shape=(cfg.hidden_dim + 1, cfg.output_dim), method="vacuous")),
    ]

    ptas = PTAS(
        omega_thetas=omega_thetas,
        operator_mapping=None,
        nn_interface=PTASInterface(cfg.port),
        trust_assessment_func=trust_assessment,
        structure=[cfg.input_dim, cfg.hidden_dim, cfg.output_dim],
        epsilon_low=cfg.epsilon_low,
        epsilon_up=cfg.epsilon_up,
        eval=cfg.eval,
        no_round=cfg.no_round,
        # trust_feedback=cfg.trust_feedback,
    )

    print("PTAS server starting on port", cfg.port)
    ptas.run_chunk(ready_event=ready_event)
    print("PTAS server finished.")

    dp = _datapath(cfg, use_ptas=True)
    os.makedirs(dp, exist_ok=True)
    ptas_evaluation(ptas, cfg.input_dim, datapath=dp)

    # Save omega_thetas (learned weight opinions) for post-hoc analysis
    import pickle as _pkl
    omega_data = [w.value.copy() for w in ptas.omega_thetas]
    with open(os.path.join(dp, "omega_thetas.pkl"), "wb") as fh:
        _pkl.dump(omega_data, fh)

    # Save EVAL tracking data (trust mass evolution during training)
    if cfg.eval and hasattr(ptas, "EVAL"):
        eval_data = {
            k: [item.value.copy() for item in v]
            for k, v in ptas.EVAL.items()
        }
        eval_hidden_data = {
            k: [item.value.copy() for item in v]
            for k, v in ptas.EVAL_HIDDEN.items()
        }
        with open(os.path.join(dp, "eval_ptas.pkl"), "wb") as fh:
            _pkl.dump({"EVAL": eval_data, "EVAL_HIDDEN": eval_hidden_data}, fh)

        PTAS.eval_plot_simpl(
            ptas.EVAL, cfg.output_dim,
            f"Trust Propagation – {cfg.dataset}",
            os.path.join(dp, "trust_evolution.pdf"),
        )


# ---------------------------------------------------------------------------
# NN client process
# ---------------------------------------------------------------------------

def start_nn_client(
    cfg: ExternalConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_ptas: bool = True,
):
    """Train the NN (optionally connected to PTAS) and save results to JSON."""
    print("=" * 70)
    print(f"NN Client  [{cfg.dataset}]  "
          f"structure: [{cfg.input_dim}, {cfg.hidden_dim}, {cfg.output_dim}]")
    print(f"  epochs: {cfg.epochs}  batch: {cfg.batch_size}  PTAS: {use_ptas}")
    print("=" * 70)

    nn = NeuralNetwork(
        cfg.input_dim, cfg.hidden_dim, cfg.output_dim,
        ptas=use_ptas,
        operation=True,
        port=cfg.port,
        binary_weights=cfg.binary,
        # trust_feedback=cfg.trust_feedback if use_ptas else False,
    )

    dp = _datapath(cfg, use_ptas)
    os.makedirs(dp, exist_ok=True)

    if cfg.mnist_poisoned_soph:
        from patas_module.NN.datasets import add_trigger_patch
        ids_6 = np.where(np.argmax(y_test, axis=1) == 6)[0]
        ids_3 = np.where(np.argmax(y_test, axis=1) == 3)[0]
        pois_X_6 = np.stack([add_trigger_patch(X_test[i], cfg.mnist_patch_size) for i in ids_6])
        pois_X_3 = np.stack([add_trigger_patch(X_test[i], cfg.mnist_patch_size) for i in ids_3])
        history = nn.train(
            X_train, y_train, X_test, y_test,
            epochs=cfg.epochs, batch_size=cfg.batch_size,
            lr_scheduler=cfg.learning_rate,
            X_non_pois_6=X_test[ids_6], X_pois_6=pois_X_6,
            X_non_pois_3=X_test[ids_3], X_pois_3=pois_X_3,
            plot=True, fname=dp,
        )
    else:
        history = nn.train(
            X_train, y_train, X_test, y_test,
            epochs=cfg.epochs, batch_size=cfg.batch_size,
            lr_scheduler=cfg.learning_rate,
            plot=True, fname=dp,
        )

    # --- per-epoch accuracy (last batch value of each epoch) ---
    n_batches = math.ceil(X_train.shape[0] / cfg.batch_size)
    epoch_test_acc, epoch_train_acc = [], []
    for e in range(cfg.epochs):
        idx = min((e + 1) * n_batches - 1, len(history["test_acc"]) - 1)
        epoch_test_acc.append(float(history["test_acc"][idx]))
        epoch_train_acc.append(float(history["train_acc"][idx]))

    final_train = float(np.mean(nn.predict(X_train) == np.argmax(y_train, axis=1)))
    final_test  = float(np.mean(nn.predict(X_test)  == np.argmax(y_test,  axis=1)))

    print(f"Final Train Accuracy: {final_train * 100:.2f}%")
    print(f"Final Test  Accuracy: {final_test  * 100:.2f}%")

    results = {
        "final_train_acc": final_train,
        "final_test_acc":  final_test,
        "epoch_train_acc": epoch_train_acc,
        "epoch_test_acc":  epoch_test_acc,
        "use_ptas": use_ptas,
        "x_trust": _trust_label(cfg.x_trust),
        "y_trust": _trust_label(cfg.y_trust),
        "run_label": cfg.run_label,
    }
    with open(os.path.join(dp, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # Save NN weights for post-hoc PTAS effectiveness analysis.
    import pickle as _pkl
    nn_weights = {
        "W1": nn.W1.copy(), "b1": nn.b1.copy(),
        "W2": nn.W2.copy(), "b2": nn.b2.copy(),
    }
    with open(os.path.join(dp, "nn_weights.pkl"), "wb") as fh:
        _pkl.dump(nn_weights, fh)

    if cfg.ipta:
        nn.forward(X_test[0], getactivated=True)

    nn.end()


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_with_external_implementation(
    data_dir: Optional[str],
    dataset: str = "5g",
    n_hidden: int = 32,
    epochs: int = 1,
    batch: int = 64,
    lr: float = 0.05,
    eps_low: float = 0.05,
    eps_up: float = 100.0,
    x_trust: Union[str, TrustOpinion] = "trusted",
    y_trust: Union[str, TrustOpinion] = "trusted",
    use_ptas: bool = True,
    port: int = 6543,
    run_label: str = "",
    # Optional pre-loaded data (skips internal loading when provided)
    X_train: Optional[np.ndarray] = None,
    y_train_oh: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    y_test_oh: Optional[np.ndarray] = None,
    # Noise parameters (applied when data is loaded internally)
    sigma_relative: float = 0.0,
    flip_rate: float = 0.0,
    # Enable PTAS trust-tracking mode (saves eval_ptas.pkl and omega_thetas.pkl)
    enable_eval: bool = False,
    # Per-sample trust: pre-computed noise levels for each training sample.
    # Set alongside x_trust="percal".  Ignored when None.
    sigma_per_sample: Optional[np.ndarray] = None,
    # Trust feedback: PTAS returns its output belief mass with each layer-1 ACK
    # and the NN scales its gradient by that scalar before the weight update.
    trust_feedback: bool = False,
) -> dict:
    """Run one PTAS+NN experiment and return the results dict.

    Data loading
    ------------
    If *X_train* is provided the function uses it directly (caller is
    responsible for any noise injection).  Otherwise the 5G dataset is loaded
    from *data_dir* (or generated synthetically) and noise is applied
    according to *sigma_relative* / *flip_rate*.

    Returns
    -------
    dict with keys: final_test_acc, final_train_acc, epoch_test_acc,
    epoch_train_acc, datapath.
    """
    if X_train is None:
        from data_loader import load_5g_dataset, make_synthetic_5g
        from noise_utils import add_feature_noise, add_label_noise

        if data_dir is None:
            print("No data directory provided — generating synthetic 5G data...")
            data_dir = make_synthetic_5g(n_bs=20, n_hours=72, cells_per_bs=2, seed=0)

        ds = load_5g_dataset(data_dir, n_classes=3, test_frac=0.2, seed=0)
        n_classes = int(ds.y_train.max()) + 1

        X_train_raw = add_feature_noise(ds.X_train, sigma_relative)
        y_train_int = add_label_noise(ds.y_train, flip_rate, n_classes)
        X_train  = X_train_raw
        y_train_oh = np.eye(n_classes, dtype=np.float32)[y_train_int]
        X_test   = ds.X_test
        y_test_oh = np.eye(n_classes, dtype=np.float32)[ds.y_test]
    else:
        n_classes = y_train_oh.shape[1]

    cfg = ExternalConfig(
        dataset=dataset,
        input_dim=X_train.shape[1],
        output_dim=n_classes,
        hidden_dim=n_hidden,
        epochs=epochs,
        batch_size=batch,
        learning_rate=lr_constant(lr),
        epsilon_low=eps_low,
        epsilon_up=eps_up,
        x_trust=x_trust,
        y_trust=y_trust,
        port=port,
        run_label=run_label,
        eval=enable_eval,
        sigma_per_sample=sigma_per_sample,
        # trust_feedback=trust_feedback,
    )

    dp = _datapath(cfg, use_ptas)

    if use_ptas:
        ready_event = multiprocessing.Event()
        ptas_proc = multiprocessing.Process(
            target=start_ptas_server, args=(cfg, ready_event)
        )
        ptas_proc.start()

        if not ready_event.wait(timeout=120):
            ptas_proc.terminate()
            raise RuntimeError("PTAS server failed to start within 120 seconds")

        print("PTAS server ready — starting NN client...")
        nn_proc = multiprocessing.Process(
            target=start_nn_client,
            args=(cfg, X_train, y_train_oh, X_test, y_test_oh, True),
        )
        nn_proc.start()
        nn_proc.join()
        ptas_proc.join()
    else:
        start_nn_client(cfg, X_train, y_train_oh, X_test, y_test_oh, use_ptas=False)

    print("\nTraining complete!")

    results_file = os.path.join(dp, "results.json")
    if os.path.exists(results_file):
        with open(results_file) as fh:
            results = json.load(fh)
    else:
        results = {}

    results["datapath"] = dp
    return results


# ---------------------------------------------------------------------------
# Preset configs
# ---------------------------------------------------------------------------

EXTERNAL_CONFIGS = {
    "5g": lambda: ExternalConfig(
        dataset="5g", input_dim=26, output_dim=3, hidden_dim=32,
        epochs=12, batch_size=64, learning_rate=lr_constant(0.05),
        epsilon_low=0.05,
    ),
    "mnist": lambda: ExternalConfig(
        dataset="mnist", input_dim=28 * 28, output_dim=10, hidden_dim=128,
        epochs=10, batch_size=128, learning_rate=lr_constant(0.05),
        epsilon_low=0.05,
    ),
    "cancer": lambda: ExternalConfig(
        dataset="cancer", input_dim=30, output_dim=2, hidden_dim=16,
        epochs=15, batch_size=64, learning_rate=lr_constant(0.2),
        epsilon_low=0.1,
    ),
}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run 5G training with external PTAS/NN")
    ap.add_argument("data_dir", nargs="?", default=None,
                    help="Path to 5G CSVs (default: synthetic)")
    ap.add_argument("--no-ptas", action="store_true", help="Run NN without PTAS")
    ap.add_argument("--port", type=int, default=6543)
    args = ap.parse_args()

    run_with_external_implementation(
        args.data_dir,
        use_ptas=not args.no_ptas,
        port=args.port,
    )
