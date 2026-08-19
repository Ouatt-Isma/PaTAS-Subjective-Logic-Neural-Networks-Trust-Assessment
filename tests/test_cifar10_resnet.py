"""
CIFAR-10 ResNet PTAS integration test (conv-PTAS prototype).

Architecture (cifar10_resnet_specs, base_channels=16): a three-stage CIFAR
ResNet in the spirit of the classic ResNet-for-CIFAR family —

    conv 3→16 (pool) → resblock 16 → conv 16→32 (pool) → resblock 32
      → conv 32→64 (pool) → resblock 64 → fc 1024→10

7 conv layers + fc = 10 PaTAS opinion matrices, ~155k parameters.

PaTAS mirrors every weight layer (conv layers through their flattened-kernel
opinion matrices, skip connections as averaging fusion; see NN/convPTAS.py).
Exact under spatially uniform input trust; IPTA is not available for conv.

Notes
-----
* Expect ~60-70% test accuracy at 20-30 epochs on CPU without augmentation;
  reaching the classic ResNet-20 ~91% needs a GPU torch build, weight decay,
  LR schedule and augmentation — the architecture here is the faithful,
  PaTAS-assessable counterpart, sized for CPU experiments.
* First run downloads CIFAR-10 (~170 MB) via torchvision and caches it to
  data/cifar10_flat.npz.

Run standalone:
    python tests/test_cifar10_resnet.py                    # trust/trust, 20 epochs
    python tests/test_cifar10_resnet.py --epochs 2         # quick smoke-test
    python tests/test_cifar10_resnet.py --xtrust vacuous --ytrust vacuous
    python tests/test_cifar10_resnet.py --base-channels 32 # wider net
    python tests/test_cifar10_resnet.py --no-ptas          # NN baseline only

Run with pytest (skipped by default; use -m integration):
    pytest tests/test_cifar10_resnet.py -m integration -s
"""

from __future__ import annotations

import os
import sys
import argparse
import multiprocessing
from typing import Any

# ── Path bootstrap ────────────────────────────────────────────────────────────
_v2_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_patas_dir = os.path.join(_v2_dir, "patas_module")
_tests_dir = os.path.dirname(os.path.abspath(__file__))
for _p in (_v2_dir, _patas_dir, _tests_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import patas_module  # noqa: F401 — triggers path bootstrap

try:
    import pytest
except ImportError:
    class _FakeMark:  # type: ignore[no-redef]
        @staticmethod
        def integration(fn): return fn
    class _FakePytest:  # type: ignore[no-redef]
        mark = _FakeMark()
    pytest = _FakePytest()  # type: ignore[assignment]

# ── Constants ─────────────────────────────────────────────────────────────────
_BASE_PORT      = 5131
_DEFAULT_EPS    = 0.05
_DEFAULT_EPOCHS = 20
_IMG_SIZE       = 32
_IN_CHANNELS    = 3
_NUM_CLASSES    = 10
_LR             = 0.05

# Upgraded CIFAR recipe (momentum SGD + weight decay + augmentation with a
# step LR schedule). Off by default so legacy caches stay reproducible;
# jobs pass recipe=True for the defensible-accuracy training.
_RECIPE = {"momentum": 0.9, "weight_decay": 5e-4, "augment": True}


def _recipe_lr(epochs):
    """0.1 stepped down 10x at 60% and 85% of the run."""
    def lr(e):
        if e < 0.6 * epochs:
            return 0.1
        if e < 0.85 * epochs:
            return 0.01
        return 0.001
    return lr


# Conv scenarios are dataset-parameterized; "cifar10" keeps every legacy
# default. MNIST digits are not mirror-invariant, so no horizontal flip.
CONV_DATASETS = {
    "cifar10": {"img": 32, "cin": 3, "classes": 10, "hflip": True},
    "mnist":   {"img": 28, "cin": 1, "classes": 10, "hflip": False},
    "fashion": {"img": 28, "cin": 1, "classes": 10, "hflip": True},
}


def conv_specs(dataset, base_channels):
    from NN.convNN import cifar10_resnet_specs, default_resnet_lite_specs
    d = CONV_DATASETS[dataset]
    fn = cifar10_resnet_specs if dataset == "cifar10" else default_resnet_lite_specs
    return fn(img_size=d["img"], in_channels=d["cin"],
              num_classes=d["classes"], base_channels=base_channels)


def _recipe_kwargs(dataset):
    kw = dict(_RECIPE)
    kw["hflip"] = CONV_DATASETS[dataset]["hflip"]
    return kw


# ─────────────────────────────────────────────────────────────────────────────
# Result caching — same naming scheme as start_ptas / start_client so that
# run_uq_comparison.py (and future scripts) can reuse the artefacts.
# ─────────────────────────────────────────────────────────────────────────────

def _cifar_paths(x_trust: str, y_trust: str, epsilon_low: float,
                 noise_level: float | None = None,
                 dataset: str = "cifar10") -> tuple[str, str]:
    """(PTAS_Eval dir, NN_Train dir) for a CIFAR-10 ResNet-lite scenario —
    delegates to the canonical helpers in patas_module/main.py so the naming
    (including the _nl<rate> corruption suffix) can never drift from
    start_ptas / start_client / run_uq_comparison."""
    from main import nn_cache_dir, ptas_cache_dir
    return (ptas_cache_dir(dataset, "resnet-lite", x_trust, y_trust,
                           epsilon_low, noise_level=noise_level),
            nn_cache_dir(dataset, "resnet-lite", x_trust, y_trust,
                         noise_level=noise_level))

# ─────────────────────────────────────────────────────────────────────────────
# Subprocess workers
# ─────────────────────────────────────────────────────────────────────────────

def _ptas_worker(result_queue, ready_event, port, x_trust, y_trust,
                 epsilon_low, base_channels, noise_level=None,
                 dataset="cifar10") -> None:
    """PTASConv server subprocess mirroring the CIFAR-10 ResNet.

    Mirrors start_ptas: loads omega_arrays.pkl when cached (skipping training)
    and always writes the evaluation files (evaluation_log.txt, at/av/ad.pkl,
    omega_arrays.pkl) into the PTAS_Eval results directory.
    """
    try:
        import pickle

        from NN.convPTAS import PTASConv
        from NN.convNN import cifar10_resnet_specs, spec_omega_shapes
        from NN.PTAStemplate import PTAS as PTASClass
        from PTASTemp.ptasInterface import PTASInterface
        from concrete.TensorTO import TensorArrayTO, fill as tfill, as_tensor
        from main import build_trust_generator, ptas_evaluation

        specs = conv_specs(dataset, base_channels)
        out_dim = specs[-1]["out"]
        x_gen = build_trust_generator(x_trust)
        y_gen = build_trust_generator(y_trust)

        def trust_assessment(x, dim):
            n = len(x)
            if dim == out_dim:
                return y_gen(n, dim)
            return x_gen(n, dim)

        ptas = PTASConv(
            layer_specs=specs,
            nn_interface=PTASInterface(port),
            trust_assessment_func=trust_assessment,
            epsilon_low=epsilon_low,
        )
        print(f"[PTASConv] Device: {ptas.device}")

        datapath, _ = _cifar_paths(x_trust, y_trust, epsilon_low, noise_level,
                                   dataset=dataset)
        omega_path = os.path.join(datapath, "omega_arrays.pkl")
        expected_shapes = [(r, c, 3) for r, c in spec_omega_shapes(specs)]

        loaded = False
        if os.path.exists(omega_path):
            with open(omega_path, "rb") as fh:
                omega_arrays = pickle.load(fh)
            shapes_ok = (
                len(omega_arrays) == len(expected_shapes)
                and all(tuple(arr.shape) == exp
                        for arr, exp in zip(omega_arrays, expected_shapes))
            )
            if shapes_ok:
                print(f"[PTAS] Loading saved weights from {omega_path} — skipping training.")
                for i, arr in enumerate(omega_arrays):
                    ptas.omega_thetas[i] = TensorArrayTO(as_tensor(arr, device=ptas.device))
                ready_event.set()
                loaded = True
            else:
                print("[PTAS] omega_arrays.pkl shape mismatch — "
                      "deleting stale file and retraining.")
                os.remove(omega_path)

        if not loaded:
            ptas.run_chunk(ready_event=ready_event)

        d = CONV_DATASETS[dataset]
        input_dim = d["cin"] * d["img"] * d["img"]
        # Writes evaluation_log.txt, at/av/ad.pkl and omega_arrays.pkl
        ptas_evaluation(ptas, input_dim, datapath=datapath)
        results = {}
        for label, method in (("trust_mass", "trust"),
                              ("vacuous_mass", "vacuous"),
                              ("distrust_mass", "distrust")):
            a = ptas.apply_feedforward(TensorArrayTO(tfill((1, input_dim), method=method)))
            agg = PTASClass.aggregation(a)
            # trust component for trusted input, distrust component for distrusted
            comp = 1 if label == "distrust_mass" else 0
            results[label] = float(agg[comp])
            if label != "vacuous_mass":
                # depth-normalized counterpart: (b+d)^(1/L), b:d ratio preserved.
                # The vacuous probe has no committed mass, so no norm for it.
                norm = ptas.depth_normalized_aggregation(a)
                results[label.replace("_mass", "_norm")] = float(norm[comp])
        result_queue.put(results)
    except Exception as exc:
        import traceback
        result_queue.put({"trust_mass": float("nan"), "trust_norm": float("nan"),
                          "error": str(exc), "tb": traceback.format_exc()})


def _client_worker(result_queue, port, epochs, x_trust, y_trust,
                   base_channels, ptas=True, epsilon_low=_DEFAULT_EPS,
                   noise_level=None, recipe=False, dataset="cifar10") -> None:
    """CIFAR-10 ResNet client subprocess.

    Mirrors start_client: caches nn_model.pkl + metrics.txt in the NN_Train
    results directory.  When the PTAS omegas are already cached the server
    never binds a socket, so PTAS streaming is disabled; when only the model
    is cached, training is replayed to feed the gradient stream to PTAS.
    """
    try:
        from NN.convNN import ConvNet, cifar10_resnet_specs
        from NN.datasets import load_data
        from NN.utils import writedict
        from main import TRUST_TO_DATASET

        ptas_dir, datapath = _cifar_paths(x_trust, y_trust, epsilon_low,
                                          noise_level, dataset=dataset)
        os.makedirs(datapath, exist_ok=True)
        nn_model_path = os.path.join(datapath, "nn_model.pkl")
        metrics_path = os.path.join(datapath, "metrics.txt")
        # Require metrics.txt alongside nn_model.pkl (see start_client).
        model_cached = os.path.exists(nn_model_path) and os.path.exists(metrics_path)
        ptas_omega_cached = os.path.exists(os.path.join(ptas_dir, "omega_arrays.pkl"))

        x_how = TRUST_TO_DATASET.get(x_trust, "clean")
        y_how = TRUST_TO_DATASET.get(y_trust, "clean")
        _load_kwargs = {} if noise_level is None else {"noise_level": noise_level}
        X_train, X_test, y_train, y_test, _ = load_data(
            dataset, x_how, y_how, **_load_kwargs)

        d = CONV_DATASETS[dataset]
        specs = conv_specs(dataset, base_channels)
        net = ConvNet(img_size=d["img"], in_channels=d["cin"],
                      num_classes=d["classes"], specs=specs,
                      ptas=ptas and not ptas_omega_cached,
                      operation=True, port=port)
        print(f"[ConvNet] Device: {net.device}")

        if model_cached:
            print(f"[NN] Saved model found — loading from {nn_model_path}, skipping training.")
            net.load_model(nn_model_path)
            if ptas and not ptas_omega_cached:
                # Replay training to feed the gradient stream to PTAS
                net.train(X_train, y_train, X_test, y_test,
                          epochs=epochs, batch_size=128,
                          lr_scheduler=(_recipe_lr(epochs) if recipe else (lambda e: _LR)),
                          **(_recipe_kwargs(dataset) if recipe else {}))
            net.end()
            m = {}
            with open(metrics_path, encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("Train:"):
                        m["train_acc"] = float(line.split(":", 1)[1])
                    elif line.startswith("Test:"):
                        m["test_acc"] = float(line.split(":", 1)[1])
            result_queue.put(m)
        else:
            hist = net.train(X_train, y_train, X_test, y_test,
                             epochs=epochs, batch_size=128,
                             lr_scheduler=(_recipe_lr(epochs) if recipe else (lambda e: _LR)),
                          **(_recipe_kwargs(dataset) if recipe else {}))
            net.end()
            train_acc = hist["train_acc"][-1] if hist["train_acc"] else float("nan")
            test_acc = hist["test_acc"][-1] if hist["test_acc"] else float("nan")
            net.save_model(nn_model_path)
            writedict({"Train": train_acc, "Test": test_acc}, metrics_path)
            result_queue.put({"train_acc": train_acc, "test_acc": test_acc})
    except Exception as exc:
        import traceback
        result_queue.put({"train_acc": float("nan"), "test_acc": float("nan"),
                          "error": str(exc), "tb": traceback.format_exc()})

# ─────────────────────────────────────────────────────────────────────────────
# Scenario runners
# ─────────────────────────────────────────────────────────────────────────────

def run_cifar_resnet_scenario(epochs: int = _DEFAULT_EPOCHS,
                              port: int = _BASE_PORT,
                              x_trust: str = "trust", y_trust: str = "trust",
                              epsilon_low: float = _DEFAULT_EPS,
                              base_channels: int = 16,
                              noise_level: float | None = None,
                              recipe: bool = False,
                              dataset: str = "cifar10") -> dict[str, Any]:
    """Two-process PTASConv + CIFAR-ResNet run; returns trust masses + accuracies."""
    ptas_q: "multiprocessing.Queue[dict]" = multiprocessing.Queue()
    client_q: "multiprocessing.Queue[dict]" = multiprocessing.Queue()
    ready_event = multiprocessing.Event()

    ptas_proc = multiprocessing.Process(
        target=_ptas_worker,
        args=(ptas_q, ready_event, port, x_trust, y_trust, epsilon_low,
              base_channels, noise_level, dataset))
    ptas_proc.start()
    ready_event.wait(timeout=60)

    client_proc = multiprocessing.Process(
        target=_client_worker,
        args=(client_q, port, epochs, x_trust, y_trust, base_channels,
              True, epsilon_low, noise_level, recipe, dataset))
    client_proc.start()

    _QUEUE_TIMEOUT = 14400   # CIFAR CPU runs are slow; generous upper bound
    try:
        ptas_res = ptas_q.get(timeout=_QUEUE_TIMEOUT)
    except Exception:
        ptas_res = {}
    try:
        client_res = client_q.get(timeout=_QUEUE_TIMEOUT)
    except Exception:
        client_res = {}

    client_proc.join(timeout=60)
    ptas_proc.join(timeout=60)
    for proc in (client_proc, ptas_proc):
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)

    for res, tag in ((ptas_res, "PTAS"), (client_res, "CLIENT")):
        if "error" in res:
            print(f"  [{tag} ERROR] {res['error']}")
            print(res.get("tb", ""))

    return {
        "arch":          f"cifar-resnet-{base_channels}",
        "x_trust":       x_trust,
        "y_trust":       y_trust,
        "trust_mass":    ptas_res.get("trust_mass", float("nan")),
        "trust_norm":    ptas_res.get("trust_norm", float("nan")),
        "vacuous_mass":  ptas_res.get("vacuous_mass", float("nan")),
        "distrust_mass": ptas_res.get("distrust_mass", float("nan")),
        "distrust_norm": ptas_res.get("distrust_norm", float("nan")),
        "train_acc":     client_res.get("train_acc", float("nan")),
        "test_acc":      client_res.get("test_acc", float("nan")),
    }


def run_baseline_no_ptas(epochs: int = _DEFAULT_EPOCHS,
                         base_channels: int = 16) -> dict[str, Any]:
    """CIFAR-10 ResNet accuracy baseline without PTAS (single subprocess)."""
    client_q: "multiprocessing.Queue[dict]" = multiprocessing.Queue()
    client_proc = multiprocessing.Process(
        target=_client_worker,
        args=(client_q, 0, epochs, "trust", "trust", base_channels, False))
    client_proc.start()
    try:
        client_res = client_q.get(timeout=14400)
    except Exception:
        client_res = {}
    client_proc.join(timeout=60)
    if client_proc.is_alive():
        client_proc.terminate()
        client_proc.join(timeout=10)
    if "error" in client_res:
        print(f"  [CLIENT ERROR] {client_res['error']}")
        print(client_res.get("tb", ""))
    return {
        "arch":       f"cifar-resnet-{base_channels}",
        "train_acc":  client_res.get("train_acc", float("nan")),
        "test_acc":   client_res.get("test_acc", float("nan")),
        "trust_mass": float("nan"),
        "trust_norm": float("nan"),
    }

# ─────────────────────────────────────────────────────────────────────────────
# pytest integration tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_cifar10_resnet_trust():
    """CIFAR-10 ResNet with conv-PTAS, trust/trust (2 epochs)."""
    result = run_cifar_resnet_scenario(epochs=2, port=5135)
    assert result["train_acc"] > 0.25, f"Low train acc: {result['train_acc']}"
    assert 0.0 <= result["trust_mass"] <= 1.0, f"Trust mass OOB: {result['trust_mass']}"


@pytest.mark.integration
def test_cifar10_resnet_baseline():
    """CIFAR-10 ResNet baseline without PTAS (2 epochs)."""
    result = run_baseline_no_ptas(epochs=2)
    assert result["train_acc"] > 0.25, f"Low train acc: {result['train_acc']}"

# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CIFAR-10 ResNet PTAS test (conv-PTAS prototype).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--xtrust", default="trust",
                   help="X trust: trust | distrust | vacuous | t,d,u")
    p.add_argument("--ytrust", default="trust",
                   help="Y trust: trust | distrust | vacuous | t,d,u")
    p.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    p.add_argument("--epsilon-low", type=float, default=_DEFAULT_EPS)
    p.add_argument("--port", type=int, default=_BASE_PORT)
    p.add_argument("--base-channels", type=int, default=16,
                   help="Stage-1 width; stages use 1×/2×/4× this (default 16)")
    p.add_argument("--no-ptas", action="store_true",
                   help="Accuracy baseline without PTAS")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"\n{'='*64}")
    print(f"  CIFAR-10 RESNET TEST  |  x={args.xtrust}  y={args.ytrust}")
    print(f"  base_channels={args.base_channels}  epochs={args.epochs}  "
          f"ε={args.epsilon_low}  port={args.port}  ptas={not args.no_ptas}")
    print(f"{'='*64}\n")

    if args.no_ptas:
        result = run_baseline_no_ptas(epochs=args.epochs,
                                      base_channels=args.base_channels)
    else:
        result = run_cifar_resnet_scenario(
            epochs=args.epochs, port=args.port,
            x_trust=args.xtrust, y_trust=args.ytrust,
            epsilon_low=args.epsilon_low,
            base_channels=args.base_channels,
        )

    print(f"\n{'='*64}")
    print("  Results")
    print(f"{'='*64}")
    print(f"  Architecture  : {result['arch']}")
    print(f"  Train Acc     : {result['train_acc']*100:.2f}%")
    print(f"  Test Acc      : {result['test_acc']*100:.2f}%")
    if not args.no_ptas:
        print(f"  Trust mass (trusted input)      : {result['trust_mass']:.4f}")
        print(f"  Norm trust (depth-normalized)   : {result['trust_norm']:.4f}"
              f"  ((b+d)^(1/L), comparable across depths)")
        print(f"  Uncertainty (vacuous input)     : {1.0 - result['vacuous_mass']:.4f}"
              f"  (trust component {result['vacuous_mass']:.4f})")
        print(f"  Distrust mass (distrusted input): {result['distrust_mass']:.4f}")
        print(f"  Norm distrust (depth-normalized): {result['distrust_norm']:.4f}")
    print(f"{'='*64}\n")
    print("=== CIFAR-10 ResNet test complete ===\n")


if __name__ == "__main__":
    # Force 'spawn' — 'fork' (Linux default) breaks once CUDA is initialized
    # in this process, which happens as soon as a torch model is built.
    multiprocessing.set_start_method("spawn", force=True)
    main()
