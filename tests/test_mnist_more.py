"""
MNIST PTAS integration test — larger architectures + ResNet-lite (conv PaTAS).

Extends test_mnist.py with:
  1) Wide/deep MLPs:   784-1000-10   and   784-500-500-10
  2) A convolutional "ResNet-lite" model (conv → residual block → fc) whose
     PaTAS mirror uses the conv-PTAS prototype (NN/convPTAS.py): each conv
     layer is assessed through its flattened-kernel opinion matrix, the skip
     connection becomes an averaging fusion, and pooling/ReLU are identity in
     opinion space.  Valid under spatially uniform input trust
     (trust / vacuous / distrust); IPTA is not available for conv layers.

Run standalone:
    python tests/test_mnist_more.py                  # full sweep → table
    python tests/test_mnist_more.py --epochs 3       # quick smoke-test
    python tests/test_mnist_more.py --resnet-only    # only the conv scenario
    python tests/test_mnist_more.py --skip-resnet    # only the big MLPs

Run with pytest (skipped by default; use -m integration):
    pytest tests/test_mnist_more.py -m integration -s
"""

from __future__ import annotations

import os
import sys
import time
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

import numpy as np
import patas_module  # noqa: F401 — triggers path bootstrap
from test_mnist import (
    make_mnist_cfg,
    run_scenario,
    print_results_table,
    _read_metrics,
)

try:
    import pytest
except ImportError:
    class _FakeMark:  # type: ignore[no-redef]
        @staticmethod
        def integration(fn): return fn
    class _FakePytest:  # type: ignore[no-redef]
        mark = _FakeMark()
    pytest = _FakePytest()  # type: ignore[assignment]

# ── Scenario definitions ──────────────────────────────────────────────────────

MLP_SCENARIOS: list[dict[str, Any]] = [
    {"hidden_dims": (1000,),    "x_trust": "trust",   "y_trust": "trust"},
    {"hidden_dims": (1000,),    "x_trust": "vacuous", "y_trust": "vacuous"},
    {"hidden_dims": (500, 500), "x_trust": "vacuous", "y_trust": "vacuous"},
    {"hidden_dims": (500, 500), "x_trust": "trust",   "y_trust": "trust"},
]

_BASE_PORT      = 5071
_DEFAULT_EPS    = 0.05
_DEFAULT_EPOCHS = 20

# ─────────────────────────────────────────────────────────────────────────────
# ResNet-lite (conv PaTAS prototype) runner
# ─────────────────────────────────────────────────────────────────────────────

def _resnet_paths(x_trust: str, y_trust: str, epsilon_low: float) -> tuple[str, str]:
    """(PTAS_Eval dir, NN_Train dir) for a ResNet-lite scenario — same naming
    scheme as start_ptas / start_client, with 'resnet-lite' as the arch string."""
    ptas_dir = (
        f"results/PTAS_Eval_mnist_resnet-lite_{x_trust}_{y_trust}"
        f"_eps_{epsilon_low}_PathSize_None"
    )
    nn_dir = f"results/NN_Train_mnist_resnet-lite_{x_trust}_{y_trust}_PathSize_None"
    return ptas_dir, nn_dir


def _resnet_ptas_worker(result_queue, ready_event, port, x_trust, y_trust,
                        epsilon_low) -> None:
    """PTASConv server subprocess for the ResNet-lite scenario.

    Mirrors start_ptas: loads omega_arrays.pkl when cached (skipping training)
    and always writes the evaluation files (evaluation_log.txt, at/av/ad.pkl,
    omega_arrays.pkl) into the PTAS_Eval results directory.
    """
    try:
        import pickle

        from NN.convPTAS import PTASConv
        from NN.convNN import default_resnet_lite_specs, spec_omega_shapes
        from NN.PTAStemplate import PTAS as PTASClass
        from PTASTemp.ptasInterface import PTASInterface
        from concrete.TensorTO import TensorArrayTO, fill as tfill, as_tensor
        from main import build_trust_generator, ptas_evaluation

        specs = default_resnet_lite_specs(img_size=28, num_classes=10)
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

        datapath, _ = _resnet_paths(x_trust, y_trust, epsilon_low)
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

        # Writes evaluation_log.txt, at/av/ad.pkl and omega_arrays.pkl
        ptas_evaluation(ptas, 28 * 28, datapath=datapath)

        a = ptas.apply_feedforward(TensorArrayTO(tfill((1, 28 * 28), method="trust")))
        result_queue.put({
            "trust_mass": float(PTASClass.aggregation(a)[0]),
            "trust_norm": float(ptas.depth_normalized_aggregation(a)[0]),
        })
    except Exception as exc:
        import traceback
        result_queue.put({"trust_mass": float("nan"), "trust_norm": float("nan"),
                          "error": str(exc), "tb": traceback.format_exc()})


def _resnet_client_worker(result_queue, port, epochs, x_trust, y_trust,
                          epsilon_low) -> None:
    """ConvNet client subprocess for the ResNet-lite scenario.

    Mirrors start_client: caches nn_model.pkl + metrics.txt in the NN_Train
    results directory.  When the PTAS omegas are already cached the server
    never binds a socket, so PTAS streaming is disabled; when only the model
    is cached, training is replayed to feed the gradient stream to PTAS.
    """
    try:
        from NN.convNN import ConvNet
        from NN.datasets import load_data
        from NN.utils import writedict
        from main import TRUST_TO_DATASET, get_lr_mnist

        ptas_dir, datapath = _resnet_paths(x_trust, y_trust, epsilon_low)
        os.makedirs(datapath, exist_ok=True)
        nn_model_path = os.path.join(datapath, "nn_model.pkl")
        metrics_path = os.path.join(datapath, "metrics.txt")
        # Require metrics.txt alongside nn_model.pkl (see start_client).
        model_cached = os.path.exists(nn_model_path) and os.path.exists(metrics_path)
        ptas_omega_cached = os.path.exists(os.path.join(ptas_dir, "omega_arrays.pkl"))

        x_how = TRUST_TO_DATASET.get(x_trust, "clean")
        y_how = TRUST_TO_DATASET.get(y_trust, "clean")
        X_train, X_test, y_train, y_test, _ = load_data("mnist", x_how, y_how)

        net = ConvNet(img_size=28, num_classes=10, ptas=not ptas_omega_cached,
                      operation=True, port=port)

        if model_cached:
            print(f"[NN] Saved model found — loading from {nn_model_path}, skipping training.")
            net.load_model(nn_model_path)
            if not ptas_omega_cached:
                # Replay training to feed the gradient stream to PTAS
                net.train(X_train, y_train, X_test, y_test,
                          epochs=epochs, batch_size=128,
                          lr_scheduler=get_lr_mnist)
            net.end()
            result_queue.put(_read_metrics(metrics_path))
        else:
            hist = net.train(X_train, y_train, X_test, y_test,
                             epochs=epochs, batch_size=128,
                             lr_scheduler=get_lr_mnist)
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


def run_resnet_scenario(epochs: int = _DEFAULT_EPOCHS, port: int = _BASE_PORT + 10,
                        x_trust: str = "trust", y_trust: str = "trust",
                        epsilon_low: float = _DEFAULT_EPS) -> dict[str, Any]:
    """Two-process PTASConv + ConvNet run; returns trust mass + accuracies.

    Results are persisted like the MLP scenarios: PTAS omegas + evaluation
    files under results/PTAS_Eval_mnist_resnet-lite_*, model + metrics under
    results/NN_Train_mnist_resnet-lite_*.  Cached runs skip training.
    """
    ptas_q: "multiprocessing.Queue[dict]" = multiprocessing.Queue()
    client_q: "multiprocessing.Queue[dict]" = multiprocessing.Queue()
    ready_event = multiprocessing.Event()

    ptas_proc = multiprocessing.Process(
        target=_resnet_ptas_worker,
        args=(ptas_q, ready_event, port, x_trust, y_trust, epsilon_low))
    ptas_proc.start()
    ready_event.wait(timeout=60)

    client_proc = multiprocessing.Process(
        target=_resnet_client_worker,
        args=(client_q, port, epochs, x_trust, y_trust, epsilon_low))
    client_proc.start()

    _QUEUE_TIMEOUT = 7200
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
        "hidden_dims": ("resnet-lite",),
        "x_trust":    x_trust,
        "y_trust":    y_trust,
        "trust_mass": ptas_res.get("trust_mass", float("nan")),
        "trust_norm": ptas_res.get("trust_norm", float("nan")),
        "train_acc":  client_res.get("train_acc", float("nan")),
        "test_acc":   client_res.get("test_acc", float("nan")),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Full sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_all_scenarios(epochs: int = _DEFAULT_EPOCHS,
                      epsilon_low: float = _DEFAULT_EPS,
                      base_port: int = _BASE_PORT,
                      skip_resnet: bool = False,
                      resnet_only: bool = False) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    if not resnet_only:
        for i, sc in enumerate(MLP_SCENARIOS):
            cfg = make_mnist_cfg(
                x_trust=sc["x_trust"], y_trust=sc["y_trust"],
                hidden_dims=sc["hidden_dims"],
                epsilon_low=epsilon_low, epochs=epochs, port=base_port + i,
            )
            arch = "-".join(str(h) for h in sc["hidden_dims"])
            print(f"\n  ► arch={arch:<10}  x={sc['x_trust']:<8}  y={sc['y_trust']}")
            result = run_scenario(cfg)
            print(f"    trust_mass={result['trust_mass']:.4f}  "
                  f"trust_norm={result['trust_norm']:.4f}  "
                  f"train={result['train_acc']*100:.2f}%  "
                  f"test={result['test_acc']*100:.2f}%")
            results.append(result)
            time.sleep(2)

    if not skip_resnet:
        print(f"\n  ► arch=resnet-lite (conv PaTAS prototype)  x=trust  y=trust")
        result = run_resnet_scenario(epochs=epochs, port=base_port + 10,
                                     epsilon_low=epsilon_low)
        print(f"    trust_mass={result['trust_mass']:.4f}  "
              f"trust_norm={result['trust_norm']:.4f}  "
              f"train={result['train_acc']*100:.2f}%  "
              f"test={result['test_acc']*100:.2f}%")
        results.append(result)

        print(f"\n  ► arch=resnet-lite (conv PaTAS prototype)  x=vacuous  y=vacuous")
        result = run_resnet_scenario(epochs=epochs, port=base_port + 10,
                                     epsilon_low=epsilon_low, x_trust="vacuous", y_trust="vacuous")
        print(f"    trust_mass={result['trust_mass']:.4f}  "
              f"trust_norm={result['trust_norm']:.4f}  "
              f"train={result['train_acc']*100:.2f}%  "
              f"test={result['test_acc']*100:.2f}%")
        results.append(result)

    return results

# ─────────────────────────────────────────────────────────────────────────────
# pytest integration tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_mnist_1000():
    """MNIST: 784-1000-10 MLP, trust/trust."""
    cfg = make_mnist_cfg(x_trust="trust", y_trust="trust",
                         hidden_dims=(1000,), epochs=2, port=5081)
    result = run_scenario(cfg)
    assert result["train_acc"] > 0.5, f"Low train acc: {result['train_acc']}"
    assert 0.0 <= result["trust_mass"] <= 1.0, f"Trust mass OOB: {result['trust_mass']}"


@pytest.mark.integration
def test_mnist_500_500():
    """MNIST: 784-500-500-10 MLP, trust/trust."""
    cfg = make_mnist_cfg(x_trust="trust", y_trust="trust",
                         hidden_dims=(500, 500), epochs=2, port=5082)
    result = run_scenario(cfg)
    assert result["train_acc"] > 0.5, f"Low train acc: {result['train_acc']}"
    assert 0.0 <= result["trust_mass"] <= 1.0, f"Trust mass OOB: {result['trust_mass']}"


@pytest.mark.integration
def test_mnist_resnet_lite():
    """MNIST: ResNet-lite conv model with the conv-PTAS prototype."""
    result = run_resnet_scenario(epochs=2, port=5083)
    assert result["train_acc"] > 0.5, f"Low train acc: {result['train_acc']}"
    assert 0.0 <= result["trust_mass"] <= 1.0, f"Trust mass OOB: {result['trust_mass']}"

# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MNIST PTAS — large-architecture + ResNet-lite sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    p.add_argument("--epsilon-low", type=float, default=_DEFAULT_EPS)
    p.add_argument("--port", type=int, default=_BASE_PORT)
    p.add_argument("--skip-resnet", action="store_true",
                   help="Only run the 1000 and 500-500 MLP scenarios")
    p.add_argument("--resnet-only", action="store_true",
                   help="Only run the ResNet-lite conv scenario")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results = run_all_scenarios(
        epochs=args.epochs,
        epsilon_low=args.epsilon_low,
        base_port=args.port,
        skip_resnet=args.skip_resnet,
        resnet_only=args.resnet_only,
    )
    print_results_table(results)
    print("\n=== MNIST (more architectures) test complete ===\n")


if __name__ == "__main__":
    main()
