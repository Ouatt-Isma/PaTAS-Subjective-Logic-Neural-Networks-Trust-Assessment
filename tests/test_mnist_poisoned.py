"""
MNIST poisoned dataset PTAS integration test.

Run standalone:
    python tests/test_mnist_poisoned.py                  # full sweep (4 patch sizes) + IPTA table
    python tests/test_mnist_poisoned.py --epochs 5       # quick smoke-test
    python tests/test_mnist_poisoned.py --patch-size 4   # single patch size only

Table 1 — Poisoned sweep (patch sizes 1, 4, 10, 27):
    Patch Size | Trust(3) | Trust(6) | Train(%) | Test(%) |
               | Clean 3(%) | Clean 6(%) | 3 w/patch(%) | 6 w/patch(%)

Table 2 — IPTA results for NN trained on 4×4-poisoned dataset:
    Sample type                  | Accuracy(%) | Trust | Distrust | Uncertainty
    Clean 3                      |
    Clean 6                      |
    6 with patch (all trusted)   |
    6 with patch (patch distrust)|

Trust for class c = feedforward output trust at neuron c when input is fully trusted.
IPTA = Individual Path Trust Assessment = GenIPTA(activated_neurons)(Tx).
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
for _p in (_v2_dir, _patas_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import patas_module  # triggers patas_module/__init__.py path bootstrap
from main import (
    TestCaseConfig,
    get_lr_mnist,
    start_ptas,
)

try:
    import pytest
except ImportError:
    class _FakeMark:  # type: ignore[no-redef]
        @staticmethod
        def integration(fn):
            return fn
    class _FakePytest:  # type: ignore[no-redef]
        mark = _FakeMark()
    pytest = _FakePytest()  # type: ignore[assignment]

# ── Constants ─────────────────────────────────────────────────────────────────

PATCH_SIZES:     list[int] = [1, 4, 10, 27]
IPTA_PATCH_SIZE: int       = 4        # patch size used for Table 2
_BASE_PORT:      int       = 5060     # ports 5060–5063 (one per patch size)
_DEFAULT_EPOCHS: int       = 20
_HIDDEN_DIM:     int       = 128      # default hidden layer width for poisoned test

# ─────────────────────────────────────────────────────────────────────────────
# Config factory
# ─────────────────────────────────────────────────────────────────────────────

def make_poisoned_cfg(
    patch_size: int = 4,
    hidden_dim: int = _HIDDEN_DIM,
    port:       int = _BASE_PORT,
    epochs:     int = _DEFAULT_EPOCHS,
) -> TestCaseConfig:
    """
    Config for MNIST poisoned training.
    ``mnist_poisoned_soph=True`` activates the poisoned-aware trust generator
    (build_mnist_poisoned_soph_generator) which trusts all features but distrusts
    output neurons 6 and 9.
    ``mnist_patch_size`` sets the trigger patch size (top-left corner).
    """
    return TestCaseConfig(
        dataset="mnist",
        input_dim=28 * 28,
        output_dim=10,
        hidden_dim=hidden_dim,
        hidden_dims=(hidden_dim,),
        epochs=epochs,
        batch_size=128,
        learning_rate=get_lr_mnist,
        epsilon_low=0.05,
        x_trust="trust",        # dataset loaded clean; poisoning via poisoned_patch
        y_trust="trust",
        port=port,
        mnist_patch_size=patch_size,
        mnist_poisoned_soph=True,
        no_round=None,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Subprocess worker — PTAS side
# ─────────────────────────────────────────────────────────────────────────────

def _ptas_worker_poisoned(
    cfg: TestCaseConfig,
    result_queue: "multiprocessing.Queue[dict]",
    ready_event=None,
) -> None:
    """
    Run PTAS training for a poisoned-MNIST config.
    Returns:
        trust_mass     : overall aggregated output trust (fully-trusted Tx)
        trust_for_3    : trust of output neuron 3 (fully-trusted Tx)
        trust_for_6    : trust of output neuron 6 (fully-trusted Tx)
        omega_arrays   : trained omega_thetas as plain numpy arrays (picklable)
    ``ready_event`` is set by run_chunk() once the socket is bound/listening.
    """
    try:
        ptas = start_ptas(cfg, ready_event=ready_event)

        from concrete.ArrayTO import ArrayTO
        from concrete.TrustOpinion import TrustOpinion
        from NN.PTAStemplate import PTAS as PTASClass

        trusted_input = ArrayTO(TrustOpinion.fill((1, cfg.input_dim), method="trust"))
        a_trust = ptas.apply_feedforward(trusted_input)
        # a_trust.value shape: (1, output_dim, 3) → index [batch, class, t/d/u]
        trust_for_3  = float(a_trust.value[0, 3, 0])
        trust_for_6  = float(a_trust.value[0, 6, 0])
        trust_mass   = float(PTASClass.aggregation(a_trust)[0])

        # Save omega_thetas as numpy arrays so they can pass through the Queue
        omega_arrays = [ot.value.copy() for ot in ptas.omega_thetas]

        result_queue.put({
            "trust_mass":    trust_mass,
            "trust_for_3":   trust_for_3,
            "trust_for_6":   trust_for_6,
            "omega_arrays":  omega_arrays,
        })
    except Exception as exc:
        import traceback
        result_queue.put({
            "trust_mass":   float("nan"),
            "trust_for_3":  float("nan"),
            "trust_for_6":  float("nan"),
            "omega_arrays": None,
            "error": str(exc),
            "tb": traceback.format_exc(),
        })

# ─────────────────────────────────────────────────────────────────────────────
# Subprocess worker — NN client side
# ─────────────────────────────────────────────────────────────────────────────

def _client_worker_poisoned(
    cfg: TestCaseConfig,
    result_queue: "multiprocessing.Queue[dict]",
    capture_ipta: bool = False,
) -> None:
    """
    Custom NN client for poisoned MNIST.
    Replicates the poisoned branch of start_client() but also captures:
      - accuracy metrics (train, test, clean 3/6, poisoned 3/6)
      - inference paths for IPTA (when capture_ipta=True)

    Inference paths are computed with a LOCAL ptas=False copy of the NN so
    no extra messages are sent over the socket.  The final nn.end() call
    sends Mode.END which causes PTAS.run_chunk() to return.
    """
    try:
        from NN.primaryNN import NeuralNetwork
        from NN.datasets import load_data, add_trigger_patch

        hidden_list = list(cfg.hidden_dims) if cfg.hidden_dims else [cfg.hidden_dim]
        hidden_size2 = hidden_list[1] if len(hidden_list) > 1 else None
        arch_str = "_".join(str(h) for h in hidden_list)

        # ── Load poisoned MNIST ───────────────────────────────────────────────
        X_train, X_test, y_train, y_test, _ = load_data(
            cfg.dataset, "clean", "clean",
            poisoned_patch=cfg.mnist_patch_size,
        )

        y_test_labels = np.argmax(y_test, axis=1)
        ids_6 = np.where(y_test_labels == 6)[0]
        ids_3 = np.where(y_test_labels == 3)[0]

        pois_X_test_6 = np.stack([
            add_trigger_patch(X_test[i], patch_size=cfg.mnist_patch_size) for i in ids_6
        ])
        pois_X_test_3 = np.stack([
            add_trigger_patch(X_test[i], patch_size=cfg.mnist_patch_size) for i in ids_3
        ])

        # ── Create NN and train ───────────────────────────────────────────────
        datapath = (
            f"results/NN_Train_{cfg.dataset}_poisoned"
            f"_p{cfg.mnist_patch_size}_{arch_str}_PathSize_{cfg.mnist_patch_size}"
        )
        os.makedirs(datapath, exist_ok=True)

        nn = NeuralNetwork(
            cfg.input_dim, hidden_list[0], cfg.output_dim,
            hidden_size2=hidden_size2,
            ptas=True, operation=True, port=cfg.port,
        )

        nn.train(
            X_train, y_train, X_test, y_test,
            epochs=cfg.epochs, batch_size=cfg.batch_size,
            lr_scheduler=cfg.learning_rate, plot=True, fname=datapath,
            X_non_pois_3=X_test[ids_3], X_non_pois_6=X_test[ids_6],
            X_pois_3=pois_X_test_3, X_pois_6=pois_X_test_6,
        )

        # ── Accuracy metrics ─────────────────────────────────────────────────
        train_acc   = float(np.mean(nn.predict(X_train) == np.argmax(y_train, axis=1)))
        test_acc    = float(np.mean(nn.predict(X_test)  == np.argmax(y_test,  axis=1)))
        acc_clean_3 = float(np.mean(nn.predict(X_test[ids_3]) == 3))
        acc_clean_6 = float(np.mean(nn.predict(X_test[ids_6]) == 6))
        acc_pois_3  = float(np.mean(nn.predict(pois_X_test_3) == 3))
        acc_pois_6  = float(np.mean(nn.predict(pois_X_test_6) == 6))

        result: dict[str, Any] = {
            "train_acc":   train_acc,
            "test_acc":    test_acc,
            "acc_clean_3": acc_clean_3,
            "acc_clean_6": acc_clean_6,
            "acc_pois_3":  acc_pois_3,
            "acc_pois_6":  acc_pois_6,
        }

        # ── Optional: inference paths for IPTA (no PTAS socket) ──────────────
        if capture_ipta:
            # Mirror the trained weights into a local ptas=False copy
            nn_local = NeuralNetwork(
                cfg.input_dim, hidden_list[0], cfg.output_dim,
                hidden_size2=hidden_size2, ptas=False,
            )
            nn_local.W1 = nn.W1.copy();  nn_local.b1 = nn.b1.copy()
            nn_local.W2 = nn.W2.copy();  nn_local.b2 = nn.b2.copy()
            if hidden_size2 is not None:
                nn_local.W3 = nn.W3.copy(); nn_local.b3 = nn.b3.copy()

            # Representative samples (first example of each category)
            s_c3 = X_test[ids_3[0]][np.newaxis]
            s_c6 = X_test[ids_6[0]][np.newaxis]
            s_p6 = pois_X_test_6[0][np.newaxis]

            _, path_c3 = nn_local.forward(s_c3, getactivated=True)
            _, path_c6 = nn_local.forward(s_c6, getactivated=True)
            _, path_p6 = nn_local.forward(s_p6, getactivated=True)

            result["inference_paths"] = {
                "clean_3": path_c3,
                "clean_6": path_c6,
                "pois_6":  path_p6,
            }

        # ── Terminate PTAS (sends Mode.END, closes socket) ───────────────────
        nn.end()

        result_queue.put(result)

    except Exception as exc:
        import traceback
        result_queue.put({
            "error":      str(exc),
            "traceback":  traceback.format_exc(),
            "train_acc":  float("nan"), "test_acc":    float("nan"),
            "acc_clean_3": float("nan"), "acc_clean_6": float("nan"),
            "acc_pois_3":  float("nan"), "acc_pois_6":  float("nan"),
        })

# ─────────────────────────────────────────────────────────────────────────────
# Single-scenario runner
# ─────────────────────────────────────────────────────────────────────────────

def run_poisoned_scenario(
    cfg: TestCaseConfig,
    capture_ipta: bool = False,
) -> dict[str, Any]:
    """
    Launch PTAS + NN client for one poisoned-MNIST scenario.
    Uses a multiprocessing.Event for PTAS-readiness synchronisation
    (avoids the cold-start race on Windows).
    """
    ptas_q:    "multiprocessing.Queue[dict]" = multiprocessing.Queue()
    client_q:  "multiprocessing.Queue[dict]" = multiprocessing.Queue()
    ready_event = multiprocessing.Event()

    ptas_proc = multiprocessing.Process(
        target=_ptas_worker_poisoned, args=(cfg, ptas_q, ready_event)
    )
    ptas_proc.start()
    ready_event.wait(timeout=60)   # wait for socket to be bound

    client_proc = multiprocessing.Process(
        target=_client_worker_poisoned, args=(cfg, client_q, capture_ipta)
    )
    client_proc.start()

    # ── Read from queues BEFORE joining ──────────────────────────────────────
    # On Windows, a subprocess that puts a large object (e.g. omega_arrays ~1 MB)
    # into a Queue can DEADLOCK if the parent calls .join() first — the pipe
    # buffer fills up and the subprocess blocks waiting for the parent to drain
    # it, while the parent is waiting for the subprocess to exit.  Reading
    # first drains the pipe and lets both processes exit cleanly.
    # Timeout: MNIST training (60K samples, 128-dim, 20 epochs) can take hours;
    # use 7200 s (2 hours) as a safe upper bound.
    _QUEUE_TIMEOUT = 7200
    try:
        ptas_res = ptas_q.get(timeout=_QUEUE_TIMEOUT)
    except Exception as e:
        print(f"  [DEBUG] ptas_q.get failed: {e}")
        ptas_res = {}
    try:
        client_res = client_q.get(timeout=_QUEUE_TIMEOUT)
    except Exception as e:
        print(f"  [DEBUG] client_q.get failed: {e}")
        client_res = {}

    client_proc.join(timeout=60)
    ptas_proc.join(timeout=60)

    if "error" in ptas_res:
        print(f"  [PTAS ERROR] {ptas_res['error']}")
        print(ptas_res.get("tb", ""))
    if "error" in client_res:
        print(f"  [CLIENT ERROR] {client_res['error']}")
        print(client_res.get("traceback", ""))

    return {
        "patch_size":      cfg.mnist_patch_size,
        "trust_mass":      ptas_res.get("trust_mass",    float("nan")),
        "trust_for_3":     ptas_res.get("trust_for_3",   float("nan")),
        "trust_for_6":     ptas_res.get("trust_for_6",   float("nan")),
        "omega_arrays":    ptas_res.get("omega_arrays",  None),
        "train_acc":       client_res.get("train_acc",   float("nan")),
        "test_acc":        client_res.get("test_acc",    float("nan")),
        "acc_clean_3":     client_res.get("acc_clean_3", float("nan")),
        "acc_clean_6":     client_res.get("acc_clean_6", float("nan")),
        "acc_pois_3":      client_res.get("acc_pois_3",  float("nan")),
        "acc_pois_6":      client_res.get("acc_pois_6",  float("nan")),
        "inference_paths": client_res.get("inference_paths", None),
    }

# ─────────────────────────────────────────────────────────────────────────────
# IPTA computation (runs in the parent process after training)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ipta_results(
    omega_arrays:    list[np.ndarray],
    inference_paths: dict[str, Any],
    patch_size:      int,
    structure:       list[int],
    input_dim:       int = 784,
) -> dict[str, tuple[float, float, float]]:
    """
    Reconstruct a PTAS from saved omega_thetas (numpy arrays) and compute
    IPTA outputs for four evaluation rows:

        clean_3        : Clean class-3 sample, fully-trusted Tx
        clean_6        : Clean class-6 sample, fully-trusted Tx
        pois_6_trusted : Poisoned class-6 sample, fully-trusted Tx
        pois_6_distrust: Poisoned class-6 sample, patch pixels distrusted

    Returns a dict  {row_key: (trust, distrust, uncertainty)}.
    """
    from concrete.TensorTO import TensorArrayTO
    from concrete.ArrayTO import ArrayTO
    from concrete.TrustOpinion import TrustOpinion
    from NN.PTAStemplate import PTAS as PTASClass

    # ── Reconstruct PTAS offline (no socket needed) ───────────────────────────
    omega_tensors = [TensorArrayTO(arr) for arr in omega_arrays]
    ptas_recon = PTASClass(
        omega_thetas=omega_tensors,
        operator_mapping=None,
        nn_interface=None,
        trust_assessment_func=None,
        structure=structure,
        epsilon_low=0.05,
        epsilon_up=None,
        eval=False,
        use_tensor=True,
    )

    # ── Build Tx opinions ─────────────────────────────────────────────────────
    # All pixels fully trusted
    Tx_trusted = ArrayTO(TrustOpinion.fill((1, input_dim), method="trust"))

    # All pixels trusted EXCEPT the trigger patch (top-left patch_size × patch_size)
    Tx_patch = ArrayTO(TrustOpinion.fill((1, input_dim), method="trust"))
    for i in range(patch_size):
        for j in range(patch_size):
            Tx_patch.value[0][28 * i + j] = TrustOpinion.dtrust()

    # ── Compute GenIPTA for each inference path ───────────────────────────────
    rows: dict[str, tuple[float, float, float]] = {}

    def _ipta_agg(path, tx):
        ipta_fn = ptas_recon.GenIPTA(path)
        agg = ipta_fn(tx)   # shape (3,): [trust, distrust, uncertainty]
        return float(agg[0]), float(agg[1]), float(agg[2])

    rows["clean_3"]         = _ipta_agg(inference_paths["clean_3"], Tx_trusted)
    rows["clean_6"]         = _ipta_agg(inference_paths["clean_6"], Tx_trusted)
    rows["pois_6_trusted"]  = _ipta_agg(inference_paths["pois_6"],  Tx_trusted)
    rows["pois_6_distrust"] = _ipta_agg(inference_paths["pois_6"],  Tx_patch)

    return rows

# ─────────────────────────────────────────────────────────────────────────────
# Full sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_poisoned_sweep(
    patch_sizes: list[int]  = PATCH_SIZES,
    epochs:      int        = _DEFAULT_EPOCHS,
    hidden_dim:  int        = _HIDDEN_DIM,
    base_port:   int        = _BASE_PORT,
    ipta_patch:  int        = IPTA_PATCH_SIZE,
) -> tuple[list[dict[str, Any]], dict[str, tuple[float, float, float]] | None]:
    """
    Run all patch-size scenarios sequentially.
    Returns:
        sweep_results : list of per-scenario result dicts (Table 1 data)
        ipta_rows     : IPTA row data for the ipta_patch scenario (Table 2 data)
                        or None if computation failed.
    """
    sweep_results: list[dict[str, Any]] = []
    ipta_rows = None

    for i, ps in enumerate(patch_sizes):
        capture = (ps == ipta_patch)
        cfg = make_poisoned_cfg(
            patch_size=ps,
            hidden_dim=hidden_dim,
            port=base_port + i,
            epochs=epochs,
        )
        print(f"\n{'='*64}")
        print(f"  Poisoned MNIST  |  patch={ps}×{ps}  |  port={cfg.port}")
        print(f"{'='*64}")

        result = run_poisoned_scenario(cfg, capture_ipta=capture)

        print(f"  trust(3)={result['trust_for_3']:.4f}  "
              f"trust(6)={result['trust_for_6']:.4f}  "
              f"train={result['train_acc']*100:.2f}%  "
              f"test={result['test_acc']*100:.2f}%")
        print(f"  clean3={result['acc_clean_3']*100:.2f}%  "
              f"clean6={result['acc_clean_6']*100:.2f}%  "
              f"pois3={result['acc_pois_3']*100:.2f}%  "
              f"pois6={result['acc_pois_6']*100:.2f}%")

        sweep_results.append(result)

        # Compute IPTA for the target patch size
        if capture and result["omega_arrays"] is not None and result["inference_paths"] is not None:
            structure = [cfg.input_dim, hidden_dim, cfg.output_dim]
            try:
                ipta_rows = compute_ipta_results(
                    omega_arrays    = result["omega_arrays"],
                    inference_paths = result["inference_paths"],
                    patch_size      = ps,
                    structure       = structure,
                    input_dim       = cfg.input_dim,
                )
            except Exception as exc:
                print(f"  [IPTA] computation failed: {exc}")
                ipta_rows = None

        time.sleep(2)

    return sweep_results, ipta_rows

# ─────────────────────────────────────────────────────────────────────────────
# Table printers
# ─────────────────────────────────────────────────────────────────────────────

def print_table1(sweep_results: list[dict[str, Any]]) -> None:
    """
    Table 1 — Poisoned training sweep.

    Patch | Trust(3) | Trust(6) | Train(%) | Test(%) | Clean3(%) | Clean6(%) | 3+patch(%) | 6+patch(%)
    """
    def _pct(v: float) -> str:
        return f"{v*100:6.2f}%" if v == v else "  N/A  "

    def _f4(v: float) -> str:
        return f"{v:.4f}" if v == v else " N/A  "

    headers = ["Patch", "Trust(3)", "Trust(6)", "Train(%)", "Test(%)",
               "Clean 3(%)", "Clean 6(%)", "3+patch(%)", "6+patch(%)"]
    col_w   = [7, 10, 10, 10, 10, 12, 12, 12, 12]
    parts   = [f"{{:<{col_w[0]}}}"] + [f"{{:>{w}}}" for w in col_w[1:]]
    row_fmt = "  ".join(parts)
    sep     = "-" * (sum(col_w) + 2 * (len(col_w) - 1))

    print()
    print("=" * len(sep))
    print("  MNIST POISONED — Patch Size Sweep (Table 1)")
    print("  Trust(3/6) = feedforward output trust at neuron 3/6 for fully-trusted input")
    print("=" * len(sep))
    print()
    print(sep)
    print(row_fmt.format(*headers))
    print(sep)

    for r in sweep_results:
        ps = r["patch_size"]
        row = [
            f"{ps}×{ps}",
            _f4(r["trust_for_3"]),
            _f4(r["trust_for_6"]),
            _pct(r["train_acc"]),
            _pct(r["test_acc"]),
            _pct(r["acc_clean_3"]),
            _pct(r["acc_clean_6"]),
            _pct(r["acc_pois_3"]),
            _pct(r["acc_pois_6"]),
        ]
        print(row_fmt.format(*row))

    print(sep)
    print()


def print_table2(
    sweep_results: list[dict[str, Any]],
    ipta_rows:     dict[str, tuple[float, float, float]] | None,
    ipta_patch:    int = IPTA_PATCH_SIZE,
) -> None:
    """
    Table 2 — IPTA results for the ipta_patch × ipta_patch scenario.

    Sample type                 | Accuracy(%) | Trust  | Distrust | Uncertainty
    Clean 3                     |
    Clean 6                     |
    6 with patch (trusted Tx)   |
    6 with patch (patch distrust|
    """
    def _pct(v: float) -> str:
        return f"{v*100:6.2f}%" if v == v else "  N/A  "

    def _f4(v: float) -> str:
        return f"{v:.4f}" if v == v else " N/A  "

    # Find accuracy values for the target patch size from the sweep
    base: dict[str, float] = {
        "acc_clean_3": float("nan"), "acc_clean_6": float("nan"),
        "acc_pois_6":  float("nan"),
    }
    for r in sweep_results:
        if r["patch_size"] == ipta_patch:
            base = r
            break

    headers = ["Sample type", "Accuracy(%)", "Trust", "Distrust", "Uncertainty"]
    col_w   = [34, 13, 8, 10, 13]
    parts   = [f"{{:<{col_w[0]}}}"] + [f"{{:>{w}}}" for w in col_w[1:]]
    row_fmt = "  ".join(parts)
    sep     = "-" * (sum(col_w) + 2 * (len(col_w) - 1))

    print()
    print("=" * len(sep))
    print(f"  MNIST POISONED — IPTA Results for {ipta_patch}×{ipta_patch} patch (Table 2)")
    print("  IPTA = GenIPTA(activated_neurons)(Tx)")
    print("  Rows 1–3: fully-trusted Tx   |  Row 4: patch pixels distrusted")
    print("=" * len(sep))
    print()

    if ipta_rows is None:
        print("  [IPTA data unavailable — PTAS or client worker failed]")
        print()
        return

    print(sep)
    print(row_fmt.format(*headers))
    print(sep)

    row_defs = [
        ("Clean 3",                        _pct(base.get("acc_clean_3", float("nan"))), "clean_3"),
        ("Clean 6",                        _pct(base.get("acc_clean_6", float("nan"))), "clean_6"),
        (f"6 with {ipta_patch}×{ipta_patch} patch (all trusted Tx)",
                                           _pct(base.get("acc_pois_6", float("nan"))), "pois_6_trusted"),
        (f"6 with {ipta_patch}×{ipta_patch} patch (patch distrusted)",
                                           _pct(base.get("acc_pois_6", float("nan"))), "pois_6_distrust"),
    ]

    for label, acc_str, key in row_defs:
        t, d, u = ipta_rows.get(key, (float("nan"), float("nan"), float("nan")))
        print(row_fmt.format(label, acc_str, _f4(t), _f4(d), _f4(u)))

    print(sep)
    print()

# ─────────────────────────────────────────────────────────────────────────────
# pytest integration tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_poisoned_patch1():
    """MNIST: poisoned training with 1×1 patch."""
    cfg = make_poisoned_cfg(patch_size=1, epochs=2, port=5060)
    result = run_poisoned_scenario(cfg)
    assert result["train_acc"] > 0.5, f"Low train acc: {result['train_acc']}"


@pytest.mark.integration
def test_poisoned_patch4():
    """MNIST: poisoned training with 4×4 patch + IPTA computation."""
    cfg = make_poisoned_cfg(patch_size=4, epochs=2, port=5061)
    result = run_poisoned_scenario(cfg, capture_ipta=True)
    assert result["train_acc"] > 0.5, f"Low train acc: {result['train_acc']}"
    if result["omega_arrays"] is not None and result["inference_paths"] is not None:
        structure = [cfg.input_dim, _HIDDEN_DIM, cfg.output_dim]
        ipta = compute_ipta_results(
            result["omega_arrays"], result["inference_paths"], 4, structure
        )
        for key in ("clean_3", "clean_6", "pois_6_trusted", "pois_6_distrust"):
            t, d, u = ipta[key]
            assert abs(t + d + u - 1.0) < 0.01 or (t + d + u) <= 1.0 + 0.01, \
                f"IPTA opinion doesn't sum to ≤1: {key} = ({t},{d},{u})"


@pytest.mark.integration
def test_poisoned_patch10():
    """MNIST: poisoned training with 10×10 patch."""
    cfg = make_poisoned_cfg(patch_size=10, epochs=2, port=5062)
    result = run_poisoned_scenario(cfg)
    assert result["train_acc"] > 0.5, f"Low train acc: {result['train_acc']}"


@pytest.mark.integration
def test_poisoned_patch27():
    """MNIST: poisoned training with 27×27 patch."""
    cfg = make_poisoned_cfg(patch_size=27, epochs=2, port=5063)
    result = run_poisoned_scenario(cfg)
    assert result["train_acc"] > 0.5, f"Low train acc: {result['train_acc']}"

# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "MNIST poisoned PTAS test.\n"
            "Default (no --patch-size): runs all 4 patch sizes and prints both tables."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--patch-size", type=int, default=None,
                   help="Run a single patch size instead of the full sweep")
    p.add_argument("--epochs",     type=int, default=_DEFAULT_EPOCHS)
    p.add_argument("--hidden-dim", type=int, default=_HIDDEN_DIM)
    p.add_argument("--port",       type=int, default=_BASE_PORT)
    p.add_argument("--ipta-patch", type=int, default=IPTA_PATCH_SIZE,
                   help=f"Patch size for IPTA table (default: {IPTA_PATCH_SIZE})")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.patch_size is not None:
        # ── Single patch size run ─────────────────────────────────────────────
        cfg = make_poisoned_cfg(
            patch_size=args.patch_size,
            hidden_dim=args.hidden_dim,
            port=args.port,
            epochs=args.epochs,
        )
        capture = (args.patch_size == args.ipta_patch)
        result = run_poisoned_scenario(cfg, capture_ipta=capture)
        print_table1([result])

        if capture and result["omega_arrays"] is not None and result["inference_paths"] is not None:
            structure = [cfg.input_dim, args.hidden_dim, cfg.output_dim]
            ipta_rows = compute_ipta_results(
                result["omega_arrays"], result["inference_paths"],
                args.patch_size, structure, cfg.input_dim,
            )
            print_table2([result], ipta_rows, ipta_patch=args.patch_size)
        return

    # ── Full sweep ────────────────────────────────────────────────────────────
    print(f"\nRunning full poisoned-MNIST sweep: patches={PATCH_SIZES}, "
          f"epochs={args.epochs}, hidden={args.hidden_dim}\n")

    sweep_results, ipta_rows = run_poisoned_sweep(
        patch_sizes=PATCH_SIZES,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        base_port=args.port,
        ipta_patch=args.ipta_patch,
    )

    print_table1(sweep_results)
    print_table2(sweep_results, ipta_rows, ipta_patch=args.ipta_patch)

    print("=== Poisoned MNIST test complete ===\n")


if __name__ == "__main__":
    main()
