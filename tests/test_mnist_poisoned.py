"""
MNIST poisoned dataset PTAS integration test.

Replicates the poisoned-MNIST setup from patas_module/main.py:
  server  →  start_ptas(cfg)
  client  →  start_client(cfg, not_ptas=False)

Run standalone:
    python tests/test_mnist_poisoned.py                  # full sweep (4 patch sizes) + IPTA table
    python tests/test_mnist_poisoned.py --epochs 5       # quick smoke-test
    python tests/test_mnist_poisoned.py --patch-size 4   # single patch size only

Table 1 — Poisoned sweep (patch sizes 1, 4, 10, 27):
    Patch | Trust(3) | Trust(6) | Train(%) | Test(%) |
          | Clean 3(%) | Clean 6(%) | 3 w/patch(%) | 6 w/patch(%)

Table 2 — IPTA results for NN trained on 4×4-poisoned dataset:
    Sample type                   | Accuracy(%) | Trust | Distrust | Uncertainty
    Clean 3                       |
    Clean 6                       |
    6 with patch (all trusted Tx) |
    6 with patch (patch distrust) |
    3 with patch (all trusted Tx) |
    3 with patch (patch distrust) |

Trust for class c = feedforward output trust at neuron c when input is fully trusted.
IPTA = GenIPTA(activated_neurons)(Tx)   — computed offline after training.
"""


from __future__ import annotations

import os
import sys
import json
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
from main import TestCaseConfig, get_lr_mnist, start_ptas, start_client

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
PATCH_SIZES:     list[int] = [1, 4, 10, 27]
IPTA_PATCH_SIZE: int       = 4        # patch size used for Table 2
_BASE_PORT:      int       = 5061     # ports 5060–5063 (one per patch size)
_DEFAULT_EPOCHS: int       = 20
_HIDDEN_DIM:     int       = 128

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _datapath(cfg: TestCaseConfig) -> str:
    """Return the result directory that start_client() writes to."""
    hidden_list = list(cfg.hidden_dims) if cfg.hidden_dims else [cfg.hidden_dim]
    arch_str = "_".join(str(h) for h in hidden_list)
    return (
        f"results/NN_Train_{cfg.dataset}_{arch_str}_{cfg.x_trust}_{cfg.y_trust}"
        f"_PathSize_{cfg.mnist_patch_size if cfg.mnist_poisoned_soph else 'None'}"
    )


def _ptas_datapath(cfg: TestCaseConfig) -> str:
    """Return the result directory that start_ptas() writes to."""
    hidden_list = list(cfg.hidden_dims) if cfg.hidden_dims else [cfg.hidden_dim]
    arch_str = "_".join(str(h) for h in hidden_list)
    return (
        f"results/PTAS_Eval_{cfg.dataset}_{arch_str}_{cfg.x_trust}_{cfg.y_trust}"
        f"_eps_{cfg.epsilon_low}"
        f"_PathSize_{cfg.mnist_patch_size if cfg.mnist_poisoned_soph else 'None'}"
    )


def _load_result_from_disk(
    cfg: TestCaseConfig,
    nn_dp: str,
    ptas_dp: str,
) -> dict[str, Any]:
    """
    Reconstruct a scenario result from files already on disk.
    Works with whatever subset of files exists:
      • at.pkl          → trust_for_3 / trust_for_6
      • omega_arrays.pkl → omega_arrays (needed for IPTA)
      • metrics.txt     → accuracy numbers
    Missing files are silently replaced with NaN / None.
    """
    import pickle

    trust_for_3 = trust_for_6 = trust_mass = float("nan")
    omega_arrays = None

    at_path = os.path.join(ptas_dp, "at.pkl")
    if os.path.exists(at_path):
        with open(at_path, "rb") as _f:
            at = pickle.load(_f)
        from NN.PTAStemplate import PTAS as PTASClass
        trust_for_3 = float(at.value[0, 3, 0])
        trust_for_6 = float(at.value[0, 6, 0])
        trust_mass  = float(PTASClass.aggregation(at)[0])

    omega_path = os.path.join(ptas_dp, "omega_arrays.pkl")
    if os.path.exists(omega_path):
        with open(omega_path, "rb") as _f:
            omega_arrays = pickle.load(_f)

    metrics = _read_metrics(os.path.join(nn_dp, "metrics.txt"))

    return {
        "patch_size":   cfg.mnist_patch_size,
        "trust_mass":   trust_mass,
        "trust_for_3":  trust_for_3,
        "trust_for_6":  trust_for_6,
        "omega_arrays": omega_arrays,
        "train_acc":    metrics.get("Train",             float("nan")),
        "test_acc":     metrics.get("Test",              float("nan")),
        "acc_clean_3":  metrics.get("Clean Images 3",    float("nan")),
        "acc_clean_6":  metrics.get("Clean Images 6",    float("nan")),
        "acc_pois_3":   metrics.get("Poisoned Images 3", float("nan")),
        "acc_pois_6":   metrics.get("Poisoned Images 6", float("nan")),
        "datapath":     nn_dp,
    }


def _read_metrics(path: str) -> dict[str, float]:
    """
    Read the metrics.txt written by nn.train(plot=True, fname=...).
    Format: one "Key: value" per line.
    Keys used: Train, Test, Poisoned Images 6, Clean Images 6, Clean Images 3,
               Poisoned Images 3.
    """
    result: dict[str, float] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if ": " in line:
                    k, v = line.split(": ", 1)
                    try:
                        result[k.strip()] = float(v.strip())
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return result


def _read_ipta_paths(path: str) -> dict[str, Any] | None:
    """
    Read the ipta_paths.json written by start_client() for the poisoned case.
    Returns dict with keys 'clean_3', 'clean_6', 'pois_6', 'pois_3' → activation lists
    and 'probs_clean_3', 'probs_clean_6', 'probs_pois_6', 'probs_pois_3' → softmax probs.
    """
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Config factory
# ─────────────────────────────────────────────────────────────────────────────

def make_poisoned_cfg(
    patch_size: int = 4,
    hidden_dim: int = _HIDDEN_DIM,
    port:       int = _BASE_PORT,
    epochs:     int = _DEFAULT_EPOCHS,
) -> TestCaseConfig:
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
        x_trust="trust",
        y_trust="trust",
        port=port,
        mnist_patch_size=patch_size,
        mnist_poisoned_soph=True,
        no_round=None,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Subprocess workers — mirror the main.py two-process setup exactly
# ─────────────────────────────────────────────────────────────────────────────

def _ptas_worker(
    cfg: TestCaseConfig,
    result_queue: "multiprocessing.Queue[dict]",
    ready_event=None,
    force_retrain: bool = False,
) -> None:
    """
    Run start_ptas() (mirrors the 'server' mode in main.py).
    Results are put in the queue immediately after training completes,
    BEFORE the slow ptas_evaluation / eval_plot phase, so the main
    process never waits for plotting when reading ptas_q.
    """
    from concrete.ArrayTO import ArrayTO
    from concrete.TrustOpinion import TrustOpinion
    from NN.PTAStemplate import PTAS as PTASClass

    _result_sent = [False]

    def _post_training(ptas):
        """Called by start_ptas() right after run_chunk(), before eval_plot."""
        try:
            trusted_input = ArrayTO(TrustOpinion.fill((1, cfg.input_dim), method="trust"))
            a_trust = ptas.apply_feedforward(trusted_input)
            # a_trust.value shape: (1, output_dim, 3)  →  [batch, class, t/d/u]
            result_queue.put({
                "trust_mass":   float(PTASClass.aggregation(a_trust)[0]),
                "trust_for_3":  float(a_trust.value[0, 3, 0]),
                "trust_for_6":  float(a_trust.value[0, 6, 0]),
                "omega_arrays": [ot.value.copy() for ot in ptas.omega_thetas],
            })
        except Exception as exc:
            import traceback
            result_queue.put({
                "trust_mass":   float("nan"),
                "trust_for_3":  float("nan"),
                "trust_for_6":  float("nan"),
                "omega_arrays": None,
                "error": str(exc),
                "tb":    traceback.format_exc(),
            })
        finally:
            _result_sent[0] = True

    try:
        start_ptas(cfg, ready_event=ready_event, post_training_callback=_post_training, force_retrain=force_retrain)
    except Exception as exc:
        if not _result_sent[0]:
            import traceback
            result_queue.put({
                "trust_mass":   float("nan"),
                "trust_for_3":  float("nan"),
                "trust_for_6":  float("nan"),
                "omega_arrays": None,
                "error": str(exc),
                "tb":    traceback.format_exc(),
            })


def _client_worker(
    cfg: TestCaseConfig,
    result_queue: "multiprocessing.Queue[dict]",
    force_retrain: bool = False,
    not_ptas: bool = False,
) -> None:
    """
    Run start_client() (mirrors the 'client' mode in main.py).
    Reads results from the files that start_client() / nn.train() writes:
      • metrics.txt  — train/test/clean/poisoned accuracies
      • ipta_paths.json — activation lists for IPTA (written by start_client())
    """
    try:
        start_client(cfg, not_ptas=not_ptas, force_retrain=force_retrain)
        dp = _datapath(cfg)
        metrics = _read_metrics(os.path.join(dp, "metrics.txt"))
        result_queue.put({
            "train_acc":   metrics.get("Train",              float("nan")),
            "test_acc":    metrics.get("Test",               float("nan")),
            "acc_clean_3": metrics.get("Clean Images 3",     float("nan")),
            "acc_clean_6": metrics.get("Clean Images 6",     float("nan")),
            "acc_pois_3":  metrics.get("Poisoned Images 3",  float("nan")),
            "acc_pois_6":  metrics.get("Poisoned Images 6",  float("nan")),
        })
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
    force_retrain: bool = False,
    not_ptas: bool = False,
) -> dict[str, Any]:
    """
    Launch PTAS + NN client for one poisoned-MNIST scenario.
    Replicates the two-process setup in main.py (server + client).
    Queue reads happen BEFORE joins to avoid Windows pipe-buffer deadlock
    (a subprocess blocking on put() when the pipe buffer fills with large data).

    If saved model artefacts already exist on disk (and force_retrain is False)
    the entire two-process setup is bypassed and results are loaded directly.

    If not_ptas is True, only the NN client is run (no PTAS server).
    Trust / omega values will be NaN / None in the returned dict.
    """
    nn_dp   = _datapath(cfg)
    ptas_dp = _ptas_datapath(cfg)

    # ── Cache check ───────────────────────────────────────────────────────────
    # Use saved artefacts when both the PTAS weights and the NN evaluation output
    # are present.  at.pkl + metrics.txt are sufficient for Table 1;
    # omega_arrays.pkl is additionally needed for Table 2 (IPTA).
    at_path      = os.path.join(ptas_dp, "at.pkl")
    metrics_path = os.path.join(nn_dp,   "metrics.txt")
    cache_hit = (
        not force_retrain and os.path.exists(metrics_path)
        and (not_ptas or os.path.exists(at_path))
    )
    if cache_hit:
        print("  [CACHED] Saved artefacts found — loading from disk, skipping training.")
        if cfg.mnist_poisoned_soph:
            ipta_json_path = os.path.join(nn_dp, "ipta_paths.json")
            try:
                with open(ipta_json_path, encoding="utf-8") as _f:
                    _existing = json.load(_f)
                _needs_refresh = "probs_clean_3" not in _existing
            except (FileNotFoundError, json.JSONDecodeError):
                _needs_refresh = True
            if _needs_refresh:
                print("  [IPTA] ipta_paths.json outdated — refreshing probabilities ...")
                start_client(cfg, not_ptas=True, force_retrain=False)
        return _load_result_from_disk(cfg, nn_dp, ptas_dp)

    client_q: "multiprocessing.Queue[dict]" = multiprocessing.Queue()
    _QUEUE_TIMEOUT = 7200

    if not_ptas:
        # NN-only mode: skip the PTAS server entirely
        ptas_res = {}
        client_proc = multiprocessing.Process(
            target=_client_worker, args=(cfg, client_q, force_retrain, True)
        )
        client_proc.start()
        try:
            client_res = client_q.get(timeout=_QUEUE_TIMEOUT)
        except Exception as e:
            print(f"  [CLIENT queue timeout/error] {e}")
            client_res = {}
        client_proc.join(timeout=60)
        if client_proc.is_alive():
            client_proc.terminate()
            client_proc.join(timeout=10)
    else:
        ptas_q:   "multiprocessing.Queue[dict]" = multiprocessing.Queue()
        ready_event = multiprocessing.Event()

        ptas_proc = multiprocessing.Process(
            target=_ptas_worker, args=(cfg, ptas_q, ready_event, force_retrain)
        )
        ptas_proc.start()
        ready_event.wait(timeout=60)   # wait for PTAS socket to bind

        client_proc = multiprocessing.Process(
            target=_client_worker, args=(cfg, client_q, force_retrain)
        )
        client_proc.start()

        # ── Read from queues BEFORE joining ──────────────────────────────────
        # MNIST training (128-dim, 20 epochs) can take ~2 hours;
        # use 7200 s timeout. The PTAS queue payload is ~1 MB (omega_arrays) which
        # can block a subprocess on Windows until the parent drains the pipe.
        try:
            ptas_res = ptas_q.get(timeout=_QUEUE_TIMEOUT)
        except Exception as e:
            print(f"  [PTAS queue timeout/error] {e}")
            ptas_res = {}
        try:
            client_res = client_q.get(timeout=_QUEUE_TIMEOUT)
        except Exception as e:
            print(f"  [CLIENT queue timeout/error] {e}")
            client_res = {}

        client_proc.join(timeout=60)
        ptas_proc.join(timeout=60)

        # Terminate any subprocess that didn't finish within the join window.
        # Without this, Python's multiprocessing atexit blocks indefinitely on
        # non-daemon processes that are still running (e.g. stuck in eval_plot
        # or waiting for a client message after the queue timeout fires).
        for proc in (client_proc, ptas_proc):
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)

    if "error" in ptas_res:
        print(f"  [PTAS ERROR] {ptas_res['error']}")
        print(ptas_res.get("tb", ""))
    if "error" in client_res:
        print(f"  [CLIENT ERROR] {client_res['error']}")
        print(client_res.get("traceback", ""))

    return {
        "patch_size":    cfg.mnist_patch_size,
        "trust_mass":    ptas_res.get("trust_mass",   float("nan")),
        "trust_for_3":   ptas_res.get("trust_for_3",  float("nan")),
        "trust_for_6":   ptas_res.get("trust_for_6",  float("nan")),
        "omega_arrays":  ptas_res.get("omega_arrays", None),
        "train_acc":     client_res.get("train_acc",   float("nan")),
        "test_acc":      client_res.get("test_acc",    float("nan")),
        "acc_clean_3":   client_res.get("acc_clean_3", float("nan")),
        "acc_clean_6":   client_res.get("acc_clean_6", float("nan")),
        "acc_pois_3":    client_res.get("acc_pois_3",  float("nan")),
        "acc_pois_6":    client_res.get("acc_pois_6",  float("nan")),
        "datapath":      _datapath(cfg),
    }

# ─────────────────────────────────────────────────────────────────────────────
# IPTA computation (offline in the parent process after training)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ipta_results(
    omega_arrays:    list[np.ndarray],
    inference_paths: dict[str, Any],
    patch_size:      int,
    structure:       list[int],
    input_dim:       int = 784,
) -> tuple[dict[str, tuple[float, float, float]], ...]:
    """
    Reconstruct a PTAS from omega_arrays and compute IPTA outputs for all rows.

    Returns (by_class, aggregate, weighted) — three dicts keyed by row name,
    each mapping to a (trust, distrust, uncertainty) triple.
      by_class  — opinion at the true-class output neuron
      aggregate — PTASClass.aggregation() over all output neurons
      weighted  — Σ p_c · opinion_c using NN softmax probabilities
    """
    from concrete.TensorTO import TensorArrayTO
    from concrete.ArrayTO import ArrayTO
    from concrete.TrustOpinion import TrustOpinion
    from NN.PTAStemplate import PTAS as PTASClass

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

    Tx_trusted = ArrayTO(TrustOpinion.fill((1, input_dim), method="trust"))

    Tx_patch = ArrayTO(TrustOpinion.fill((1, input_dim), method="trust"))
    for i in range(patch_size):
        for j in range(patch_size):
            Tx_patch.value[0][28 * i + j] = TrustOpinion.dtrust()

    def _ipta_all(path, tx, cls_idx, probs):
        ipta_fn = ptas_recon.GenIPTA(path)
        output  = ipta_fn(tx)               # TensorArrayTO (1, n_out, 3)
        v       = output.value[0]           # (n_out, 3)
        v_cls   = v[cls_idx]
        cls_op  = (float(v_cls[0]), float(v_cls[1]), float(v_cls[2]))
        agg_v   = PTASClass.aggregation(output)   # (3,) numpy array
        agg_op  = (float(agg_v[0]), float(agg_v[1]), float(agg_v[2]))
        if probs is not None:
            p    = np.array(probs, dtype=np.float64).flatten()
            w    = (p[:, None] * v).sum(axis=0)
            w_op = (float(w[0]), float(w[1]), float(w[2]))
        else:
            w_op = (float("nan"), float("nan"), float("nan"))
        return cls_op, agg_op, w_op

    by_class:  dict[str, tuple[float, float, float]] = {}
    aggregate: dict[str, tuple[float, float, float]] = {}
    weighted:  dict[str, tuple[float, float, float]] = {}

    rows = [
        ("clean_3",         inference_paths["clean_3"], Tx_trusted, 3, inference_paths.get("probs_clean_3")),
        ("clean_6",         inference_paths["clean_6"], Tx_trusted, 6, inference_paths.get("probs_clean_6")),
        ("pois_6_trusted",  inference_paths["pois_6"],  Tx_trusted, 6, inference_paths.get("probs_pois_6")),
        ("pois_6_distrust", inference_paths["pois_6"],  Tx_patch,   6, inference_paths.get("probs_pois_6")),
    ]
    if "pois_3" in inference_paths:
        rows += [
            ("pois_3_trusted",  inference_paths["pois_3"], Tx_trusted, 3, inference_paths.get("probs_pois_3")),
            ("pois_3_distrust", inference_paths["pois_3"], Tx_patch,   3, inference_paths.get("probs_pois_3")),
        ]

    for key, path, tx, cls_idx, probs in rows:
        by_class[key], aggregate[key], weighted[key] = _ipta_all(path, tx, cls_idx, probs)

    return by_class, aggregate, weighted

# ─────────────────────────────────────────────────────────────────────────────
# Full sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_poisoned_sweep(
    patch_sizes:   list[int] = PATCH_SIZES,
    epochs:        int       = _DEFAULT_EPOCHS,
    hidden_dim:    int       = _HIDDEN_DIM,
    base_port:     int       = _BASE_PORT,
    ipta_patch:    int       = IPTA_PATCH_SIZE,
    force_retrain: bool      = False,
    not_ptas:      bool      = False,
) -> tuple[list[dict[str, Any]], tuple[dict, dict, dict] | None]:
    """Run all patch-size scenarios sequentially."""
    sweep_results: list[dict[str, Any]] = []
    ipta_rows = None

    for i, ps in enumerate(patch_sizes):
        cfg = make_poisoned_cfg(
            patch_size=ps, hidden_dim=hidden_dim,
            port=base_port + i, epochs=epochs,
        )
        print(f"\n{'='*64}")
        print(f"  Poisoned MNIST  |  patch={ps}×{ps}  |  port={cfg.port}")
        print(f"{'='*64}")

        result = run_poisoned_scenario(cfg, force_retrain=force_retrain, not_ptas=not_ptas)

        print(f"  trust(3)={result['trust_for_3']:.4f}  "
              f"trust(6)={result['trust_for_6']:.4f}  "
              f"train={result['train_acc']*100:.2f}%  "
              f"test={result['test_acc']*100:.2f}%")
        print(f"  clean3={result['acc_clean_3']*100:.2f}%  "
              f"clean6={result['acc_clean_6']*100:.2f}%  "
              f"pois3={result['acc_pois_3']*100:.2f}%  "
              f"pois6={result['acc_pois_6']*100:.2f}%")

        sweep_results.append(result)

        # Compute IPTA offline for the target patch size
        if not not_ptas and ps == ipta_patch and result["omega_arrays"] is not None:
            ipta_paths = _read_ipta_paths(
                os.path.join(result["datapath"], "ipta_paths.json")
            )
            if ipta_paths is not None:
                structure = [cfg.input_dim, hidden_dim, cfg.output_dim]
                try:
                    ipta_rows = compute_ipta_results(
                        omega_arrays    = result["omega_arrays"],
                        inference_paths = ipta_paths,
                        patch_size      = ps,
                        structure       = structure,
                        input_dim       = cfg.input_dim,
                    )
                except Exception as exc:
                    print(f"  [IPTA] computation failed: {exc}")

        time.sleep(2)

    return sweep_results, ipta_rows

# ─────────────────────────────────────────────────────────────────────────────
# Table printers
# ─────────────────────────────────────────────────────────────────────────────

def print_table1(sweep_results: list[dict[str, Any]]) -> None:
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
        print(row_fmt.format(
            f"{ps}×{ps}",
            _f4(r["trust_for_3"]), _f4(r["trust_for_6"]),
            _pct(r["train_acc"]),  _pct(r["test_acc"]),
            _pct(r["acc_clean_3"]), _pct(r["acc_clean_6"]),
            _pct(r["acc_pois_3"]),  _pct(r["acc_pois_6"]),
        ))

    print(sep)
    print()


def plot_table2a(
    by_class:   dict[str, tuple[float, float, float]],
    patch_size: int,
    save_path:  str,
) -> None:
    """Save a grouped bar chart of Table 2a (by-class opinion) as a PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as _np
    plt.rcParams.update({
    'font.size': 16,  
    })

    row_defs = [
        ("Clean 3",                                    "clean_3"),
        ("Clean 6",                                    "clean_6"),
        (f"6 w/ patch",                   "pois_6_trusted"),
        # (f"6 w/ patch (patch distrusted)",             "pois_6_distrust"),
        (f"3 w/ patch",                   "pois_3_trusted"),
        # (f"3 w/ patch (patch distrusted)",             "pois_3_distrust"),
    ]

    labels = [r[0] for r in row_defs]
    keys   = [r[1] for r in row_defs]

    trust_vals   = [by_class.get(k, (float("nan"),)*3)[0] for k in keys]
    distrust_vals= [by_class.get(k, (float("nan"),)*3)[1] for k in keys]
    uncert_vals  = [by_class.get(k, (float("nan"),)*3)[2] for k in keys]

    x      = _np.arange(len(labels))
    width  = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))

    bars_t = ax.bar(x - width, trust_vals,    width, label="Trust",       color="#4C72B0")
    bars_d = ax.bar(x,         distrust_vals, width, label="Distrust",    color="#DD8452")
    bars_u = ax.bar(x + width, uncert_vals,   width, label="Uncertainty", color="#55A868")

    for bars in (bars_t, bars_d, bars_u):
        for bar in bars:
            v = bar.get_height()
            if v == v:  # skip NaN
                ax.text(
                    bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=16, rotation=90,
                )

    ax.set_ylabel("Opinion mass")
    ax.set_title(f"IPTA by-class opinion  ({patch_size}×{patch_size} patch)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=16)
    ax.set_ylim(0, 1.2)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] Table 2a histogram saved → {save_path}")


def plot_trust_2a_vs_2c(
    by_class:  dict[str, tuple[float, float, float]],
    weighted:  dict[str, tuple[float, float, float]],
    patch_size: int,
    save_path:  str,
) -> None:
    """Save a grouped bar chart comparing trust mass from Table 2a and Table 2c."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as _np

    row_defs = [
        ("Clean 3",                          "clean_3"),
        ("Clean 6",                          "clean_6"),
        (f"6 w/ patch",         "pois_6_trusted"),
        # (f"6 w/ patch (patch distrusted)",   "pois_6_distrust"),
        (f"3 w/ patch",         "pois_3_trusted"),
        # (f"3 w/ patch (patch distrusted)",   "pois_3_distrust"),
    ]

    labels      = [r[0] for r in row_defs]
    keys        = [r[1] for r in row_defs]
    trust_2a    = [by_class.get(k, (float("nan"),)*3)[0] for k in keys]
    trust_2c    = [weighted.get(k, (float("nan"),)*3)[0] for k in keys]

    x     = _np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))

    bars_a = ax.bar(x - width / 2, trust_2a, width, label="per-class neuron", color="#4C72B0")
    bars_c = ax.bar(x + width / 2, trust_2c, width, label="softmax-weighted", color="#C44E52")

    for bars in (bars_a, bars_c):
        for bar in bars:
            v = bar.get_height()
            if v == v:  # skip NaN
                ax.text(
                    bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=16, rotation=90,
                )

    ax.set_ylabel("Trust mass")
    ax.set_title(f"IPTA trust mass — Table 2a vs 2c  ({patch_size}×{patch_size} patch)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=16)
    ax.set_ylim(0, 1.2)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] Trust 2a vs 2c histogram saved → {save_path}")


def print_table2(
    sweep_results: list[dict[str, Any]],
    ipta_rows:     dict[str, tuple[float, float, float]] | tuple | None,
    ipta_patch:    int = IPTA_PATCH_SIZE,
) -> None:
    def _pct(v: float) -> str:
        return f"{v*100:6.2f}%" if v == v else "  N/A  "
    def _f4(v: float) -> str:
        return f"{v:.4f}" if v == v else " N/A  "

    # Unpack 3-tuple (by_class, aggregate, weighted) or fall back gracefully
    if isinstance(ipta_rows, tuple) and len(ipta_rows) == 3:
        ipta_by_class, ipta_agg, ipta_weighted = ipta_rows
    elif isinstance(ipta_rows, tuple) and len(ipta_rows) == 2:
        ipta_by_class, ipta_agg = ipta_rows
        ipta_weighted = None
    elif isinstance(ipta_rows, dict):
        ipta_by_class, ipta_agg, ipta_weighted = ipta_rows, None, None
    else:
        ipta_by_class = ipta_agg = ipta_weighted = None

    base: dict[str, float] = {}
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
    print("  IPTA = GenIPTA(activated_neurons)(Tx)  [computed offline after training]")
    print("=" * len(sep))

    row_defs = [
        ("Clean 3",
         _pct(base.get("acc_clean_3", float("nan"))), "clean_3"),
        ("Clean 6",
         _pct(base.get("acc_clean_6", float("nan"))), "clean_6"),
        (f"6 with {ipta_patch}×{ipta_patch} patch (all trusted Tx)",
         _pct(base.get("acc_pois_6",  float("nan"))), "pois_6_trusted"),
        (f"6 with {ipta_patch}×{ipta_patch} patch (patch distrusted)",
         _pct(base.get("acc_pois_6",  float("nan"))), "pois_6_distrust"),
        (f"3 with {ipta_patch}×{ipta_patch} patch (all trusted Tx)",
         _pct(base.get("acc_pois_3",  float("nan"))), "pois_3_trusted"),
        (f"3 with {ipta_patch}×{ipta_patch} patch (patch distrusted)",
         _pct(base.get("acc_pois_3",  float("nan"))), "pois_3_distrust"),
    ]

    def _print_sub(title, data):
        print()
        print(f"  {title}")
        if data is None:
            print("  [data unavailable]")
            return
        print(sep)
        print(row_fmt.format(*headers))
        print(sep)
        for label, acc_str, key in row_defs:
            t, d, u = data.get(key, (float("nan"), float("nan"), float("nan")))
            print(row_fmt.format(label, acc_str, _f4(t), _f4(d), _f4(u)))
        print(sep)

    _print_sub("Table 2a — per-class output neuron opinion  (output at true-class neuron)", ipta_by_class)
    _print_sub("Table 2b — aggregated opinion  (PTASClass.aggregation over all neurons)", ipta_agg)
    _print_sub("Table 2c — NN-softmax-weighted opinion  (Σ p_c · opinion_c)", ipta_weighted)
    print()

    # Generate histogram PDF for Table 2a
    if ipta_by_class is not None:
        plot_dir  = base.get("datapath", "results")
        save_path = os.path.join(plot_dir, "ipta_table2a.pdf")
        try:
            plot_table2a(ipta_by_class, ipta_patch, save_path)
        except Exception as _e:
            print(f"  [plot] Table 2a histogram failed: {_e}")

    # Generate trust-only comparison plot (Table 2a vs Table 2c)
    if ipta_by_class is not None and ipta_weighted is not None:
        plot_dir  = base.get("datapath", "results")
        save_path = os.path.join(plot_dir, "ipta_trust_2a_vs_2c.pdf")
        try:
            plot_trust_2a_vs_2c(ipta_by_class, ipta_weighted, ipta_patch, save_path)
        except Exception as _e:
            print(f"  [plot] Trust 2a vs 2c histogram failed: {_e}")

# ─────────────────────────────────────────────────────────────────────────────
# pytest integration tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_poisoned_patch1():
    cfg = make_poisoned_cfg(patch_size=1, epochs=2, port=5060)
    result = run_poisoned_scenario(cfg)
    assert result["train_acc"] > 0.5, f"Low train acc: {result['train_acc']}"


@pytest.mark.integration
def test_poisoned_patch4():
    cfg = make_poisoned_cfg(patch_size=4, epochs=2, port=5061)
    result = run_poisoned_scenario(cfg)
    assert result["train_acc"] > 0.5, f"Low train acc: {result['train_acc']}"
    if result["omega_arrays"] is not None:
        ipta_paths = _read_ipta_paths(os.path.join(result["datapath"], "ipta_paths.json"))
        if ipta_paths is not None:
            structure = [cfg.input_dim, _HIDDEN_DIM, cfg.output_dim]
            ipta = compute_ipta_results(result["omega_arrays"], ipta_paths, 4, structure)
            for key in ("clean_3", "clean_6", "pois_6_trusted", "pois_6_distrust",
                        "pois_3_trusted", "pois_3_distrust"):
                if key not in ipta:
                    continue
                t, d, u = ipta[key]
                assert t + d + u <= 1.01, f"IPTA opinion sums > 1: {key} = ({t},{d},{u})"


@pytest.mark.integration
def test_poisoned_patch10():
    cfg = make_poisoned_cfg(patch_size=10, epochs=2, port=5062)
    result = run_poisoned_scenario(cfg)
    assert result["train_acc"] > 0.5, f"Low train acc: {result['train_acc']}"


@pytest.mark.integration
def test_poisoned_patch27():
    cfg = make_poisoned_cfg(patch_size=27, epochs=2, port=5063)
    result = run_poisoned_scenario(cfg)
    assert result["train_acc"] > 0.5, f"Low train acc: {result['train_acc']}"

# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "MNIST poisoned PTAS test — replicates the main.py two-process setup.\n"
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
    p.add_argument("--force-retrain", action="store_true", default=False,
                   help="Ignore saved artefacts and retrain from scratch")
    p.add_argument("--not-ptas", action="store_true", default=False,
                   help="Run the NN only — skip the PTAS server, trust/IPTA outputs will be N/A")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.patch_size is not None:
        # Single patch-size run
        cfg = make_poisoned_cfg(
            patch_size=args.patch_size,
            hidden_dim=args.hidden_dim,
            port=args.port,
            epochs=args.epochs,
        )
        result = run_poisoned_scenario(cfg, force_retrain=args.force_retrain,
                                        not_ptas=args.not_ptas)
        print_table1([result])

        if not args.not_ptas and result["omega_arrays"] is not None:
            ipta_paths = _read_ipta_paths(
                os.path.join(result["datapath"], "ipta_paths.json")
            )
            if ipta_paths is not None and args.patch_size == args.ipta_patch:
                structure = [cfg.input_dim, args.hidden_dim, cfg.output_dim]
                ipta_rows = compute_ipta_results(
                    result["omega_arrays"], ipta_paths,
                    args.patch_size, structure, cfg.input_dim,
                )
                print_table2([result], ipta_rows, ipta_patch=args.patch_size)
        return

    # Full sweep
    print(f"\nRunning full poisoned-MNIST sweep: patches={PATCH_SIZES}, "
          f"epochs={args.epochs}, hidden={args.hidden_dim}\n")

    sweep_results, ipta_rows = run_poisoned_sweep(
        patch_sizes=PATCH_SIZES,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        base_port=args.port,
        ipta_patch=args.ipta_patch,
        force_retrain=args.force_retrain,
        not_ptas=args.not_ptas,
    )

    print_table1(sweep_results)
    if not args.not_ptas:
        print_table2(sweep_results, ipta_rows, ipta_patch=args.ipta_patch)
    print("=== Poisoned MNIST test complete ===\n")


if __name__ == "__main__":
    main()
