"""
Poisoned-GTSRB PTAS integration test.

Backdoor setup mirrors poisoned MNIST (see test_mnist_poisoned.py):
  * a white trigger patch is placed top-left on class-6 and class-9 training
    images in the last third of the training set, and their labels are
    swapped (6↔9);
  * the trust generator distrusts the patch pixels of poisoned samples and
    the class-6/9 outputs of samples whose true label is 6 or 9;
  * evaluation adds the patch to test images of the poisoned class (6) and a
    clean control class (3) and measures how often the backdoor fires.

GTSRB: 43 classes, grayscale 32×32 (input_dim = 1024), pixels in [0,1].

Run standalone:
    python tests/test_gtsrb_pois.py                  # patch sweep (1, 4, 10)
    python tests/test_gtsrb_pois.py --epochs 5       # quick smoke-test
    python tests/test_gtsrb_pois.py --patch-size 4   # single patch size

Run with pytest (skipped by default; use -m integration):
    pytest tests/test_gtsrb_pois.py -m integration -s
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from typing import Any

# ── Path bootstrap ────────────────────────────────────────────────────────────
_v2_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_patas_dir = os.path.join(_v2_dir, "patas_module")
_tests_dir = os.path.dirname(os.path.abspath(__file__))
for _p in (_v2_dir, _patas_dir, _tests_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import patas_module  # noqa: F401 — triggers path bootstrap
from main import TestCaseConfig, get_lr_gtsrb
from test_mnist_poisoned import run_poisoned_scenario, print_table1

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
PATCH_SIZES:     list[int] = [1, 4, 10]
_BASE_PORT:      int       = 5111
_DEFAULT_EPOCHS: int       = 20
_HIDDEN_DIMS:    tuple[int, ...] = (128,)
_INPUT_DIM:      int       = 32 * 32
_OUTPUT_DIM:     int       = 43

# ─────────────────────────────────────────────────────────────────────────────
# Config factory
# ─────────────────────────────────────────────────────────────────────────────

def make_gtsrb_poisoned_cfg(
    patch_size:  int = 4,
    hidden_dims: tuple[int, ...] = _HIDDEN_DIMS,
    port:        int = _BASE_PORT,
    epochs:      int = _DEFAULT_EPOCHS,
) -> TestCaseConfig:
    return TestCaseConfig(
        dataset="gtsrb",
        input_dim=_INPUT_DIM,
        output_dim=_OUTPUT_DIM,
        hidden_dim=hidden_dims[0],
        hidden_dims=hidden_dims,
        epochs=epochs,
        batch_size=128,
        learning_rate=get_lr_gtsrb,
        epsilon_low=0.05,
        x_trust="trust",
        y_trust="trust",
        port=port,
        mnist_patch_size=patch_size,   # field name is legacy; applies to GTSRB too
        mnist_poisoned_soph=True,
        no_round=None,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_gtsrb_poisoned_sweep(
    patch_sizes: list[int] = PATCH_SIZES,
    epochs: int = _DEFAULT_EPOCHS,
    hidden_dims: tuple[int, ...] = _HIDDEN_DIMS,
    base_port: int = _BASE_PORT,
    force_retrain: bool = False,
) -> list[dict[str, Any]]:
    """Run all patch-size scenarios sequentially (reuses the MNIST runner —
    the trust/accuracy semantics are identical, classes 6/9 swapped, 3 as
    the clean control class)."""
    sweep_results: list[dict[str, Any]] = []
    for i, ps in enumerate(patch_sizes):
        cfg = make_gtsrb_poisoned_cfg(patch_size=ps, hidden_dims=hidden_dims,
                                      port=base_port + i, epochs=epochs)
        print(f"\n{'='*64}")
        print(f"  Poisoned GTSRB  |  patch={ps}×{ps}  |  port={cfg.port}")
        print(f"{'='*64}")
        result = run_poisoned_scenario(cfg, force_retrain=force_retrain)
        print(f"  trust(3)={result['trust_for_3']:.4f}  "
              f"trust(6)={result['trust_for_6']:.4f}  "
              f"train={result['train_acc']*100:.2f}%  "
              f"test={result['test_acc']*100:.2f}%")
        print(f"  clean3={result['acc_clean_3']*100:.2f}%  "
              f"clean6={result['acc_clean_6']*100:.2f}%  "
              f"pois3={result['acc_pois_3']*100:.2f}%  "
              f"pois6={result['acc_pois_6']*100:.2f}%")
        sweep_results.append(result)
        time.sleep(2)
    return sweep_results

# ─────────────────────────────────────────────────────────────────────────────
# pytest integration tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_gtsrb_poisoned_patch4():
    """Poisoned GTSRB: 4×4 trigger patch, 128 hidden neurons."""
    cfg = make_gtsrb_poisoned_cfg(patch_size=4, port=5115, epochs=2)
    result = run_poisoned_scenario(cfg, force_retrain=True)
    assert result["train_acc"] > 0.2, f"Low train acc: {result['train_acc']}"
    assert 0.0 <= result["trust_for_6"] <= 1.0


@pytest.mark.integration
def test_gtsrb_poisoned_patch10():
    """Poisoned GTSRB: 10×10 trigger patch, 128 hidden neurons."""
    cfg = make_gtsrb_poisoned_cfg(patch_size=10, port=5116, epochs=2)
    result = run_poisoned_scenario(cfg, force_retrain=True)
    assert result["train_acc"] > 0.2, f"Low train acc: {result['train_acc']}"

# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Poisoned-GTSRB PTAS test (trigger-patch 6↔9 backdoor).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--patch-size", type=int, default=None,
                   help="Single patch size (default: sweep 1, 4, 10)")
    p.add_argument("--hidden-neurons", type=int, default=_HIDDEN_DIMS[0])
    p.add_argument("--hidden-neurons-2", type=int, default=None)
    p.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    p.add_argument("--port", type=int, default=_BASE_PORT)
    p.add_argument("--force-retrain", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    h2 = args.hidden_neurons_2
    hidden_dims = (args.hidden_neurons, h2) if h2 is not None else (args.hidden_neurons,)
    patch_sizes = [args.patch_size] if args.patch_size is not None else PATCH_SIZES

    results = run_gtsrb_poisoned_sweep(
        patch_sizes=patch_sizes, epochs=args.epochs,
        hidden_dims=hidden_dims, base_port=args.port,
        force_retrain=args.force_retrain,
    )
    print_table1(results)
    print("\n=== Poisoned GTSRB test complete ===\n")


if __name__ == "__main__":
    main()
