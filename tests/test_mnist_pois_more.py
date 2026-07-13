"""
Poisoned-MNIST PTAS integration test — larger architectures.

Same backdoor setup as test_mnist_poisoned.py (trigger patch top-left,
6↔9 label swap on the last third of the training set, patch-distrusting
trust generator) but with the wider/deeper architectures:

    784-1000-10      (one hidden layer, 1000 neurons)
    784-500-500-10   (two hidden layers, 500 neurons each)

Run standalone:
    python tests/test_mnist_pois_more.py                 # both archs, patch 4
    python tests/test_mnist_pois_more.py --epochs 5      # quick smoke-test
    python tests/test_mnist_pois_more.py --patch-size 10 # different patch
    python tests/test_mnist_pois_more.py --arch 1000     # single architecture
    python tests/test_mnist_pois_more.py --arch 500-500

Run with pytest (skipped by default; use -m integration):
    pytest tests/test_mnist_pois_more.py -m integration -s
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
from main import TestCaseConfig, get_lr_mnist
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
ARCHS: list[tuple[int, ...]] = [(1000,), (500, 500)]
_DEFAULT_PATCH:  int = 4
_BASE_PORT:      int = 5101
_DEFAULT_EPOCHS: int = 20

# ─────────────────────────────────────────────────────────────────────────────
# Config factory
# ─────────────────────────────────────────────────────────────────────────────

def make_poisoned_more_cfg(
    hidden_dims: tuple[int, ...] = (1000,),
    patch_size:  int = _DEFAULT_PATCH,
    port:        int = _BASE_PORT,
    epochs:      int = _DEFAULT_EPOCHS,
) -> TestCaseConfig:
    return TestCaseConfig(
        dataset="mnist",
        input_dim=28 * 28,
        output_dim=10,
        hidden_dim=hidden_dims[0],
        hidden_dims=hidden_dims,
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
# Sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_arch_sweep(
    archs: list[tuple[int, ...]] = ARCHS,
    patch_size: int = _DEFAULT_PATCH,
    epochs: int = _DEFAULT_EPOCHS,
    base_port: int = _BASE_PORT,
    force_retrain: bool = False,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for i, dims in enumerate(archs):
        arch = "-".join(str(h) for h in dims)
        cfg = make_poisoned_more_cfg(hidden_dims=dims, patch_size=patch_size,
                                     port=base_port + i, epochs=epochs)
        print(f"\n{'='*64}")
        print(f"  Poisoned MNIST  |  arch={arch}  patch={patch_size}×{patch_size}  port={cfg.port}")
        print(f"{'='*64}")
        result = run_poisoned_scenario(cfg, force_retrain=force_retrain)
        result["arch"] = arch
        print(f"  trust(3)={result['trust_for_3']:.4f}  "
              f"trust(6)={result['trust_for_6']:.4f}  "
              f"train={result['train_acc']*100:.2f}%  "
              f"test={result['test_acc']*100:.2f}%")
        print(f"  clean3={result['acc_clean_3']*100:.2f}%  "
              f"clean6={result['acc_clean_6']*100:.2f}%  "
              f"pois3={result['acc_pois_3']*100:.2f}%  "
              f"pois6={result['acc_pois_6']*100:.2f}%")
        results.append(result)
        time.sleep(2)
    return results


def print_arch_table(results: list[dict[str, Any]]) -> None:
    def _pct(v: float) -> str:
        return f"{v*100:6.2f}%" if v == v else "  N/A  "
    def _f4(v: float) -> str:
        return f"{v:.4f}" if v == v else " N/A  "

    headers = ["Arch", "Trust(3)", "Trust(6)", "Train(%)", "Test(%)",
               "Clean 3(%)", "Clean 6(%)", "3+patch(%)", "6+patch(%)"]
    col_w = [10, 9, 9, 9, 9, 11, 11, 11, 11]
    row_fmt = "  ".join([f"{{:<{col_w[0]}}}"] + [f"{{:>{w}}}" for w in col_w[1:]])
    sep = "-" * (sum(col_w) + 2 * (len(col_w) - 1))

    print()
    print(sep)
    print(row_fmt.format(*headers))
    print(sep)
    for r in results:
        print(row_fmt.format(
            r.get("arch", "?"),
            _f4(r["trust_for_3"]), _f4(r["trust_for_6"]),
            _pct(r["train_acc"]), _pct(r["test_acc"]),
            _pct(r["acc_clean_3"]), _pct(r["acc_clean_6"]),
            _pct(r["acc_pois_3"]), _pct(r["acc_pois_6"]),
        ))
    print(sep)
    print()

# ─────────────────────────────────────────────────────────────────────────────
# pytest integration tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_poisoned_1000():
    """Poisoned MNIST: 784-1000-10, patch 4×4."""
    cfg = make_poisoned_more_cfg(hidden_dims=(1000,), patch_size=4,
                                 port=5105, epochs=2)
    result = run_poisoned_scenario(cfg, force_retrain=True)
    assert result["train_acc"] > 0.5, f"Low train acc: {result['train_acc']}"
    assert 0.0 <= result["trust_for_6"] <= 1.0


@pytest.mark.integration
def test_poisoned_500_500():
    """Poisoned MNIST: 784-500-500-10, patch 4×4."""
    cfg = make_poisoned_more_cfg(hidden_dims=(500, 500), patch_size=4,
                                 port=5106, epochs=2)
    result = run_poisoned_scenario(cfg, force_retrain=True)
    assert result["train_acc"] > 0.5, f"Low train acc: {result['train_acc']}"
    assert 0.0 <= result["trust_for_6"] <= 1.0

# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_arch(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.replace("_", "-").split("-"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Poisoned-MNIST PTAS test — larger architectures (1000, 500-500).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--arch", default=None,
                   help="Single architecture, e.g. 1000 or 500-500 (default: both)")
    p.add_argument("--patch-size", type=int, default=_DEFAULT_PATCH)
    p.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    p.add_argument("--port", type=int, default=_BASE_PORT)
    p.add_argument("--force-retrain", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    archs = [_parse_arch(args.arch)] if args.arch else ARCHS
    results = run_arch_sweep(
        archs=archs, patch_size=args.patch_size,
        epochs=args.epochs, base_port=args.port,
        force_retrain=args.force_retrain,
    )
    print_arch_table(results)
    print("\n=== Poisoned MNIST (more architectures) test complete ===\n")


if __name__ == "__main__":
    main()
