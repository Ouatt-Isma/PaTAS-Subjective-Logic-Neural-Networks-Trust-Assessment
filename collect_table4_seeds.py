"""collect_table4_seeds.py — aggregate repeated poisoned-MNIST IPTA runs.

Parses the "Table 2a" blocks printed by tests/test_mnist_poisoned.py from
one or more job output files and reports mean ± std per sample type and
opinion component, restoring the multi-seed protocol of the paper's
Table IV.

Usage:
    python collect_table4_seeds.py job_1234.out [job_5678.out ...]
"""
from __future__ import annotations

import re
import sys
import json

import numpy as np

ROW = re.compile(r"^(?P<name>[0-9A-Za-z].*?)\s{2,}(?P<acc>[\d.]+)%\s+"
                 r"(?P<t>[\d.]+)\s+(?P<d>[\d.]+)\s+(?P<u>[\d.]+)\s*$")


def parse(paths):
    runs = []
    for path in paths:
        text = open(path, encoding="utf-8", errors="replace").read()
        for block in re.findall(
                r"Table 2a[^\n]*\n-+\n[^\n]*\n-+\n(.*?)\n-+\n", text, re.S):
            rows = {}
            for line in block.splitlines():
                m = ROW.match(line.strip())
                if m:
                    rows[m["name"].strip()] = (float(m["acc"]), float(m["t"]),
                                               float(m["d"]), float(m["u"]))
            if rows:
                runs.append(rows)
    return runs


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    runs = parse(sys.argv[1:])
    if not runs:
        raise SystemExit("No 'Table 2a' blocks found in the given files.")
    print(f"{len(runs)} runs parsed")
    names = [n for n in runs[0] if all(n in r for r in runs)]
    agg = {}
    print(f"\n{'Sample type':40s} {'acc%':>14} {'trust':>16} "
          f"{'distrust':>16} {'uncertainty':>16}")
    for name in names:
        vals = np.array([r[name] for r in runs])       # (n_runs, 4)
        mean, std = vals.mean(0), vals.std(0)
        agg[name] = {"acc": [round(mean[0], 2), round(std[0], 2)],
                     "trust": [round(mean[1], 4), round(std[1], 4)],
                     "distrust": [round(mean[2], 4), round(std[2], 4)],
                     "uncertainty": [round(mean[3], 4), round(std[3], 4)]}
        print(f"{name:40s} "
              f"{mean[0]:7.2f}±{std[0]:5.2f} "
              f"{mean[1]:9.4f}±{std[1]:6.4f} "
              f"{mean[2]:9.4f}±{std[2]:6.4f} "
              f"{mean[3]:9.4f}±{std[3]:6.4f}")
    out = "results/table4_seeds.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"n_runs": len(runs), "rows": agg}, fh, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
