"""
Data loader and pre-processing for the 5G Network Energy Consumption Dataset.

Expected input files (all CSV):
    BSinfo.csv  : BS, CellName, RUType, Mode, Bandwidth, Frequency, Antennas, TXpower
    CLstat.csv  : Time, BS, CellName, Load, ESMode1..ESMode6
    ECstat.csv  : Time, BS, Energy

Pipeline:
    1.  Aggregate cell-level static information up to the BS level
        (means for numeric, modes for categorical).
    2.  Aggregate hourly cell-level statistics up to (Time, BS).
    3.  Join with hourly energy and bin Energy into K classes by
        empirical quantiles → target variable ``y`` (default K=3).
    4.  One-hot encode categorical fields, standardise numerics.

Also provides a synthetic generator so the rest of the codebase can be
exercised end-to-end before the real CSVs are available.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test:  np.ndarray
    y_test:  np.ndarray
    feature_names: List[str]
    class_names: List[str]
    raw: pd.DataFrame | None = None  # merged hourly table, for trust assessment


# ------------------------------------------------------------------ #
#  Real-data loading                                                 #
# ------------------------------------------------------------------ #
def load_5g_dataset(data_dir: str,
                    n_classes: int = 3,
                    test_frac: float = 0.2,
                    seed: int = 0) -> Dataset:
    """Load and pre-process the 5G dataset from ``data_dir``."""
    bs_info = pd.read_csv(os.path.join(data_dir, "BSinfo.csv"))
    cl_stat = pd.read_csv(os.path.join(data_dir, "CLstat.csv"))
    ec_stat = pd.read_csv(os.path.join(data_dir, "ECstat.csv"))

    # --- 1.  BS-level static info (aggregate cells) ---
    num_cols = ["Bandwidth", "Frequency", "Antennas", "TXpower"]
    cat_cols = ["RUType", "Mode"]
    bs_static = (
        bs_info.groupby("BS")
        .agg({**{c: "mean" for c in num_cols},
              **{c: lambda x: x.mode().iloc[0] for c in cat_cols},
              "CellName": "nunique"})
        .rename(columns={"CellName": "NumCells"})
        .reset_index()
    )

    # --- 2.  Hourly cell stats → (Time, BS) ---
    es_cols = [c for c in cl_stat.columns if c.startswith("ESMode")]
    hourly = (
        cl_stat.groupby(["Time", "BS"])
        .agg({**{"load": "mean"},
              **{c: "mean" for c in es_cols}})
        .reset_index()
    )

    # --- 3.  Merge with energy + label ---
    df = hourly.merge(ec_stat, on=["Time", "BS"], how="inner")
    df = df.merge(bs_static, on="BS", how="left")

    # Quantile-based binning of Energy
    qs = np.linspace(0, 1, n_classes + 1)[1:-1]
    thresholds = df["Energy"].quantile(qs).values
    y = np.digitize(df["Energy"].values, thresholds)

    # --- 4.  Encode + standardise ---
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    feat_cols = (["load"] + es_cols + num_cols + ["NumCells"]
                 + [c for c in df_enc.columns
                    if c.startswith("RUType_") or c.startswith("Mode_")])
    X = df_enc[feat_cols].values.astype(np.float32)

    # standardise numerics
    mu, sd = X.mean(0), X.std(0) + 1e-8
    X = (X - mu) / sd

    # train / test split
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(test_frac * len(X))
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    class_names = [f"low", "mid", "high"][:n_classes] if n_classes <= 3 \
        else [f"class_{i}" for i in range(n_classes)]

    return Dataset(
        X_train=X[train_idx], y_train=y[train_idx],
        X_test=X[test_idx],   y_test=y[test_idx],
        feature_names=feat_cols, class_names=class_names,
        raw=df,
    )


# ------------------------------------------------------------------ #
#  Synthetic generator -- lets you run train.py without the CSVs     #
# ------------------------------------------------------------------ #
def make_synthetic_5g(n_bs: int = 60,
                      n_hours: int = 192,           # 8 days
                      cells_per_bs: int = 3,
                      seed: int = 0) -> str:
    """Create a synthetic copy of the 5G dataset and return its directory."""
    rng = np.random.default_rng(seed)
    out_dir = "/tmp/synthetic_5g"
    os.makedirs(out_dir, exist_ok=True)

    rutypes = [f"RU_{i}" for i in range(12)]
    modes = ["TDD", "FDD"]

    # BSinfo
    rows = []
    for bs in range(n_bs):
        ru = rutypes[bs % 12]
        mode = modes[bs % 2]
        for c in range(cells_per_bs):
            rows.append({
                "BS": bs, "CellName": f"Cell{c}",
                "RUType": ru, "Mode": mode,
                "Bandwidth": float(rng.uniform(0.1, 1.0)),
                "Frequency": float(rng.uniform(0.1, 1.0)),
                "Antennas": int(rng.choice([2, 4, 8, 16, 32, 64])),
                "TXpower":  float(rng.uniform(0.1, 1.0)),
            })
    bs_info = pd.DataFrame(rows)
    bs_info.to_csv(os.path.join(out_dir, "BSinfo.csv"), index=False)

    # CLstat
    times = pd.date_range("2024-01-01", periods=n_hours, freq="h")
    rows = []
    for t in times:
        for bs in range(n_bs):
            for c in range(cells_per_bs):
                load = float(np.clip(rng.normal(0.45, 0.18), 0, 1))
                es = rng.uniform(0, 1, 6) * (1.0 - load)
                rows.append({
                    "Time": t, "BS": bs, "CellName": f"Cell{c}",
                    "load": load,
                    **{f"ESMode{i+1}": float(es[i]) for i in range(6)},
                })
    cl_stat = pd.DataFrame(rows)
    cl_stat.to_csv(os.path.join(out_dir, "CLstat.csv"), index=False)

    # ECstat -- energy depends on load + antennas + txpower + noise
    energies = []
    for t in times:
        for bs in range(n_bs):
            ld = cl_stat[(cl_stat.Time == t) & (cl_stat.BS == bs)]["load"].mean()
            bs_rows = bs_info[bs_info.BS == bs]
            tx = bs_rows["TXpower"].mean()
            ant = bs_rows["Antennas"].mean() / 64.0
            e = 0.4 * ld + 0.3 * tx + 0.2 * ant + 0.1 * rng.normal(0, 0.15)
            energies.append({"Time": t, "BS": bs, "Energy": float(max(0, e))})
    ec_stat = pd.DataFrame(energies)
    ec_stat.to_csv(os.path.join(out_dir, "ECstat.csv"), index=False)

    return out_dir
