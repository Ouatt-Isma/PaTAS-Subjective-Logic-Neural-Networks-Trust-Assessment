# PaTAS — Parallel Trust Assessment System: 5G Energy Use Case

This repository contains the complete evaluation code for the **Parallel Trust
Assessment System (PaTAS)** applied to a 5G network energy-consumption
classification task, as reported in the PhD dissertation (Chapters 6–7).

The neural network and PaTAS trust engine are implemented via the
`patas_module` submodule (pure NumPy), making every gradient and subjective-
logic operation directly visible to the trust machinery.

## Cloning

`patas_module` is a git submodule. Use one of the following:

```bash
# Clone everything in one command (recommended):
git clone --recurse-submodules https://github.com/Ouatt-Isma/PaTAS-Subjective-Logic-Neural-Networks-Trust-Assessment.git

# Or, after a plain clone, initialise the submodule separately:
git submodule update --init
```

> Run `git submodule update --init` from the **repository root**, not from
> inside the `patas_module/` directory.

## Requirements

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Dataset

The evaluation uses the **5G Network Energy Consumption Dataset** with three
CSV files placed in `data/`:

| File | Contents |
|---|---|
| `BSinfo.csv` | Per-cell static info: RUType, Mode, Bandwidth, Frequency, Antennas, TXpower |
| `CLstat.csv` | Hourly cell-level statistics: Load, ESMode1–ESMode6 |
| `ECstat.csv` | Hourly per-BS energy consumption |

`data_loader.py` aggregates cells to BS level, joins the three files, and bins
energy into three classes (low / mid / high) by empirical quantiles.  A
synthetic generator (`make_synthetic_5g`) is available when the real CSVs are
not present.

## Evaluation scripts

### 1. Noise-robustness study — `eval_5g_noise.py`

The main experiment.  Trains and evaluates two configurations across a grid of
noise conditions, averaged over `N_RUNS = 5` independent noise draws:

| Configuration | Description |
|---|---|
| `nn` | Standard NN, no trust reasoning (baseline) |
| `patas-cal-fb` | PaTAS with trust opinions calibrated to the noise level |

**Noise grid:**

- Feature noise: Gaussian measurement noise at σ ∈ {0, 0.1, 0.3, 0.5}
- Label noise: random mislabelling at flip rate p ∈ {0, 0.05, 0.15, 0.30}
- Combined: four (σ, p) pairs covering both noise sources simultaneously

Trust opinions are mapped from noise parameters to subjective-logic triples
(b, d, u) by `noise_utils.py`:  feature noise grows uncertainty (u), label
noise grows disbelief (d).

```bash
python eval_5g_noise.py data/                            # run all + plot
python eval_5g_noise.py data/ --force                    # re-run even if cached
python eval_5g_noise.py data/ --plots-only               # skip training, plot from cache
python eval_5g_noise.py data/ --plots-only --per-sample  # per-sample trust in effectiveness plot
python eval_5g_noise.py data/ --plots-only --trust       # fully-trusted opinion in effectiveness plot
```

The following mutually exclusive flags control how trust opinions are assigned in the PaTAS
effectiveness analysis (`patas_effectiveness_agg.pdf`) only:

| Flag | Trust assigned to each test input |
|---|---|
| *(default)* | Uniform `feature_noise_to_trust(σ)` — same opinion for every sample |
| `--per-sample` | Per-sample `feature_noise_to_trust(σ_i)` where σ_i ~ Uniform(0, σ); input is also noised accordingly |
| `--trust` | Fully-trusted opinion `(b=1, d=0, u=0)` for every sample |

**Outputs** (saved under `results_noise/` by default):

| Path | Content |
|---|---|
| `plots/trust_mapping.pdf` | Trust opinion (b, d, u) vs noise parameter |
| `plots/feature_noise_accuracy.pdf` | NN accuracy vs feature noise level |
| `plots/label_noise_accuracy.pdf` | NN accuracy vs label flip rate |
| `plots/combined_accuracy.pdf` | Accuracy under combined noise (grouped bars) |
| `plots/patas_improvement.pdf` | Δ accuracy = PaTAS − NN across all conditions |
| `plots/learning_curves.pdf` | Epoch-by-epoch accuracy for selected conditions |
| `plots/patas_effectiveness_agg.pdf` | PaTAS opinion masses in predicted class (correct vs wrong) |
| `plots/coverage_accuracy_*.pdf` | Coverage–accuracy curves sweeping confidence threshold |
| `tables/trust_mapping.tex` | Trust-opinion mapping table |
| `tables/feature_noise_results.tex` | Per-σ accuracy table (NN vs PaTAS, mean ± std) |
| `tables/label_noise_results.tex` | Per-p accuracy table |
| `tables/combined_results.tex` | Combined-noise accuracy table |
| `tables/patas_effectiveness.tex` | Mean opinion masses split by correct/wrong predictions |

---

### 2. Chapter evaluation — `eval_chapter.py`

Evaluates PaTAS under five canonical trust configurations on the 5G dataset:
fully trusted, vacuous, trusted features + vacuous labels, distrusted features,
and fully distrusted.  Produces accuracy comparisons, learning curves, trust-
mass evolution plots, and weight-opinion heatmaps for the dissertation chapter.

```bash
python eval_chapter.py data/
python eval_chapter.py data/ --plots-only
python eval_chapter.py data/ --force
python eval_chapter.py data/ --output results_chapter/
```

**Outputs** (saved under `results_chapter/` by default):

| Path | Content |
|---|---|
| `plots/accuracy_comparison.pdf` | Final accuracy bar chart per trust config |
| `plots/learning_curves.pdf` | Accuracy vs epoch per config |
| `plots/trust_convergence_tt.pdf` | Trust mass evolution — fully trusted |
| `plots/trust_convergence_vv.pdf` | Trust mass evolution — vacuous |
| `plots/trust_convergence_dt.pdf` | Trust mass evolution — distrusted features |
| `plots/omega_beliefs.pdf` | Heatmap of per-weight belief opinions after training |
| `tables/accuracy_summary.tex` | Final accuracy and trust mass per config |
| `tables/trust_masses.tex` | Detailed trust mass summary |

---

### 3. Calibration trust evaluation — `calibration_trust_eval.py`

Implements **Algorithm 5** (Calibration-based Trust Evaluation).  For each
trained NN, predicted probabilities are binned, and calibration error per bin
is converted to a binomial opinion (b, d, u) via BPQ.  Class opinions are
fused with cumulative belief fusion to produce a single trust opinion per model.

Scans cached `nn_weights.pkl` files produced by `eval_5g_noise.py` and writes:

```bash
python calibration_trust_eval.py results_noise/
```

**Outputs:**

| Path | Content |
|---|---|
| `results_noise/dissertation/plots/calibration_trust.pdf` | Trust opinion (b, d, u) vs noise condition |
| `results_noise/dissertation/tables/calibration_trust.tex` | Numerical summary table |

---

### 4. Latency benchmark — `latency_eval.py`

Measures **inference** and **training** latency of the NN baseline vs PaTAS.

- **Inference (Part A):** NN forward pass vs `PTAS.apply_feedforward`, varying
  batch size ∈ {1, 8, 32, 64, 128, 256, 512, 1024}.  Repeated 200 times;
  median reported.
- **Training (Part B):** Wall-clock time for one full training run (10 epochs)
  without vs with PaTAS.  Repeated 5 times; mean ± std reported.

```bash
python latency_eval.py results_noise/ [--data-dir data/]
```

**Outputs:**

| Path | Content |
|---|---|
| `results_noise/dissertation/plots/latency.pdf` | Inference and training latency plots |
| `results_noise/dissertation/tables/latency.tex` | Latency summary table |

---

### 5. Hardcoded plots — `plot_effectiveness_hardcoded.py`, `plot_latency_hardcoded.py`

Reproduce the dissertation figures from hardcoded table data — no model files
required.  Useful for regenerating plots without re-running experiments.

```bash
python plot_effectiveness_hardcoded.py   # → patas_effectiveness_test.pdf/.png
python plot_latency_hardcoded.py         # → latency_hardcoded.pdf/.png
```

---

## File map

| File | Role |
|---|---|
| `eval_5g_noise.py` | **Main evaluation** — noise robustness study |
| `eval_chapter.py` | Chapter evaluation — trust configurations |
| `calibration_trust_eval.py` | Algorithm 5 — calibration-based trust |
| `latency_eval.py` | Inference and training latency benchmark |
| `plot_effectiveness_hardcoded.py` | Reproduce effectiveness figure from hardcoded data |
| `plot_latency_hardcoded.py` | Reproduce latency figure from hardcoded data |
| `external_bridge.py` | Adapter connecting eval scripts to `patas_module` |
| `data_loader.py` | 5G dataset loader and synthetic generator |
| `noise_utils.py` | Noise injection and trust-opinion mapping |
| `hyperparams.py` | Shared hyperparameters (grid, epochs, LR, hidden size) |
| `subjective_logic.py` | Shim re-exporting SL operators from `patas_module` |
| `degradations.py` | Feature/label perturbation functions |
| `eval_helpers.py` | Trust evaluation helpers (canonical profiles, IPTA) |
| `tex.py` | Collects result PDFs into `latex/filelist.tex` |
| `patas_module/` | Full PaTAS implementation — see [`patas_module/readme.md`](patas_module/readme.md) |
| `data/` | Real 5G CSVs (BSinfo.csv, CLstat.csv, ECstat.csv) |

## Hyperparameters

All shared hyperparameters are in `hyperparams.py`:

| Parameter | Value | Description |
|---|---|---|
| `FEATURE_SIGMAS` | [0, 0.1, 0.3, 0.5] | Feature noise levels for the robustness grid |
| `LABEL_FLIPS` | [0, 0.05, 0.15, 0.30] | Label flip rates for the robustness grid |
| `COMBINED` | 4 pairs | (σ, p) conditions for the combined-noise study |
| `N_RUNS` | 5 | Independent noise-draw repetitions (results averaged) |
| `EPOCHS` | 20 | Training epochs per run |
| `BATCH` | 64 | Mini-batch size |
| `LR` | 0.01 | Learning rate |
| `N_HIDDEN` | 32 | Hidden-layer size |

## Trust-opinion mapping

Feature noise and label noise are mapped to subjective-logic opinions by
`noise_utils.py`:

- **Feature noise** (measurement uncertainty) → uncertainty-dominated opinion:
  `u = 2σ²/(1+σ²)`, then `b` and `d` split the remaining mass by SNR.
- **Label noise** (active mislabelling) → disbelief-dominated opinion with
  fixed residual uncertainty `u = 0.2`:
  `b = (1−p)(1−u)`, `d = p(1−u)`.

Clean conditions (σ = 0 or p = 0) map to the fully-trusted opinion (b=1, d=0, u=0).
