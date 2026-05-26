# PaTAS — Parallel Trust Assessment System

A complete, self-contained implementation of the **Parallel Trust Assessment
System (PaTAS)** from the dissertation (Chapters 5–7), applied to:

1. Classifying the energy-consumption class of a 5G base station from the
   *5G Network Energy Consumption Dataset* (BSinfo / CLstat / ECstat).
2. Reproducing **all five experiments** of §6.5, §6.6, and §7.8.

The neural network and PaTAS are implemented in pure NumPy so that every
gradient and activation is directly visible to the trust machinery, as
required by Algorithm 5 (Trust Feedforward) and Algorithm 6
(Parameter-Trust Update).

## Requirements

```bash
# Core (required for all experiments in main.py):
pip install numpy pandas scikit-learn

# Additional (required for the standalone evaluation/plot scripts):
pip install matplotlib
```

## Quick start

```bash
# Main 5G use case — synthetic data (no CSVs needed):
python main.py 5g

# Main 5G use case — real CSVs:
python main.py 5g --data-dir ./data

# Individual dissertation experiments:
python main.py bc           # §7.8.1 Exp 1 — Breast Cancer    (Table 7.2)
python main.py mnist        # §7.8.1 Exp 2 — MNIST             (Table 7.3)
python main.py poisoned     # §7.8.1 Exp 3 — Poisoned MNIST    (Tables 7.4-7.5)
python main.py gtsrb        # §6.5         — Balancing Bias
python main.py cifar10h     # §6.6         — Labeling Bias

# Run everything (a few minutes on a laptop CPU):
python main.py all

# Skip slow experiments:
python main.py all --skip poisoned gtsrb

# List all experiment keys with descriptions:
python main.py --list
```

## File map

| File | Dissertation section |
|---|---|
| `main.py` | **Entry point** — run any experiment from the command line |
| `subjective_logic.py` | SL operators (⊗ ⊕ ⊖ ⊙ ⊘ ⊚) and BPQ / EWQ / CUQ (Eq. 6.3–6.5) |
| `data_loader.py` | Joins the three 5G CSVs, bins Energy into K classes |
| `primary_nn.py` | Standard MLP (input → ReLU → softmax), §7.8 family |
| `patas.py` | Trust Nodes Network (Def. 7.2), Trust Function (Def. 7.4), Trust Feedforward (Alg. 5), Parameter-Trust Update (Alg. 6), GenIPTA / IPTA (§7.3) |
| `trust_assessment.py` | Chapter 6 dataset trust assessment: (P, S, G) framework |
| `degradations.py` | Feature/label perturbations and patch injection from §7.8.1 |
| `training.py` | Shared training loop with optional Parameter-Trust Update |
| `eval_helpers.py` | Trust under canonical profiles, per-class trust, per-sample IPTA |
| `train.py` | 5G end-to-end pipeline function `train_and_evaluate()` |
| `breast_cancer.py` | Experiment 1 — Table 7.2 |
| `mnist.py` | Experiment 2 — Table 7.3 (sklearn digits proxy) |
| `poisoned_mnist.py` | Experiment 3 — Tables 7.4-7.5 (sklearn digits proxy) |
| `gtsrb.py` | Experiment 4 — Balancing Bias (§6.5, Ref. [5]) |
| `cifar10h.py` | Experiment 5 — Labeling Bias (§6.6, Ref. [6]) |
| `data/` | Real 5G CSVs (BSinfo.csv, CLstat.csv, ECstat.csv) |

**Standalone evaluation and plot scripts** (require `matplotlib`):

| File | Purpose |
|---|---|
| `eval_chapter.py` | Generates accuracy/trust figures and LaTeX tables for the dissertation chapter |
| `eval_5g_noise.py` | Noise-robustness grid: Gaussian feature noise × label-flip rates × three PaTAS configs |
| `calibration_trust_eval.py` | Algorithm 5 — calibration-based trust evaluation (produces `calibration_trust.pdf`) |
| `latency_eval.py` | NN vs PaTAS inference and training latency benchmark |
| `noise_utils.py` | Noise-injection helpers and trust-opinion mapping shared by the eval scripts |
| `plot_effectiveness_hardcoded.py` | Reproduces the effectiveness figure from hardcoded table data |
| `plot_latency_hardcoded.py` | Reproduces the latency figure from hardcoded table data |
| `external_bridge.py` | Adapter layer connecting the eval scripts to the `patas_module` implementation |
| `patas_module/` | Full socket-based PaTAS implementation used by the eval scripts |

## What each experiment reproduces

**Breast Cancer** (§7.8.1 Exp 1, Table 7.2).  30-16-2 MLP, 15 epochs,
lr 0.2.  Trained under nine X/Y trust combinations (trusted / uncertain /
distrusted) plus two intermediate scenarios, for ε ∈ {0.01, 0.1}.  Reports
final trust mass and accuracy.  Clean X+Y → high trust mass; uncertain or
distrusted Y → trust collapses to ≈ 0; distrusted X → trust collapses
regardless of Y.

**MNIST** (§7.8.1 Exp 2, Table 7.3).  Four architectures
input-{16,32,64,128}-10 trained on uncertain X/Y, plus a fully-trusted bonus
row on the smallest architecture.  Trust mass grows with hidden size on
uncertain data, but **training a small architecture on trusted data beats
training a larger one on uncertain data** by a wide margin.

> *Real 28×28 MNIST requires an internet download.  We use sklearn's 8×8
> `load_digits` as a local proxy.  All trust mechanics are identical; only
> `n_in` changes from 784 to 64.*

**Poisoned MNIST** (§7.8.1 Exp 3, Tables 7.4-7.5).  Patch + label-flip
attack (6 ↔ 9) on one third of the training rows.  Trust setup: patch pixels
distrusted, labels of patched rows distrusted.  **PaTAS-TP gives the clean
reference class non-zero trust mass while the poisoned class collapses to
zero**, even though both classes reach similar accuracy on clean test samples.
The IPTA block flags poisoned-class inference paths as vacuous.

**GTSRB Balancing Bias** (§6.5, Reference [5]).  The Chapter 6 framework
applied to class balance, with two trust sources:
- **Method 1 (CUQ)** on the global class-probabilities distribution
  (Scenario 1: original vs SMOTE-augmented).
- **Method 2 (BPQ)** on per-contributor entropies (Scenario 2: M = 10 and
  M = 100 contributors with a growing fraction of imbalanced ones).

> *Real GTSRB requires an internet download; we synthesise equivalent
> class counts.  The trust mechanics are exact.*

**CIFAR-10H Labeling Bias** (§6.6, Reference [6]).  Per-entry opinions from
M annotators per image, plus the annotator-level extension.  Reproduces
Fig. 6.8: full (M ≈ 50) gives b ≈ 0.77, u ≈ 0.04 (evidence saturation);
cropped to 10 gives b ≈ 0.50, d ≈ 0.33, u ≈ 0.17 (mass redistributed).

> *Real CIFAR-10H requires a 250 MB download; we simulate annotators with
> three reliability tiers.  The trust mechanics are exact.*

## How the dissertation's theorems are verified

After running the 5G classifier you should observe:

- **Theorem 7.2** — vacuous input ⇒ vacuous output:
  `Input = vacuous → ω(b=0.000, d=0.000, u=1.000)`
- **Theorem 7.3** — symmetric inference: trusted inputs give
  `(b=t, d=0, u=1-t)` and distrusted inputs give the exact mirror
  `(b=0, d=t, u=1-t)`.

## Implementation notes

Three places where the dissertation leaves room for interpretation are
documented inline in the code:

1. **Auxiliary update (Step 6 of Alg. 6).**  ⊙ in §7.6 admits two
   readings: Jøsang's binomial AND or trust discounting.  We use
   discounting because applying AND every mini-batch compounds distrust
   unboundedly and contradicts the t = 0.87 result in Table 7.2.

2. **Bias in the Trust Function.**  Def. 7.4 strictly fuses only the
   weighted-input contributions — bias trust is maintained and updated
   but is not fused into the per-neuron output opinion.

3. **Class-aware label trust on layer 2.**  When per-row labels are
   available, the label-trust opinion T_y used in Step 6 of Alg. 6 is
   aggregated *per output class* — only rows whose true label is c
   contribute to the T_y of edges feeding output neuron c.  This lets a
   class-targeted poisoning attack show up as a class-specific drop in
   parameter trust (cf. Table 7.4).  Pass `y_labels=yb` to
   `parameter_trust_update` to enable this; if omitted, the batch-wide
   T_y is used for all edges.

## Choosing the gradient threshold ε

ε controls how NODETRUST (Alg. 6) splits gradients into positive and
negative evidence.  Per §7.9, ε must track the gradient scale.  The
default is **ε = learning_rate** (`eps_factor = 1.0`); override per
experiment with the `eps_factor` argument to `training.train()`.
