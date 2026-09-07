#!/bin/bash
#SBATCH --job-name=attr
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=10:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

module load compiler/gnu/14.2
module load devel/python/3.11.7-gnu-14.2
module load devel/cuda/12.8
source ~/myenv/bin/activate

cd "$HOME/PaTAS"
REPO=PaTAS-Subjective-Logic-Neural-Networks-Trust-Assessment

# ===========================================================================
# Repaired parameter trust: attribution + magnitude-weighted path trust.
#
# The controls showed the published model-side results were produced by the
# analyst's asserted opinions, not by the data. The repair attributes each
# parameter's opinion to the samples that actually formed it, so coarse
# provenance knowledge ("this data source is untrusted") reaches the
# parameters and the inference paths that used them. All runs below are
# offline replays of models that are already cached; nothing retrains.
# ===========================================================================

# ---- 1. Backdoored model (4x4 patch), four trust sources -------------------
# provenance: source-level knowledge only, no trigger or sample knowledge.
python $REPO/attribution.py --dataset mnist --arch 128 --poisoned-patch 4 \
    --trust-source provenance --untrusted-opinion 0,1,0 --tag attr-prov-distrust
python $REPO/attribution.py --dataset mnist --arch 128 --poisoned-patch 4 \
    --trust-source provenance --untrusted-opinion 0,0,1 --tag attr-prov-vacuous
# conformity: estimated from the (contaminated) training data, no provenance.
python $REPO/attribution.py --dataset mnist --arch 128 --poisoned-patch 4 \
    --trust-source conformity
# controls: the published oracle assumption, and no knowledge at all.
python $REPO/attribution.py --dataset mnist --arch 128 --poisoned-patch 4 --trust-source oracle
python $REPO/attribution.py --dataset mnist --arch 128 --poisoned-patch 4 --trust-source trusted

# ---- 2. Single-channel poisoning controls under the same repair ------------
for M in flip patch; do
  python $REPO/attribution.py --dataset mnist --arch 128 --poisoned-patch 4 --poison-mode $M \
      --trust-source provenance --untrusted-opinion 0,1,0 --tag attr-prov-distrust
done

# ---- 3. False-alarm control: a CLEAN model, same provenance assumption -----
# The analyst distrusts the same data source, but nothing was poisoned. This
# is the run that decides whether the method cries wolf.
python $REPO/attribution.py --dataset mnist --arch 128 \
    --trust-source provenance --untrusted-opinion 0,1,0 --tag attr-prov-distrust
python $REPO/attribution.py --dataset mnist --arch 128 --trust-source trusted

# ---- 4. Label-noise audit models under attribution -------------------------
# Does attribution also sharpen the audit? Same models as Table V.
for P in 0.1 0.3 0.5; do
  python $REPO/attribution.py --dataset mnist --arch 512 --x-trust trust --y-trust vacuous \
      --noise-level $P --trust-source provenance --untrusted-opinion 0,1,0 --tag attr-prov-distrust
done
python $REPO/attribution.py --dataset mnist --arch 512 --trust-source provenance \
    --untrusted-opinion 0,1,0 --tag attr-prov-distrust

# ---- 5. Per-class output trust for every cache, old and new ---------------
python $REPO/eval_class_trust.py

# ---- 6. Inference-time effect: clean vs patched, repaired vs published -----
# Same evaluation as the paper's no-oracle experiment, now reading the
# attributed opinions and using the magnitude-weighted path.
python $REPO/eval_poisoned_conformity.py --patch-size 4 --hidden 128 --eps 0.05
python $REPO/eval_poisoned_conformity.py --patch-size 4 --hidden 128 \
    --eps attr-prov-distrust --path-mode activity
python $REPO/eval_poisoned_conformity.py --patch-size 4 --hidden 128 \
    --eps attr-prov-distrust --path-mode binary

# ---- 7. Does the magnitude-weighted path cost anything on the main tables? -
# The detection batteries rerun with the new path conditioning, scoring-only.
python $REPO/run_uq_comparison.py --dataset mnist --arch 512 --train-missing \
    --patas-path activity --out-tag actpath --seeds 0 1 2
python $REPO/run_uq_comparison.py --dataset fashion --arch 512 --train-missing \
    --patas-path activity --out-tag actpath --seeds 0 1 2
