#!/bin/bash
#SBATCH --job-name=attr2
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=06:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

module load compiler/gnu/14.2
module load devel/python/3.11.7-gnu-14.2
module load devel/cuda/12.8
source ~/myenv/bin/activate

cd "$HOME/PaTAS"
REPO=PaTAS-Subjective-Logic-Neural-Networks-Trust-Assessment

# ===========================================================================
# Follow-up to job_attribution.sh. Two things the first run left open.
#
# (1) The conformity trust source localised the trigger at AUROC 0.998 with
#     NO provenance knowledge, which is a stronger claim than expected. It
#     is only a detector if a CLEAN model under the same source produces no
#     low-trust features. That control was not run. It is the decisive one.
#
# (2) The three eval_poisoned_conformity variants wrote to one directory and
#     overwrote each other; output dirs are now separated by eps and path.
# ===========================================================================

# ---- (1) clean-model controls, the false-alarm side of every trust source --
python $REPO/attribution.py --dataset mnist --arch 128 --trust-source conformity
python $REPO/attribution.py --dataset mnist --arch 128 \
    --trust-source provenance --untrusted-opinion 0,0,1 --tag attr-prov-vacuous

# Same controls on the wider audit architecture, where the clean model has a
# different activation profile.
python $REPO/attribution.py --dataset mnist --arch 512 --trust-source conformity
python $REPO/attribution.py --dataset mnist --arch 512 --trust-source trusted

# Fashion-MNIST: does the conformity source stay quiet on a clean model of a
# domain whose pixels are far less constant?
python $REPO/attribution.py --dataset fashion --arch 512 --trust-source conformity

# ---- (2) inference-time comparison, now into separate directories ---------
python $REPO/eval_poisoned_conformity.py --patch-size 4 --hidden 128 --eps 0.05
python $REPO/eval_poisoned_conformity.py --patch-size 4 --hidden 128 \
    --eps attr-prov-distrust --path-mode binary
python $REPO/eval_poisoned_conformity.py --patch-size 4 --hidden 128 \
    --eps attr-prov-distrust --path-mode activity
python $REPO/eval_poisoned_conformity.py --patch-size 4 --hidden 128 \
    --eps attr-conformity --path-mode activity

python $REPO/eval_class_trust.py
