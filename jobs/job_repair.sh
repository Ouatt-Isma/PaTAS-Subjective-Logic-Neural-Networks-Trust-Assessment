#!/bin/bash
#SBATCH --job-name=repair
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=04:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

module load compiler/gnu/14.2
module load devel/python/3.11.7-gnu-14.2
module load devel/cuda/12.8
source ~/myenv/bin/activate

cd "$HOME/PaTAS"
REPO=PaTAS-Subjective-Logic-Neural-Networks-Trust-Assessment

# Headline experiment for the SaTML submission: provenance-guided removal.
# The defender knows only that one data source is untrusted. Parameters whose
# attributed trust is lowest are removed; we measure the implanted behaviour
# and clean accuracy against three baselines given the same knowledge.
#
# Locally (unstandardised probe) this drove attack success from 98.9% to 0.1%
# at a 10% budget for 0.02 points of clean accuracy, while a random control
# stayed at 98% and fine-pruning stayed above 96%. This validates it on the
# real standardised pipeline, where localisation was sharper (16 features
# flagged, all trigger).
python $REPO/eval_repair.py --dataset mnist --arch 128 --poisoned-patch 4

# Sanity: the same procedure on models where there is nothing to remove.
python $REPO/eval_repair.py --dataset mnist --arch 128 --poisoned-patch 4 --poison-mode flip
python $REPO/eval_repair.py --dataset mnist --arch 128 --poisoned-patch 4 --poison-mode patch

# Does the trigger size change the picture?
for P in 1 10; do
  python $REPO/eval_repair.py --dataset mnist --arch 128 --poisoned-patch $P
done
