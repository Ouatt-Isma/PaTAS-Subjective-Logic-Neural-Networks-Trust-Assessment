#!/bin/bash
#SBATCH --job-name=pois-ctrl
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=08:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

module load compiler/gnu/14.2
module load devel/python/3.11.7-gnu-14.2
module load devel/cuda/12.8
source ~/myenv/bin/activate

cd "$HOME/PaTAS"
REPO=PaTAS-Subjective-Logic-Neural-Networks-Trust-Assessment

# Poisoning control, the exact counterpart of the audit control.
# The published per-class gap (clean 0.947 vs flipped 0.756) was produced
# with a trust generator that ASSERTS distrust on classes 6 and 9 for every
# poisoned sample. These runs keep the data poisoning but replace that
# generator with the plain trust/trust specs, so any surviving gap must come
# from the gradient evidence alone.
#
#   eps 0.05   : the paper's threshold, where gradients are saturated
#                (expected: no gap -> the published gap was asserted)
#   eps auto0.5: the scale-tied threshold, where the audit control showed
#                gradient evidence IS informative (the fair test)
python $REPO/tests/test_mnist_poisoned.py --patch-size 4 --no-oracle-trust --epsilon-low 0.05
python $REPO/tests/test_mnist_poisoned.py --patch-size 4 --no-oracle-trust --epsilon-low auto0.5

# Reference: the published (oracle-asserting) setup re-read at the scale-tied
# threshold, to separate "assertion" from "threshold" in the published gap.
python $REPO/tests/test_mnist_poisoned.py --patch-size 4 --epsilon-low auto0.5

# Per-class trust for every cached poisoned run, including the new ones.
python $REPO/eval_class_trust.py
