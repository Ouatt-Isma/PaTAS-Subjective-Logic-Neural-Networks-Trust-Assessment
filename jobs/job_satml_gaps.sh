#!/bin/bash
#SBATCH --job-name=satml-gaps
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=12:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

module load compiler/gnu/14.2
module load devel/python/3.11.7-gnu-14.2
module load devel/cuda/12.8
source ~/myenv/bin/activate

cd "$HOME/PaTAS"
REPO=PaTAS-Subjective-Logic-Neural-Networks-Trust-Assessment

# ===========================================================================
# The three submission-blocking gaps for SaTML.
# ===========================================================================

# (1) What a practitioner would actually do, and the method a reviewer will
#     assume we should have used. Every arm gets the same knowledge: which
#     source is untrusted, nothing about the trigger. Retraining arms cost a
#     full training run each; ours does not retrain at all.
python $REPO/influence_baseline.py --dataset mnist --arch 128 --poisoned-patch 4 --seeds 0 1 2

# (2) A second dataset. GTSRB supports the same poisoning and has no cached
#     poisoned model, so this trains one, then runs the removal comparison.
python $REPO/tests/test_mnist_poisoned.py --patch-size 4 --epochs 20 2>/dev/null || true
python $REPO/influence_baseline.py --dataset gtsrb --arch 128 --poisoned-patch 4 --seeds 0 1
python $REPO/eval_repair.py --dataset gtsrb --arch 128 --poisoned-patch 4

# (3) Adaptive adversary: the attacker also compromises part of the audited
#     source, so the provenance labels are partly wrong. How much leakage
#     does the defence tolerate before it fails?
for LEAK in 0.1 0.25 0.5; do
  python $REPO/eval_multisource.py --arch 128 --epochs 20 --seeds 0 1 \
      --unknown-classes 4 7 --poison-leak $LEAK
done
