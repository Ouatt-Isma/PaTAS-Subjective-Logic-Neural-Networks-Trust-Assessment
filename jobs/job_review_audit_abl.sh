#!/bin/bash
#SBATCH --job-name=rev-audit-abl
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

# Reviewer W3 + Q3: audit with training-data baselines (disagreement
# fraction, mean training loss) and the rate-informed label-opinion sweep
# that turns the tripwire into a graded severity signal.
python $REPO/eval_model_trust.py --dataset mnist --arch 512 --train-missing --rate-informed

# Reviewer W5: fusion-operator sensitivity (averaging vs cumulative vs
# weighted; retrains the PaTAS mirror per operator at the default arch).
python $REPO/ablation_fusion.py --dataset mnist --epochs 20

# Reviewer W6: epsilon sensitivity grid.
python $REPO/ablation_epsilon.py --dataset mnist --epochs 20

# Reviewer W6 confound check: width sweep at two additional epsilon values
# (vacuous/vacuous scenarios; ports 5041+).
for EPS in 0.02 0.1; do
  for H in 16 32 64 128; do
    python $REPO/tests/test_mnist.py --xtrust vacuous --ytrust vacuous \
      --hidden-neurons $H --epsilon-low $EPS
  done
done

# Reviewer W8: measured runtime overhead vs the analytical 3x estimate.
python $REPO/latency_eval.py results/latency_review
