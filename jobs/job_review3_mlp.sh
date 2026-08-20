#!/bin/bash
#SBATCH --job-name=rev3-mlp
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

# Review round 3, MLP part. All model caches exist -> scoring-only reruns.
#
# Major 3a/3b: the three MLP batteries rerun with
#   * raw-pixel Mahalanobis + pixel-kNN baselines (input-space
#     counterparts of the feature-space detectors, on by default), and
#   * --hard-corruptions: two extra paired mixed batches (random +/-2px
#     translation = marginal-preserving; low-amplitude blended center
#     trigger = adaptive adversary), stress-testing the marginal
#     conformity model where it is blind by construction.
python $REPO/run_uq_comparison.py --dataset mnist   --arch 512 --train-missing --seeds 0 1 2 --hard-corruptions
python $REPO/run_uq_comparison.py --dataset fashion --arch 512 --train-missing --seeds 0 1 2 --hard-corruptions
python $REPO/run_uq_comparison.py --dataset gtsrb              --train-missing --seeds 0 1 2 --hard-corruptions

# Major 1: dead-unit confound check on the cached audit sweep (activation
# frequency vs unit trust; activity-weighted feedforward audit).
python $REPO/eval_dead_units.py --dataset mnist --arch 512 --train-missing
