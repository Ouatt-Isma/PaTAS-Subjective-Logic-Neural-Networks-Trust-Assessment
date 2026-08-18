#!/bin/bash
#SBATCH --job-name=rev-baselines
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=03:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

module load compiler/gnu/14.2
module load devel/python/3.11.7-gnu-14.2
module load devel/cuda/12.8
source ~/myenv/bin/activate

cd "$HOME/PaTAS"
REPO=PaTAS-Subjective-Logic-Neural-Networks-Trust-Assessment

# Reviewer W2: rerun the three batteries scoring-only (all model caches
# hit) with the new attribution rows: conformity-only (no propagation),
# Mahalanobis, kNN, energy, plus the P_input/P_model/P_serial/score
# decomposition on the OOD and mixed conditions.
python $REPO/run_uq_comparison.py --dataset mnist   --arch 512 --train-missing --seeds 0 1 2
python $REPO/run_uq_comparison.py --dataset fashion --arch 512 --train-missing --seeds 0 1 2
python $REPO/run_uq_comparison.py --dataset gtsrb              --train-missing --seeds 0 1 2
