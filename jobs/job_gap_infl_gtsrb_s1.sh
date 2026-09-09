#!/bin/bash
#SBATCH --job-name=infl-g1
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

# Gap 2b, seed 1. Same comparison as gap 1, on the second dataset. Trains
# its own models, so it does not wait on job_gap_gtsrb_train.sh.
python $REPO/influence_baseline.py --dataset gtsrb --arch 128 --poisoned-patch 4 --seeds 1
