#!/bin/bash
#SBATCH --job-name=repair-g
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=01:30:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

module load compiler/gnu/14.2
module load devel/python/3.11.7-gnu-14.2
module load devel/cuda/12.8
source ~/myenv/bin/activate

cd "$HOME/PaTAS"
REPO=PaTAS-Subjective-Logic-Neural-Networks-Trust-Assessment

# Gap 2c. The removal sweep on GTSRB, with fine-pruning and the random
# control. Reads the scenario cache, so run job_gap_gtsrb_train.sh first.
python $REPO/eval_repair.py --dataset gtsrb --arch 128 --poisoned-patch 4
