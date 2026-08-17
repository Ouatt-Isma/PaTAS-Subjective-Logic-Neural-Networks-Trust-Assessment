#!/bin/bash
#SBATCH --job-name=uq-gtsrb
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

if [ ! -d ~/myenv ]; then
    python -m venv ~/myenv
fi
source ~/myenv/bin/activate

cd "$HOME/PaTAS"
REPO=PaTAS-Subjective-Logic-Neural-Networks-Trust-Assessment

# Second dataset: GTSRB (43 classes), clean-trained + label-noise-trained,
# 3 evaluation seeds, OOD = grayscale CIFAR-10.  The FIRST seed trains the
# base NN + PaTAS mirrors + MC/EDL for both conditions (hence the 6 h
# limit); seeds 1-2 then reuse every cache and are scoring-only.
python $REPO/run_uq_comparison.py --dataset gtsrb --train-missing --seeds 0 1 2
