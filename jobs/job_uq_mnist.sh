#!/bin/bash
#SBATCH --job-name=uq-mnist
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=02:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

module load compiler/gnu/14.2
module load devel/python/3.11.7-gnu-14.2
module load devel/cuda/12.8

if [ ! -d ~/myenv ]; then
    python -m venv ~/myenv
fi
source ~/myenv/bin/activate

# Caches (results/, data/) live in ~/PaTAS — run from there regardless of
# where sbatch was invoked.
cd "$HOME/PaTAS"
REPO=PaTAS-Subjective-Logic-Neural-Networks-Trust-Assessment

# Main paper tables: MNIST 512, clean-trained + label-noise-trained,
# 3 evaluation seeds aggregated (mean±std), mixed batch + OOD included.
# All model caches exist -> scoring-only.
python $REPO/run_uq_comparison.py --dataset mnist --arch 512 --train-missing --seeds 0 1 2

# Training-quality audit (label-flip-rate sweep; caches exist).
python $REPO/eval_model_trust.py --dataset mnist --arch 512 --train-missing
