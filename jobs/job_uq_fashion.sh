#!/bin/bash
#SBATCH --job-name=uq-fashion
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

if [ ! -d ~/myenv ]; then
    python -m venv ~/myenv
fi
source ~/myenv/bin/activate

cd "$HOME/PaTAS"
REPO=PaTAS-Subjective-Logic-Neural-Networks-Trust-Assessment

# Second primary use case: Fashion-MNIST (registered domain — the
# applicability diagnostic predicts informative input trust), symmetric OOD
# partner of MNIST (OOD = MNIST test through Fashion's pipeline).  First
# seed trains base + PaTAS + MC/EDL for both conditions; seeds 1-2 reuse
# every cache.  Own port (5271) and caches — safe alongside other jobs.
python $REPO/run_uq_comparison.py --dataset fashion --arch 512 --train-missing --seeds 0 1 2

# A-priori applicability table across all four datasets (fit-only, no
# training; uses the warmed data/ caches).
python $REPO/itm_diagnostics.py
