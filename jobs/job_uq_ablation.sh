#!/bin/bash
#SBATCH --job-name=uq-ablation
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

# Method-section ablations on MNIST 512 (all scoring-only, caches hit;
# --out-tag keeps each variant's outputs away from the main tables):
#   1. constant fully-trusted input  (no per-sample input information)
#   2. unweighted evidence (alpha=0) (every feature an equal witness)
#   3. propagated combination        (input opinions through the IPTA
#                                     feedforward instead of serial SL
#                                     discounting)
python $REPO/run_uq_comparison.py --dataset mnist --arch 512 --train-missing --patas-input-trust constant --out-tag abl_constant
python $REPO/run_uq_comparison.py --dataset mnist --arch 512 --train-missing --itm-alpha 0 --out-tag abl_unweighted
python $REPO/run_uq_comparison.py --dataset mnist --arch 512 --train-missing --patas-score propagated --out-tag abl_propagated

# Depth row: 512-256 rerun under the current scoring (caches exist).
python $REPO/run_uq_comparison.py --dataset mnist --arch 512 256 --train-missing
