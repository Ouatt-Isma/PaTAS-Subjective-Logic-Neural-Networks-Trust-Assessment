#!/bin/bash
#SBATCH --job-name=abl-const
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
source ~/myenv/bin/activate

cd "$HOME/PaTAS"
REPO=PaTAS-Subjective-Logic-Neural-Networks-Trust-Assessment

# Table IV ablation row ("w/o input trust"): the PaTAS score with the input
# opinion replaced by the constant fully trusted opinion. Previously a
# single-seed MNIST-only run; this gives three seeds (for the +/- spread) and
# adds the Fashion-MNIST battery, which fills the two dashed cells (F-MNIST
# corruption detection, OOD F->M). Scoring-only: every model cache exists.
python $REPO/run_uq_comparison.py --dataset mnist   --arch 512 --train-missing --patas-input-trust constant --out-tag abl_constant --seeds 0 1 2 --hard-corruptions
python $REPO/run_uq_comparison.py --dataset fashion --arch 512 --train-missing --patas-input-trust constant --out-tag abl_constant --seeds 0 1 2 --hard-corruptions
