#!/bin/bash
#SBATCH --job-name=rev3-conv
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=20:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

module load compiler/gnu/14.2
module load devel/python/3.11.7-gnu-14.2
module load devel/cuda/12.8
source ~/myenv/bin/activate

cd "$HOME/PaTAS"
REPO=PaTAS-Subjective-Logic-Neural-Networks-Trust-Assessment

# Review round 3, conv part. All model caches exist -> scoring-only reruns
# (per-sample conv IPTA is the slow step; two extra mixed conditions per
# battery roughly double it, hence the generous walltime).
#
# Major 3a/3b on the CNN batteries: pixel-space baselines + hard corruptions.
python $REPO/run_uq_comparison.py --dataset mnistconv   --epochs 30 --train-missing --seeds 0 1 2 --hard-corruptions
python $REPO/run_uq_comparison.py --dataset fashionconv --epochs 30 --train-missing --seeds 0 1 2 --hard-corruptions
python $REPO/run_uq_comparison.py --dataset cifar10     --epochs 60 --train-missing --seeds 0 1 2 --hard-corruptions

# Major 5 follow-up (Fashion-CNN reverse-OOD inversion): rescore
# fashionconv with per-layer mean-1 activity normalization, which removes
# the global activity level (the bias-row domination mechanism) from the
# conv IPTA and keeps only the relative participation pattern. Separate
# output dir via --out-tag; single evaluation seed is enough for the
# diagnostic.
python $REPO/run_uq_comparison.py --dataset fashionconv --epochs 30 --train-missing --conv-activity-norm --out-tag actnorm
python $REPO/run_uq_comparison.py --dataset mnistconv   --epochs 30 --train-missing --conv-activity-norm --out-tag actnorm
