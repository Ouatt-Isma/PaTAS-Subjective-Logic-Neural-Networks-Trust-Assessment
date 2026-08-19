#!/bin/bash
#SBATCH --job-name=conv-cifar
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=10:00:00
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

# CIFAR-10 conv battery with per-sample conv IPTA (activity-weighted paths)
# and the upgraded training recipe (momentum 0.9, wd 5e-4, augmentation,
# stepped LR, 60 epochs). Old CIFAR caches were trained with the legacy
# plain-SGD recipe and must not be reused: archive them first.
ARCHIVE="results/OLD_cifar_prerecipe_${SLURM_JOB_ID}"
mkdir -p "$ARCHIVE"
for d in results/NN_Train_cifar10_resnet-lite_* \
         results/PTAS_Eval_cifar10_resnet-lite_* \
         results/UQ_Train_cifar10_* \
         results/UQ_Compare_cifar10*; do
    [ -d "$d" ] && mv "$d" "$ARCHIVE/"
done

# First seed trains base + PaTAS mirrors + MC/EDL for both training
# conditions; seeds 1-2 reuse every cache (scoring-only). OOD is auto-none
# for CIFAR; the corruption-detection and conformity rows document the
# predicted applicability boundary on unregistered imagery, while the
# label-noise condition carries the architecture-transferable model-side
# result (parameter trust under corrupted training).
python $REPO/run_uq_comparison.py --dataset cifar10 --epochs 60 --train-missing --seeds 0 1 2

# Rerun the training-latency benchmark at the enlarged scale.
python $REPO/latency_eval.py results/latency_review
