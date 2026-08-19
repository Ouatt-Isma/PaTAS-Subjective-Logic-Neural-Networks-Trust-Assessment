#!/bin/bash
#SBATCH --job-name=conv-mnf
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=08:00:00
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

# CNN batteries on the registered domains: the full per-sample conv-IPTA
# comparison on an architecture defensible for the domain (~99% MNIST),
# where the diagnostics predict input trust IS informative. Old
# mnist-resnet caches (legacy recipe/width) must not be reused.
ARCHIVE="results/OLD_convmnf_prerecipe_${SLURM_JOB_ID}"
mkdir -p "$ARCHIVE"
for d in results/NN_Train_mnist_resnet-lite_* \
         results/PTAS_Eval_mnist_resnet-lite_* \
         results/NN_Train_fashion_resnet-lite_* \
         results/PTAS_Eval_fashion_resnet-lite_* \
         results/UQ_Train_mnistconv_* results/UQ_Train_fashionconv_* \
         results/UQ_Compare_mnistconv* results/UQ_Compare_fashionconv*; do
    [ -d "$d" ] && mv "$d" "$ARCHIVE/"
done

# Smoke first (2 epochs, 300 samples): validates the conv-IPTA scorer
# end to end in ~10 min before committing hours.
python $REPO/run_uq_comparison.py --dataset mnistconv --quick --train-missing
for d in results/NN_Train_mnist_resnet-lite_* results/PTAS_Eval_mnist_resnet-lite_* \
         results/UQ_Train_mnistconv_* results/UQ_Compare_mnistconv*; do
    [ -d "$d" ] && mv "$d" "$ARCHIVE/"      # sweep smoke caches away
done

python $REPO/run_uq_comparison.py --dataset mnistconv   --epochs 30 --train-missing --seeds 0 1 2
python $REPO/run_uq_comparison.py --dataset fashionconv --epochs 30 --train-missing --seeds 0 1 2

# Reviewer W8: training-latency benchmark, now actually executed ([B]).
python $REPO/latency_eval.py results/latency_review
