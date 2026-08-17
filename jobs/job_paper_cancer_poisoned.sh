#!/bin/bash
#SBATCH --job-name=paper-cancer-pois
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

# Regenerates the paper's Experiment 1 (Breast Cancer, Table I) and
# Experiment 3 (Poisoned MNIST, Tables III/IV + figures) under the updated
# code.  The poisoned test has no --force flag, so its caches are archived
# first; the patterns match ONLY the PathSize_<digit> poisoned dirs — the
# UQ battery's caches (mnist_512*, gtsrb_*, PathSize_None) are untouched.
ARCHIVE="results/OLD_legacy_${SLURM_JOB_ID}"
mkdir -p "$ARCHIVE"
for d in results/NN_Train_mnist_*_PathSize_[0-9]* \
         results/PTAS_Eval_mnist_*_PathSize_[0-9]*; do
    [ -d "$d" ] && mv "$d" "$ARCHIVE/"
done

python $REPO/tests/test_cancer.py --force-retrain
python $REPO/tests/test_mnist_poisoned.py
