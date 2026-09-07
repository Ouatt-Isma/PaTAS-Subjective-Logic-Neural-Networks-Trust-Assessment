#!/bin/bash
#SBATCH --job-name=attr3
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=04:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

module load compiler/gnu/14.2
module load devel/python/3.11.7-gnu-14.2
module load devel/cuda/12.8
source ~/myenv/bin/activate

cd "$HOME/PaTAS"
REPO=PaTAS-Subjective-Logic-Neural-Networks-Trust-Assessment

# The inference-time comparison, which crashed in the previous two jobs:
# PTAS rejected the 'attr-*' cache label as a malformed threshold. Fixed.
# These are the runs that show what the repaired parameter trust does at
# scoring time, against the published parameter trust on the same model.
python $REPO/eval_poisoned_conformity.py --patch-size 4 --hidden 128 \
    --eps attr-prov-distrust --path-mode binary
python $REPO/eval_poisoned_conformity.py --patch-size 4 --hidden 128 \
    --eps attr-prov-distrust --path-mode activity
python $REPO/eval_poisoned_conformity.py --patch-size 4 --hidden 128 \
    --eps attr-conformity --path-mode activity
python $REPO/eval_poisoned_conformity.py --patch-size 4 --hidden 128 \
    --eps attr-trusted --path-mode activity
