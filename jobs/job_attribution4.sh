#!/bin/bash
#SBATCH --job-name=attr4
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
source ~/myenv/bin/activate

cd "$HOME/PaTAS"
REPO=PaTAS-Subjective-Logic-Neural-Networks-Trust-Assessment

# The inference-time runs showed the no-knowledge control (attr-trusted)
# matching the informative trust sources, so the gain over the published
# score is NOT coming from the trust information. Two cells complete the
# 2x2 that says whether it comes from attribution or from the magnitude
# weighting, which is the ablation a reviewer will ask for.
#
#   published omegas + binary path   : 0.857 / 0.827   (have)
#   published omegas + activity path : this run
#   attributed (trusted) + binary    : this run
#   attributed (trusted) + activity  : 0.916 / 0.868   (have)
python $REPO/eval_poisoned_conformity.py --patch-size 4 --hidden 128 \
    --eps 0.05 --path-mode activity
python $REPO/eval_poisoned_conformity.py --patch-size 4 --hidden 128 \
    --eps attr-trusted --path-mode binary
