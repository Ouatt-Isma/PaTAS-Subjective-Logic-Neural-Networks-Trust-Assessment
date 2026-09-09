#!/bin/bash
#SBATCH --job-name=leak05
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
source ~/myenv/bin/activate

cd "$HOME/PaTAS"
REPO=PaTAS-Subjective-Logic-Neural-Networks-Trust-Assessment

# Gap 3, leakage 0.5. The attacker also compromises part of the audited
# source, so the provenance labels are partly wrong: 0.5 of the poison sits
# in the VERIFIED source. Each leakage level has its own training data and
# therefore its own model cache, so these are independent.
python $REPO/eval_multisource.py --arch 128 --epochs 20 --seeds 0 1 \
    --unknown-classes 4 7 --poison-leak 0.5
