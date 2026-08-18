#!/bin/bash
#SBATCH --job-name=conf-fit
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:45:00
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

# Conformity-fit provenance experiment (three configs, same patch-4 cache,
# scoring-only). Must run AFTER rev-remainder: its seed loop rewrites the
# patch-4 caches, and all three configs below must score the same final
# model for an internally consistent comparison. Submit with
#   sbatch --dependency=afterany:<rev-remainder-jobid> jobs/job_conformity_fit.sh
# or simply after it has finished.
#
#   1. clean fit, moment statistics      (reference, matches the paper's W7)
#   2. poisoned fit, moment statistics   (deployment-honest; expected to
#      degrade: the patch contaminates the very witnesses that detect it)
#   3. poisoned fit, robust statistics   (median/MAD; expected recovery,
#      no clean feature source assumed anywhere in the pipeline)
python $REPO/eval_poisoned_conformity.py --patch-size 4
python $REPO/eval_poisoned_conformity.py --patch-size 4 --fit-on poisoned
python $REPO/eval_poisoned_conformity.py --patch-size 4 --fit-on poisoned --stats robust
