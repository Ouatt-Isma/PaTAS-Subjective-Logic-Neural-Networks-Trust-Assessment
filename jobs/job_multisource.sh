#!/bin/bash
#SBATCH --job-name=multisrc
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

# The experiment that decides whether this is a subjective-logic paper.
# With one untrusted source the opinion and a scalar influence ratio are
# provably equivalent. With sources carrying different amounts of evidence
# (verified / unknown / compromised) they diverge. Locally, at a 10% pruning
# budget the opinion drove attack success to 0.05% while the scalar left it
# at 85.6%. Three seeds here, on the standardised pipeline.
python $REPO/eval_multisource.py --arch 128 --epochs 20 --seeds 0 1 2

# Sensitivity: how large must the unknown source be for the gap to appear?
for UC in "4" "4 7 1" ; do
  python $REPO/eval_multisource.py --arch 128 --epochs 20 --seeds 0 \
      --unknown-classes $UC
done
