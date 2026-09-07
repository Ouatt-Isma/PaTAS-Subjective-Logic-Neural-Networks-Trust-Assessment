#!/bin/bash
#SBATCH --job-name=pois-global
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

# The poison controls showed the per-class gap is asserted, not detected.
# One question remains: is the poisoning visible GLOBALLY at the scale-tied
# threshold, the way train-time label noise is in the audit? That needs the
# missing clean-trained 128 baseline at the same threshold. The poisoned
# neutral-opinion counterpart (PathSize_4nt, mean 0.6073) already exists.
python $REPO/tests/test_mnist.py --xtrust trust --ytrust trust --hidden-neurons 128 --epsilon-low auto0.5

python $REPO/eval_class_trust.py
