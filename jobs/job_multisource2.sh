#!/bin/bash
#SBATCH --job-name=multisrc2
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

# The three scenarios previously overwrote one another's summary; they now
# write per-scenario files. The trained models are cached and shared (the
# training data does not depend on the opinion assignment), so this is an
# attribution replay only. Each run also reports the threshold operating
# point, where the defender never picks a pruning budget.
python $REPO/eval_multisource.py --arch 128 --epochs 20 --seeds 0 1 2 --unknown-classes 4 7
python $REPO/eval_multisource.py --arch 128 --epochs 20 --seeds 0 1 2 --unknown-classes 4
python $REPO/eval_multisource.py --arch 128 --epochs 20 --seeds 0 1 2 --unknown-classes 4 7 1

# How large may the compromised source be before the defence stops working?
for CF in 0.10 0.40; do
  python $REPO/eval_multisource.py --arch 128 --epochs 20 --seeds 0 \
      --unknown-classes 4 7 --compromised-frac $CF
done
