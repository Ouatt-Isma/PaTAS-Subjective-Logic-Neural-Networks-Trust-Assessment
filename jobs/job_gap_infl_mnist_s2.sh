#!/bin/bash
#SBATCH --job-name=infl-m2
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

# Gap 1, seed 2. Provenance-guided pruning against what a defender with the
# same knowledge would do instead: discard the untrusted source and retrain,
# drop the most self-influential samples and retrain, or drop the same number
# at random. Eight trainings for this seed; models are cached per seed, so
# seeds are independent and run in parallel.
python $REPO/influence_baseline.py --dataset mnist --arch 128 --poisoned-patch 4 --seeds 2
