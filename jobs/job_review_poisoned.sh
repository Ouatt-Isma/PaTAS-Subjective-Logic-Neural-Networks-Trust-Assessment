#!/bin/bash
#SBATCH --job-name=rev-poisoned
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

# Reviewer W7: conformity opinions through the poisoned model's IPTA,
# no oracle patch knowledge (uses the existing patch-4 caches).
python $REPO/eval_poisoned_conformity.py --patch-size 4

# Reviewer W8 / Table IV: restore the multi-seed protocol. Ten fresh
# trainings of the patch-4 scenario (unseeded init = a new seed per run);
# the collector aggregates the printed Table 2a blocks from THIS job log.
for i in $(seq 1 10); do
  echo "===== TABLE4 SEED RUN $i / 10 ====="
  python $REPO/tests/test_mnist_poisoned.py --patch-size 4 --force-retrain
done
python $REPO/collect_table4_seeds.py job_${SLURM_JOB_ID}.out
