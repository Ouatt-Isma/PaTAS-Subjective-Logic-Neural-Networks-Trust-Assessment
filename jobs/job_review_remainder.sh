#!/bin/bash
#SBATCH --job-name=rev-remainder
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=08:00:00
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

# Remainder of the timed-out rev-audit job plus the Table IV seed redo.
# Cheap items first so another timeout can only cost the tail.

# (1) Reviewer W8: measured runtime overhead vs the analytical 3x estimate.
python $REPO/latency_eval.py results/latency_review

# (2) Reviewer W6 confound check: width sweep at two additional epsilon
#     values (vacuous/vacuous scenarios; ports 5041+).
for EPS in 0.02 0.1; do
  for H in 16 32 64 128; do
    python $REPO/tests/test_mnist.py --xtrust vacuous --ytrust vacuous \
      --hidden-neurons $H --epsilon-low $EPS
  done
done

# (3) Table IV multi-seed redo. The previous loop produced ten IDENTICAL
#     runs: primaryNN seeded np/torch to 42 at import, so --force-retrain
#     retrained from bit-identical initialization every time. PATAS_SEED
#     (new) varies the init per run; the collector aggregates the Table 2a
#     blocks from THIS job's log (it can also be run locally on the synced
#     .out if this job is interrupted).
for i in $(seq 1 10); do
  echo "===== TABLE4 SEED RUN $i / 10 ====="
  PATAS_SEED=$i python $REPO/tests/test_mnist_poisoned.py --patch-size 4 --force-retrain
done
python $REPO/collect_table4_seeds.py job_${SLURM_JOB_ID}.out
