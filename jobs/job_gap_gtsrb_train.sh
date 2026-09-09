#!/bin/bash
#SBATCH --job-name=gtsrb-pois
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

# Gap 2a. GTSRB is the second dataset. It supports the same 6<->9 patch
# poisoning but has no cached poisoned model, so this trains the scenario
# that eval_repair.py reads. Required before job_gap_repair_gtsrb.sh;
# job_gap_infl_gtsrb.sh does NOT need it (it trains its own models).
python $REPO/tests/test_gtsrb_pois.py --patch-size 4 --hidden-neurons 128
