#!/bin/bash
#SBATCH --job-name=rev3-ctrl
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=12:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

module load compiler/gnu/14.2
module load devel/python/3.11.7-gnu-14.2
module load devel/cuda/12.8
source ~/myenv/bin/activate

cd "$HOME/PaTAS"
REPO=PaTAS-Subjective-Logic-Neural-Networks-Trust-Assessment

# Review round 3, controls and the scale-tied epsilon.
#
# Major 2: scale-tied threshold eps_layer = c * median(|delta|) (c = 0.5),
# estimated per layer over the first revision batches and frozen after.
# Validation 1: the audit sweep rerun at eps "auto0.5" (retrains the PaTAS
# mirror per rate under the new rule; NN caches are shared). The tripwire
# margin should be preserved without any per-task eps tuning.
python $REPO/eval_model_trust.py --dataset mnist --arch 512 --eps auto0.5 --train-missing

# Validation 2: cancer grid corners at the same rule (different task,
# different gradient scale; the trust ordering across x/y trust scenarios
# should match the fixed-eps grid).
python $REPO/tests/test_cancer.py --xtrust trust    --ytrust trust    --epsilon auto0.5
python $REPO/tests/test_cancer.py --xtrust trust    --ytrust distrust --epsilon auto0.5
python $REPO/tests/test_cancer.py --xtrust distrust --ytrust trust    --epsilon auto0.5

# Minor (flip/patch conflation): single-channel poisoning controls at the
# paper's patch size. 'flip' = label flip only (pure pair label noise),
# 'patch' = trigger patch only (benign spurious feature). Caches get a
# pm<mode> suffix, so the full-backdoor PathSize_4 run is untouched.
python $REPO/tests/test_mnist_poisoned.py --patch-size 4 --poison-mode flip
python $REPO/tests/test_mnist_poisoned.py --patch-size 4 --poison-mode patch

# Minor (digit-9 trust + clean-class variance): per-class output trust for
# every cached poisoned run (all patch sizes + the two controls above),
# with the clean-class band and z-scores for both flipped classes.
python $REPO/eval_class_trust.py
