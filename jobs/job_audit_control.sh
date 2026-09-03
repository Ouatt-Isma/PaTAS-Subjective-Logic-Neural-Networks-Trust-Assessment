#!/bin/bash
#SBATCH --job-name=audit-ctrl
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

# Audit control: separate the effect of the ASSUMED label opinion from the
# effect of the corrupted DATA. Table V changes both at once (trusted labels
# at p=0, vacuous labels at p>0). Here the label opinion is held fixed
# across all rates, on exactly the same trained networks as Table V (the
# symlinks reuse the cached base models, so only the PaTAS mirrors are
# retrained by offline gradient replay).
R=results
link() { [ -e "$R/$1" ] || ln -s "$2" "$R/$1"; }
# fixed trusted opinion on corrupted labels -> same noisy-trained NN as the vacuous rows
for P in 0.1 0.2 0.3 0.4 0.5; do
  link "NN_Train_mnist_512_trust_trust_PathSize_None_nl$P" "NN_Train_mnist_512_trust_vacuous_PathSize_None_nl$P"
done
# fixed vacuous opinion on clean labels -> same clean-trained NN as the p=0 row
link "NN_Train_mnist_512_trust_vacuous_PathSize_None" "NN_Train_mnist_512_trust_trust_PathSize_None"

python $REPO/eval_model_trust.py --dataset mnist --arch 512 --train-missing --y-opinion trust
python $REPO/eval_model_trust.py --dataset mnist --arch 512 --train-missing --y-opinion vacuous
