#!/bin/bash
#SBATCH --job-name=audit-ctrl2
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
source ~/myenv/bin/activate

cd "$HOME/PaTAS"
REPO=PaTAS-Subjective-Logic-Neural-Networks-Trust-Assessment

R=results
link() { [ -e "$R/$1" ] || ln -s "$2" "$R/$1"; }
for P in 0.1 0.2 0.3 0.4 0.5; do
  link "NN_Train_mnist_512_trust_trust_PathSize_None_nl$P" "NN_Train_mnist_512_trust_vacuous_PathSize_None_nl$P"
done
link "NN_Train_mnist_512_trust_vacuous_PathSize_None" "NN_Train_mnist_512_trust_trust_PathSize_None"

# (1) The vacuous-opinion control again, in case the first job did not
#     reach it (its output dir was absent from the rsync).
python $REPO/eval_model_trust.py --dataset mnist --arch 512 --train-missing --y-opinion vacuous

# (2) Trusted label opinion at every rate, but with the scale-tied
#     threshold: at eps=0.05 nearly every gradient of the 512-wide net is
#     'small', so gradient evidence cannot register corruption. A threshold
#     tied to each layer's own gradient scale is the fair test of whether
#     gradient evidence carries a corruption signal at all.
python $REPO/eval_model_trust.py --dataset mnist --arch 512 --train-missing --y-opinion trust --eps auto0.5
python $REPO/eval_model_trust.py --dataset mnist --arch 512 --train-missing --y-opinion trust --eps auto0.2
