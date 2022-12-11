#!/bin/bash
#SBATCH --job-name=ADNI
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/yx2105/shared/MLH/results/exp_cnn_lr0.1_0.o
#SBATCH --mem=64GB
#SBATCH --time=30:20:00
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"

conda activate myenv
cd /scratch/yx2105/shared/MLH/MML-ADNI/main

python network.py /scratch/yx2105/shared/MLH/results/ -n exp_cnn_lr0.1_0 --classifier cnn --n_classes 3 --batch_size 8 --gpu --learning_rate 0.1