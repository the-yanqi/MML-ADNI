#!/bin/bash
#SBATCH --job-name=ADNI
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/yx2105/shared/MLH/results/training.o
#SBATCH --mem=64GB
#SBATCH --time=10:20:00
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"

conda activate myenv
cd /scratch/yx2105/shared/MLH/MML-ADNI/main

python network.py /scratch/yx2105/shared/MLH/results/ -n baseline_vgg_lr0.1 --classifier vgg --n_classes 3 --batch_size 8 --gpu --learning_rate 0.1