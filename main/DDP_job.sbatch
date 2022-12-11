#!/bin/bash
#SBATCH --job-name=ADNI
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/yx2105/shared/MLH/results/ddp_cnn_lr0.1_0.o
#SBATCH --mem=128GB
#SBATCH --time=20:20:00
#SBATCH --gres=gpu:4

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"

conda activate myenv
cd /scratch/yx2105/shared/MLH/MML-ADNI/main

python DDP_network.py /scratch/yx2105/shared/MLH/results/ --name ddp_cnn_lr0.1_0 --classifier cnn --n_classes 3 --batch_size 8 --learning_rate 0.1