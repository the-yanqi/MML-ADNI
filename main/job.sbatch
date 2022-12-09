#!/bin/bash
#SBATCH --job-name=ADNI
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=model_output.out
#SBATCH --mem=64GB
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:4

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate deep_ADNI
cd /scratch/di2078/shared/MLH/data/deep_learning_ADNI/main
chmod +x train.sh
./train.sh > output
