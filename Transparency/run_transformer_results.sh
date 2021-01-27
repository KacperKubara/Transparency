#!/bin/bash

#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=TransformerExperiments
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=10:00:00
#SBATCH --mem=32000M

module purge
module load 2019
module load Python/3.7.5-foss-2019b
module load CUDA/10.1.243
module load cuDNN/7.6.5.32-CUDA-10.1.243
module load NCCL/2.5.6-CUDA-10.1.243
module load Anaconda3/2018.12

# Go to the script dir
cd $HOME/Transparency/Transparency

# Activate your environment
source activate fact_paper

# Line required by the repo
export PYTHONPATH=$HOME/Transparency

# Run the script and train the model

python transformer_results.py --num_runs 5