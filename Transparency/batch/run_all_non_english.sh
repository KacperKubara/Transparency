#!/bin/bash
#SBATCH --time=30:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kjk1u17@soton.ac.uk

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
# English
datasets="cls_en cls_de cls_fr cls_jp"
models="vanilla_lstm ortho_lstm diversity_lstm"
diversity_weight=0.5
n_epochs=1

for data in $datasets:
do 
    output_path="./experiments/${data}"
    for model in $models:
    do
        echo $data $model
        echo python train_and_run_experiments_bc.py --dataset ${data} --data_dir . --output_dir ${output_path} --encoder ${model} --diversity ${diversity_weight} --n_iter ${n_epochs}

    done
done