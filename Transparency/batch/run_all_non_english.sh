#!/bin/bash
#SBATCH --time=30:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kjk1u17@soton.ac.uk

# Go to the script dir
cd $HOME/Transparency/Transparency

# Activate your environment
source activate maka_paper

# Line required by the repo
export PYTHONPATH=$HOME/Transparency

# Run the script and train the model
datasets="cls_en cls_de cls_fr cls_jp"
models="vanilla_lstm ortho_lstm diversity_lstm"
diversity_weight=0.5
n_epochs=1

for i in {0..4}
do
    echo "RUN NUMBER: $i"
    echo "============================================================\n\n\n\n"
    for data in $datasets
    do 
        output_path="./experiments/${data}"
        for model in $models
        do
            echo $data $model
            python train_and_run_experiments_bc.py --dataset ${data} --data_dir . --output_dir ${output_path} --encoder ${model} --diversity ${diversity_weight} --n_iter ${n_epochs}

        done
    done
done