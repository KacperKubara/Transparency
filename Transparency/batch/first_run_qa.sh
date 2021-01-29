#!/bin/bash

# Go to the script dir
cd $HOME/Transparency/Transparency

# Activate your environment
source activate maka_paper

# Line required by the repo
export PYTHONPATH=$HOME/Transparency

# Run the script and train the model
dataset_name=snli
model_name=diversity_lstm
output_path=./experiments
diversity_weight=0.5
attention=tanh
n_epochs=5
python train_and_run_experiments_qa.py --dataset ${dataset_name} --data_dir . --output_dir ${output_path} --encoder ${model_name} --diversity ${diversity_weight} --n_iter ${n_epochs} --attention ${attention}

