#!/bin/bash

# Go to the script dir
cd $HOME/Transparency/Transparency

# Activate your environment
source activate maka_paper

# Line required by the repo
export PYTHONPATH=$HOME/Transparency

# general parameters
output_path=./experiments
diversity_weight=0.5

# Use this parameter for testing, otherwise don't specify so that the predefined number of epochs is used 
# n_epochs=1

# Change the number of runs to higher to measure perfomance standard deviation
num_runs=1

# I put small datasets first so if there is an error in the script, it will occur faster
qa_datasets="snli qqp"
# Skipped QA datasets: cnn
models="vanilla_lstm ortho_lstm diversity_lstm"


# train and evaluate other NLP datasets (NLI, QA)
for dataset_name in $qa_datasets; do
    echo "### Dataset "${dataset_name}
    for model_name in $models; do
        echo "--- Model "${model_name}
        for (( i=0; i<num_runs; ++i)); do 
            echo "*** Run "${i}
            python train_and_run_experiments_qa.py --dataset ${dataset_name} --data_dir . --output_dir ${output_path} --encoder ${model_name} --diversity ${diversity_weight} --attention tanh
            #  --n_iter ${n_epochs}
        done
    done
done
