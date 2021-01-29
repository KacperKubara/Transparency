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


num_runs=1
bc_datasets="sst imdb 20News_sports tweet"
qa_datasets="snli qqp babi_1 babi_2 babi_3"
models="vanilla_lstm ortho_lstm diversity_lstm"

# train and evaluate binary classification datasets
for dataset_name in $bc_datasets; do
    echo "### Dataset "${dataset_name}
    for model_name in $models; do
        echo "--- Model "${model_name}
        for (( i=0; i<num_runs; ++i)); do 
            echo "*** Run "${i}
            python train_and_run_experiments_bc.py --dataset ${dataset_name} --data_dir . --output_dir ${output_path} --encoder ${model_name} --diversity ${diversity_weight}
        done
    done
done

# train and evaluate other NLP datasets (NLI, QA)
for dataset_name in $qa_datasets; do
    echo "### Dataset "${dataset_name}
    for model_name in $models; do
        echo "--- Model "${model_name}
        for (( i=0; i<num_runs; ++i)); do 
            echo "*** Run "${i}
            python train_and_run_experiments_qa.py --dataset ${dataset_name} --data_dir . --output_dir ${output_path} --encoder ${model_name} --diversity ${diversity_weight} --attention tanh
        done
    done
done
