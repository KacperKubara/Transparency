#!/bin/bash

# Go to the script dir
cd $HOME/Transparency/Transparency

# Activate your environment
source activate maka_paper

# Line required by the repo
export PYTHONPATH=$HOME/Transparency

# Run the script and train the model

python transformer_results.py --num_runs 5