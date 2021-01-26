import argparse
import nltk
import time 

nltk.download('averaged_perceptron_tagger')


parser = argparse.ArgumentParser(description='Run experiments on a dataset')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str)
parser.add_argument('--encoder', type=str, choices=['vanilla_lstm', 'ortho_lstm', 'diversity_lstm', "transformer"], required=True)
parser.add_argument("--diversity",type=float,default=0)

args, extras = parser.parse_known_args()
args.extras = extras
args.attention = 'tanh'

from Transparency.Trainers.DatasetBC import *
from Transparency.ExperimentsBC import *

start_time = time.time()
dataset = datasets[args.dataset](args)

if args.output_dir is not None :
    dataset.output_dir = args.output_dir

dataset.diversity = args.diversity
encoders = [args.encoder]
train_dataset_on_encoders(dataset, encoders)
generate_graphs_on_encoders(dataset, encoders)
end_time = time.time()

print(f"Computation time was {end_time - start_time}")
print(f"For {args.dataset}, {args.encoder}")