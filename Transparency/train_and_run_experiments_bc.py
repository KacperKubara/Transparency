import argparse
import time
import nltk


nltk.download('averaged_perceptron_tagger')

start_time = time.time()
parser = argparse.ArgumentParser(description='Run experiments on a dataset')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str)
parser.add_argument('--encoder', type=str, choices=['vanilla_lstm', 'ortho_lstm', 'diversity_lstm'], required=True)
parser.add_argument("--diversity",type=float,default=0)
parser.add_argument("--n_iter",type=int)

args, extras = parser.parse_known_args()
args.extras = extras
args.attention = 'tanh'

from Transparency.Trainers.DatasetBC import *
from Transparency.ExperimentsBC import *

dataset = datasets[args.dataset](args)
if hasattr(dataset, "n_iter"):
    print("dataset.n_iter", dataset.n_iter)
else:
    print("Number of iterations not specified")

if args.output_dir is not None :
    dataset.output_dir = args.output_dir

dataset.diversity = args.diversity
encoders = [args.encoder]
train_dataset_on_encoders(dataset, encoders)
# if we are not running all experiments, we won't create files required for generating graphs
generate_graphs_on_encoders(dataset, encoders)
end_time = time.time()


end_time = time.time()
elapsed_time = end_time-start_time
elapsed_time_mins = elapsed_time/60
print("Elapsed time", elapsed_time_mins)
print(f"For {args.dataset}, {args.encoder}")
