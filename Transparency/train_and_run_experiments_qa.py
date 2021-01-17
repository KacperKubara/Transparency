import argparse
parser = argparse.ArgumentParser(description='Run experiments on a dataset')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str)
parser.add_argument('--encoder', type=str, choices=['ortho_lstm','vanilla_lstm','diversity_lstm'], required=True)
parser.add_argument('--attention', type=str, choices=['tanh', 'dot', 'all'], required=True)
parser.add_argument("--diversity",type=float,default=0)
parser.add_argument("--n_iter",type=int)

args, extras = parser.parse_known_args()
args.extras = extras

from Transparency.Trainers.DatasetQA import *
from Transparency.ExperimentsQA import *

dataset = datasets[args.dataset](args)
args.attention='tanh'

print("dataset.n_iter", dataset.n_iters)
if args.output_dir is not None :
    dataset.output_dir = args.output_dir

encoders = [args.encoder]

dataset.diversity = args.diversity

train_dataset_on_encoders(dataset, encoders)
# if we are not running all experiments, we won't create files required for generating graphs
# generate_graphs_on_encoders(dataset, encoders)


