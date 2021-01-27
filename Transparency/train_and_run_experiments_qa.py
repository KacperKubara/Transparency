import argparse
import time

start_time = time.time()

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

if hasattr(dataset, "n_iters"):
    print("dataset.n_iter", dataset.n_iters)
else:
    print("Number of iterations not specified")

if args.output_dir is not None :
    dataset.output_dir = args.output_dir

encoders = [args.encoder]

dataset.diversity = args.diversity

train_dataset_on_encoders(dataset, encoders)
# if we are not running all experiments, we won't create files required for generating graphs
generate_graphs_on_encoders(dataset, encoders)

end_time = time.time()
elapsed_time = end_time-start_time
elapsed_time_mins = elapsed_time/60
print("Elapsed time", elapsed_time_mins)
