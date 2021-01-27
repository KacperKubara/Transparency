from Transparency.Trainers.TrainerTransformer import go
from argparse import ArgumentParser
import os 

def run_experiments(arg):
    result = go(arg)
    return result



#parser.add_argument("--csv_path",
#                    help = "Path to the csv containing the preprocessed information",
#                    type = str, default = )



if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--model_type",
                            choices= ["diversity_transformer", "vanilla_transformer"],
                            required=True, type=str)

    parser.add_argument("--dataset_name",
                        help = "Name of the dataset to train on",
                        required = True,type = str)
    parser.add_argument("--random-seed", help = "What random seed to use?",
                        default = 0, type = int)
    

    arg = parser.parse_args()
    run_experiments(arg)

"""
parser.add_argument("-e", "--num-epochs",
                dest="num_epochs",
                help="Number of epochs.",
                default=10, type=int)

parser.add_argument("-b", "--batch-size",
                dest="batch_size",
                help="The batch size.",
                default=4, type=int)

parser.add_argument("-l", "--learn-rate",
                dest="lr",
                help="Learning rate",
                default=0.0001, type=float)

#parser.add_argument("-T", "--tb_dir", dest="tb_dir",
#                    help="Tensorboard logging directory",
#                    default='./runs')

parser.add_argument("--max_pool", dest="max_pool",
                help="Use max pooling in the final classification layer.",
                action="store_true")

parser.add_argument("-E", "--embedding", dest="embedding_size",
                help="Size of the character embeddings.",
                default=300, type=int) # 128

parser.add_argument("-M", "--max", dest="max_length",
                help="Max sequence length. Longer sequences are clipped (-1 for no limit).",
                default=512, type=int)

parser.add_argument("-H", "--heads", dest="num_heads",
                help="Number of attention heads.",
                default=8, type=int)

parser.add_argument("-d", "--depth", dest="depth",
                help="Depth of the network (nr. of self-attention layers/ transformer blocks)",
                default=1, type=int) # 6

parser.add_argument("-r", "--random-seed",
                dest="seed",
                help="RNG seed. Negative for random",
                default=1, type=int)

parser.add_argument("--gradient_clipping",
                    dest="gradient_clipping",
                    help="Gradient clipping.",
                    default=1.0, type=float)

parser.add_argument("--diversity_transformer",
                    help="Add diveristy term to loss?",
                    default=False, type=bool)

parser.add_argument("--diversity_weight",
                    help="Diversity weight for the diveristy part of the loss",
                    default=0.5, type=float) 

parser.add_argument("--delete_prop",
                    help="What proportion of the values with the heighest weight should be set to 0?",
                    default=0, type=float) 

parser.add_argument("--dataset_name",
                help = "Name of the dataset to train on",
                required = True,type = str)

parser.add_argument("--lr-warmup",
            dest="lr_warmup",
            help="Learning rate warmup.",
            default=5_000, type=int)
"""       