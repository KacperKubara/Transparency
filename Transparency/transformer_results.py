from Transparency.model.transformerDataHandling import vectorize_data, dataset_config, datasets
from argparse import ArgumentParser
import os
from Transparency.ExperimentsTransformer import run_experiments
import easydict
import numpy as np

def gather_stats(arg):
    log_file = open("god_script_transformer", "w+" )
    options = ["diversity_transformer", "vanilla_transformer"]
    
    for d in datasets.keys():

        print(f"\n Working on the dataset {d}")
        for opt in options:
            test_accs = []
            conicities = []
            print(f"\n Running model type {opt}")
            for i in range(arg.num_runs):
                print(f'\n Dataset {d}, model type {opt}, run {i+1}')
                run_params = easydict.EasyDict({"model_type":opt,"dataset_name": d})
                best_acc, test_acc, mean_epoch_conicity = run_experiments(run_params)
                
                test_accs.append(test_acc)
                conicities.append(conicities)

                log_file.write(f'\n Dataset {d}, model type {opt}, run {i+1}')
                log_file.write(f'    \n Val_acc {best_acc}, test_acc {test_acc}, conicity {mean_epoch_conicity}')
                
                print(f'\n Val_acc {best_acc}, test_acc {test_acc}, conicity {mean_epoch_conicity}')

            log_file.write(f'\n Mean test accuracies {np.mean(test_accs)}, std {np.std(test_accs)}')
            log_file.write(f'\n Mean conicities {np.mean(conicities)}, std {np.std(conicities)}')
            print(f'Mean test accuracies {np.mean(test_accs)}, std {np.std(test_accs)}')
            print(f'Mean conicities {np.mean(conicities)}, std {np.std(conicities)}')





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_runs", help = "How many times to run all datasets?",
                        default = 2, type = int)
    

    arg = parser.parse_args()
    gather_stats(arg)   


