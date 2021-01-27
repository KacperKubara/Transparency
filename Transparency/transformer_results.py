from Transparency.model.transformerDataHandling import vectorize_data, dataset_config, datasets
from argparse import ArgumentParser
import os
from Transparency.ExperimentsTransformer import run_experiments
import easydict
import numpy as np

def gather_stats(arg):
    log_file = open("god_script_transformer", "w+" )
    options = ["vanilla_transformer", "diversity_transformer"]
    
    for d in datasets.keys():
        if d == "amazon":
            print("Skipping amazon")
            continue
        print(f"\n \n \n \n Working on the dataset {d}")
        log_file.write(f"\n \n \n \n Working on the dataset {d}")
        for opt in options:
            try:
                test_accs = []
                conicities = []
                print(f"\n \n  \n Running model type {opt}")
                log_file.write(f"\n \n  \n Running model type {opt}")
                for i in range(arg.num_runs):
                    print(f'\n Dataset {d}, model type {opt}, run {i+1}')
                    run_params = easydict.EasyDict({"model_type":opt,"dataset_name": d})
                    best_acc, test_acc, mean_epoch_conicity = run_experiments(run_params)
                    
                    test_accs.append(test_acc)
                    conicities.append(mean_epoch_conicity.item())

                    log_file.write(f'\n Dataset {d}, model type {opt}, run {i+1}')
                    log_file.write(f'    \n Val_acc {best_acc}, test_acc {test_acc}, conicity {mean_epoch_conicity}')
                    
                    print(f'\n Val_acc {best_acc}, test_acc {test_acc}, conicity {mean_epoch_conicity}')

                mean_test_accs = np.mean(test_accs)
                std_test_accs = np.std(test_accs)
                mean_conicities = np.mean(conicities)
                std_conicities =  np.std(conicities)

                log_file.write(f'\n \n Mean test accuracies {mean_test_accs}, std {std_test_accs}')
                log_file.write(f'\n \n  Mean conicities {mean_conicities}, std {std_conicities}')
                print(f'\n Mean test accuracies {mean_test_accs}, std {std_test_accs}')
                print(f'\n Mean conicities {mean_conicities}, std {std_conicities}')
            except Exception as e:
                print(f'\n Dataset {d}, model type {opt} failed with exception {e}')
            

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_runs", help = "How many times to run all datasets?",
                        default = 3, type = int)
    

    arg = parser.parse_args()
    gather_stats(arg)   


