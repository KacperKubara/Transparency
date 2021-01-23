import os, pickle, numpy as np
from Transparency.common_code.plotting import * 
from datetime import datetime
import seaborn as sns
import ast
import json

'''
    Table and figure numbers are corresponding to tables and figures in the original paper:
    
    Mohankumar, A. K., Nema, P., Narasimhan, S., Khapra, M. M., Srinivasan, B. V., & Ravindran, B. (2020). 
    Towards transparent and explainable attention models. arXiv preprint arXiv:2004.14243.
'''

def plot_importance_ranking_all_models(dataset_folder, model_folders, dataset_type):
    print("Plot importance for all models...")
    experiment_name = "importance_ranking"
    model_dfs = []
    for model in model_folders:
        # get path to the model file
        file = os.path.join(in_dataset_path, model_folders[model], experiment_name + '_pdump.pkl')

        if not os.path.isfile(file) :
            raise FileNotFoundError(file + " doesn't exist")

        # read the file
        importance_ranking = pickle.load(open(file, 'rb')) 
        
        # in the lines below we reuse some code from plot_boxplot() and plot_importance_ranking()
        if dataset_type == "bc":
            values = [importance_ranking['attention'],importance_ranking['random']]
        elif dataset_type == "qa":
            values = [importance_ranking[0],importance_ranking[1]]
        else:
            assert False, "Wrong dataset type. Dataset type can be either 'bc', or 'qa'."

        classes=['Attention','Random']

        
        model_df = {'y':[],'class':[]}
        for y,class_ in zip(values,classes):
            model_df['y'].extend(y)
            model_df['class'].extend([class_]*len(y))

        model_df = pd.DataFrame.from_dict(model_df)
        model_df['model'] = model
        model_dfs.append(model_df)

    fig, ax = init_gridspec(2, 2, 1, figsize=(15,8))
    model_dfs = pd.concat(model_dfs)

    ax = sns.boxplot(x="model", y="y", hue="class", data=model_dfs)
    annotate(ax, ylim=(-0.05, 1.05), ylabel="Fraction of attention weights removed", xlabel="", legend=None, title=dataset_folder, fontsize = 12)
    # adjust_gridspec()
    save_axis_in_file(fig, ax, out_path, f"{dataset_folder}_importance_ranking_MAvDY_all_models")

def plot_bar_pos_att_all_models(dataset_folder, model_folders):
    print("Plot POS attention for all models...")
    experiment_name = "quant_analysis"
    model_dfs = []
    for model in model_folders:
        # get path to the model file
        file = os.path.join(in_dataset_path, model_folders[model], experiment_name + '_pdump.pkl')

        if not os.path.isfile(file) :
            raise FileNotFoundError(file + " doesn't exist")

        # read the file
        pos_att = pickle.load(open(file, 'rb'))

        cum_att = [tag[1][1] for tag in pos_att['pos_tags']]
        tags = [tag[0] for tag in pos_att['pos_tags']]

        cum_att = np.array(cum_att)
        
        # compute percentages
        cum_att_perc = cum_att*100/np.sum(cum_att)

        model_df = pd.DataFrame(data={"pos_tag":tags,"cum_att_perc":cum_att_perc, "model":model})
        model_dfs.append(model_df)

    fig, ax = init_gridspec(2, 2, 1, figsize=(15, 8))
    model_dfs = pd.concat(model_dfs)
    
    ax = sns.barplot(x="cum_att_perc", y="pos_tag", hue="model", data=model_dfs)
    annotate(ax, ylim=None, ylabel="Part of Speech", xlabel="", legend=None, title=dataset_folder, fontsize = 15)
    
    # adjust_gridspec()
    save_axis_in_file(fig, ax, out_path, f"{dataset_folder}_pos_cummulative_attention")

def plot_permutations_all_models(dataset_folder, model_folders, dataset_type):
    print("Plot permutations for all models...")
    experiment_name = "permutations"
    model_dfs = []
    xlim=(0, 1.0)
    for model in model_folders:
        # get path to the model file
        file = os.path.join(in_dataset_path, model_folders[model], experiment_name + '_pdump.pkl')

        if not os.path.isfile(file) :
            raise FileNotFoundError(file + " doesn't exist")

        # read the file
        permutations = pickle.load(open(file, 'rb'))

        # get path to the model test outputs
        test_outputs_file = os.path.join(in_dataset_path, model_folders[model], 'test_output_pdump.pkl')
        if not os.path.isfile(test_outputs_file) :
            raise FileNotFoundError(test_outputs_file + " doesn't exist")
        # read test outputs
        test_data = pickle.load(open(test_outputs_file, 'rb'))


        # code from plot_permutations() in PlottingBC.py and PlottingQA.py
        if dataset_type == "bc":
            X, yhat, attn = test_data['X'], test_data['yt_hat'], test_data['attn_hat']
            med_diff = np.abs(np.array(permutations) - yhat[:, None, :]).mean(-1)
            med_diff = np.median(med_diff, 1)
            max_attn = calc_max_attn(X, attn)
        elif dataset_type == "qa":
            X, attn, yhat = test_data['P'], test_data['attn_hat'], test_data['yt_hat']
            ad_y, ad_diffs = permutations
            ad_diffs = 0.5*np.array(ad_diffs)

            med_diff = np.median(ad_diffs, 1)
            max_attn = calc_max_attn(X, attn)
        else:
            assert False, "Wrong dataset type. Dataset type can be either 'bc', or 'qa'."
        
        # plot_violin_by_class(ax[0], max_attn, med_diff, yhat, xlim=(0, 1.0))
        # code from plot_violin_by class()
        # def plot_violin_by_class(ax, X_vals, Y_vals, yhat, xlim, bins=4) :

        # don't move this line out of this for loop because it is modified below, so we need to reset it to 4 for every model
        bins = 4
        bins = xlim[0] + np.arange(bins+1) / bins * (xlim[1]+1e-4 - xlim[0])
        xbins = np.digitize(max_attn, bins[1:])
        order = ["[" + "{:.2f}".format(bins[p]) + ',\n' + "{:.2f}".format(bins[p+1]) + ")" for p in np.arange(len(bins)-1)]

        xnames = []
        for p in xbins :
            xnames.append("[" + "{:.2f}".format(bins[p]) + ',\n' + "{:.2f}".format(bins[p+1]) + ")")
        
        # classes = np.zeros((len(yhat,)))
        # if len(yhat.shape) == 1 or yhat.shape[1] == 1:
        #     yhat = yhat.flatten()
        #     yhat = np.round(yhat)
        #     unique_y = np.sort(np.unique(yhat))
        #     if len(unique_y) < 4 :
        #         classes = yhat

        model_df = pd.DataFrame({'bin' : xnames, 'val' : med_diff, 'model' : model})
        model_dfs.append(model_df)

    fig, ax = init_gridspec(2, 2, 1)

    model_dfs = pd.concat(model_dfs)
    ax = sns.violinplot(data=model_dfs, y="bin", x="val", hue="model", linewidth=1.0, order=order, cut=0.02, inner='quartiles', dodge=True)

    # ax.get_legend().remove()

    annotate(ax, xlim=xlim, ylabel="Max attention", xlabel="Median Output Difference", legend=None, title=dataset_folder)

    # adjust_gridspec()
    save_axis_in_file(fig, ax, out_path, f"{dataset_folder}_permutation")

'''
    Return current timestamp in unix format
'''
def now_to_unix_ts():
    time_now = datetime.now()
    return int((time_now - datetime(1970,1,1)).total_seconds())

def make_table2(datasets):
    table2_latex = "### Latex content for Table 2 \n"
    for dataset_name, dataset in datasets.items():
        dataset_results = f"{dataset_name} & "
        for model_name, model_folder in dataset["model_folders"].items():
            evaluate_file = os.path.join(experiments_path, dataset["exp_folder_name"], model_folder, "evaluate.json")
            with open(evaluate_file) as f:
                table2 = json.load(f)
            dataset_results += f"{np.round(float(table2['accuracy']*100),2)} & {np.round(float(table2['conicity_mean']),2)} & "
        dataset_results = dataset_results.rstrip("& ") + " \\\\ \n"
        table2_latex += dataset_results
    table2_path = os.path.join(out_path,"table2_latex.txt")
    with open(table2_path, "w") as f:
        f.write(table2_latex)

def make_table3(datasets):
    table3_latex = "### Latex content for Table 3 \n"
    for dataset_name, dataset in datasets.items():
        dataset_type = dataset["type"]
        # Rationale are only possible for BC tasks
        if dataset_type == "bc":
            dataset_results = f"{dataset_name} & "
            for model_name, model_folder in dataset["model_folders"].items():
                evaluate_file = os.path.join(experiments_path, dataset["exp_folder_name"], model_folder, "rationale_summary_test.txt")
                with open(evaluate_file) as f:
                    table3 = ast.literal_eval(f.read())
                dataset_results += f"{np.round(float(table3['Attn Sum Average']),3)} & {np.round(float(table3['Fraction Length Average']),3)} & "
            dataset_results = dataset_results.rstrip("& ") + " \\\\ \n"
            table3_latex += dataset_results
    table3_path = os.path.join(out_path,"table3_latex.txt")
    with open(table3_path, "w") as f:
        f.write(table3_latex)

def make_table4(datasets):
    table4_latex = "### Latex content for Table 4 \n"
    for dataset_name, dataset in datasets.items():
        dataset_results = f"{dataset_name} & "
        
        table4_files = ["Attn_Gradient_X_val_pearsonr.csv", "Attn_Integrated_Gradient_val_pearsonr.csv", "Attn_Gradient_X_val_jsd.csv", "Attn_Integrated_Gradient_val_jsd.csv"]

        for file in table4_files:
            for model_name, model_folder in dataset["model_folders"].items():
                if model_name != "Orthogonal LSTM":
                    file_path = os.path.join(experiments_path, dataset["exp_folder_name"], model_folder, file)
                    file_pd = pd.read_csv(file_path, index_col="Unnamed: 0").round(2)
                    mean = file_pd.loc["Overall", "mean"]
                    std = file_pd.loc["Overall", "std"]
                    dataset_results += f"{mean} \pm {std} & "
        dataset_results = dataset_results.rstrip("& ") + " \\\\ \n"
        table4_latex += dataset_results
    table4_path = os.path.join(out_path,"table4_latex.txt")
    with open(table4_path, "w") as f:
        f.write(table4_latex)

if __name__=="__main__":

    datasets = {"SST":{"exp_folder_name":"sst",
                       "type":"bc",
                       "model_folders":{"Vanilla":"lstm+tanh/Wed_Jan_20_02:23:00_2021",
                                        "Diversity":"lstm+tanh__diversity_weight_0.5/Wed_Jan_20_03:10:56_2021",
                                        "Orthogonal":"ortho_lstm+tanh/Wed_Jan_20_02:45:48_2021"
                                        }
                            
                        },
                
                "IMDB":{"exp_folder_name":"imdb",
                        "type":"bc",
                        "model_folders":{"Vanilla":"lstm+tanh/Wed_Jan_20_03:32:58_2021",
                                         "Diversity":"lstm+tanh__diversity_weight_0.5/Wed_Jan_20_10:39:41_2021",
                                         "Orthogonal":"ortho_lstm+tanh/Wed_Jan_20_06:31:42_2021"}
                        },
                # Amazon
                "20News":{"exp_folder_name":"20News_sports",
                          "type":"bc",
                          "model_folders":{"Vanilla":"lstm+tanh/Wed_Jan_20_01:14:09_2021",
                                           "Diversity":"lstm+tanh__diversity_weight_0.5/Wed_Jan_20_02:01:35_2021",
                                           "Orthogonal": "ortho_lstm+tanh/Wed_Jan_20_01:35:55_2021"},
                        },
                # Tweets
                # SNLI
                # QQP
                "bAbI 1":{"exp_folder_name":"babi_1",
                          "type":"qa",
                          "model_folders":{"Vanilla":"lstm+tanh/Wed_Jan_20_23:54:13_2021",
                                           "Diversity":"lstm+tanh__diversity_weight_0.5/Thu_Jan_21_01:14:50_2021",
                                           "Orthogonal":"ortho_lstm+tanh/Thu_Jan_21_00:28:30_2021"}
                         },
                "bAbi 2":{"exp_folder_name":"babi_2",
                          "type":"qa",
                          "model_folders":{"Vanilla":"lstm+tanh/Thu_Jan_21_01:49:56_2021",
                                           "Diversity":"lstm+tanh__diversity_weight_0.5/Thu_Jan_21_06:04:25_2021",
                                           "Orthogonal":"ortho_lstm+tanh/Thu_Jan_21_03:38:39_2021"}
                         }
                # bAbi 3
                
                }

    
    experiments_path = os.path.abspath('../experiments')
    
    # put all plots from this run into a folder which is its start time in unix format
    unix_start_time = str(now_to_unix_ts())
    out_path = os.path.join(experiments_path, "all_datasets", unix_start_time)
    os.makedirs(out_path)

    make_table2(datasets)
    make_table3(datasets)
    make_table4(datasets)

    for dataset_name, dataset in datasets.items():
        print(f"### Plots for dataset ${dataset_name}")

        dataset_folder = dataset["exp_folder_name"]
        model_folders = dataset["model_folders"]
        dataset_type = dataset["type"]
        in_dataset_path = os.path.join(experiments_path, dataset_folder)

        # plot importance (Figure 3)
        plot_importance_ranking_all_models(dataset_folder = dataset_folder, model_folders = model_folders, dataset_type = dataset_type)

        # create POS tag attention bar plots for all models (Figure 4)
        plot_bar_pos_att_all_models(dataset_folder = dataset_folder, model_folders = model_folders)

        # create permutations violin plots for all models (Figure 5)
        plot_permutations_all_models(dataset_folder = dataset_folder, model_folders = model_folders, dataset_type = dataset_type)
    