import os, pickle, numpy as np
from Transparency.common_code.plotting import * 
from datetime import datetime
import seaborn as sns

def plot_importance_ranking_all_models(dataset_folder, model_folders):
    print("Plot importance for all models...")
    experiment_name = "importance_ranking"
    model_dfs = []
    for model in model_folders:
        # get path to the model file
        file = os.path.join(dataset_path, model_folders[model], experiment_name + '_pdump.pkl')

        if not os.path.isfile(file) :
            raise FileNotFoundError(file + " doesn't exist")

        # read the file
        importance_ranking = pickle.load(open(file, 'rb'))
        
        # in the lines below we reuse some code from plot_boxplot() and plot_importance_ranking()
        values = [importance_ranking['attention'],importance_ranking['random']]
        classes=['Attention','Random']

        
        model_df = {'y':[],'class':[]}
        for y,class_ in zip(values,classes):
            model_df['y'].extend(y)
            model_df['class'].extend([class_]*len(y))

        model_df = pd.DataFrame.from_dict(model_df)
        model_df['model'] = model
        model_dfs.append(model_df)

    fig, ax = init_gridspec(2, 2, 1)
    model_dfs = pd.concat(model_dfs)

    ax = sns.boxplot(x="model", y="y", hue="class", data=model_dfs)

    annotate(ax, ylim=(-0.05, 1.05), ylabel="Fraction of attention weights removed", xlabel="", legend=None, title=dataset_folder)
    # adjust_gridspec()
    save_axis_in_file(fig, ax, out_path, "importance_ranking_MAvDY_all_models")

def plot_bar_pos_att_all_models(dataset_folder, model_folders):
    print("Plot POS attention for all models...")
    experiment_name = "quant_analysis"
    model_dfs = []
    for model in model_folders:
        # get path to the model file
        file = os.path.join(dataset_path, model_folders[model], experiment_name + '_pdump.pkl')

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

    fig, ax = init_gridspec(2, 2, 1)
    model_dfs = pd.concat(model_dfs)
    
    ax = sns.barplot(x="cum_att_perc", y="pos_tag", hue="model", data=model_dfs)
    annotate(ax, ylim=None, ylabel="Part of Speech", xlabel="", legend=None, title=dataset_folder)
    
    # adjust_gridspec()
    save_axis_in_file(fig, ax, out_path, "pos_cummulative_attention")

def plot_permutations_all_models(dataset_folder, model_folders):
    print("Plot permutations for all models...")
    experiment_name = "permutations"
    model_dfs = []
    xlim=(0, 1.0)
    for model in model_folders:
        # get path to the model file
        file = os.path.join(dataset_path, model_folders[model], experiment_name + '_pdump.pkl')

        if not os.path.isfile(file) :
            raise FileNotFoundError(file + " doesn't exist")

        # read the file
        permutations = pickle.load(open(file, 'rb'))

        # get path to the model test outputs
        test_outputs_file = os.path.join(dataset_path, model_folders[model], 'test_output_pdump.pkl')
        if not os.path.isfile(test_outputs_file) :
            raise FileNotFoundError(test_outputs_file + " doesn't exist")
        # read test outputs
        test_data = pickle.load(open(test_outputs_file, 'rb'))

        # code from plot_permutations()
        X, yhat, attn = test_data['X'], test_data['yt_hat'], test_data['attn_hat']
        med_diff = np.abs(np.array(permutations) - yhat[:, None, :]).mean(-1)
        med_diff = np.median(med_diff, 1)
        max_attn = calc_max_attn(X, attn)

        
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
    save_axis_in_file(fig, ax, out_path, "permutation")

'''
    Return current timestamp in unix format
'''
def now_to_unix_ts():
    time_now = datetime.now()
    return int((time_now - datetime(1970,1,1)).total_seconds())

if __name__=="__main__":

    dataset_name = '20News_sports'
    experiments_path = os.path.abspath('../experiments')
    dataset_path = os.path.join(experiments_path, dataset_name)
    
    # put all plots from this run into a folder which is its start time in unix format
    unix_start_time = str(now_to_unix_ts())
    out_path = os.path.join(dataset_path, "all_models", unix_start_time)
    os.makedirs(out_path)
    # create decision flip box plots for all models
    model_folders = {"Vanilla":"lstm+tanh/Mon_Jan_18_09:21:26_2021", 
                     "Diversity":"lstm+tanh__diversity_weight_0.5/Mon_Jan_18_09:24:40_2021",
                     "Ortho":"ortho_lstm+tanh/Mon_Jan_18_09:23:03_2021"}

    plot_importance_ranking_all_models(dataset_folder = dataset_name, model_folders=model_folders)

    # create POS tag attention bar plots for all models
    model_folders = {"Vanilla":"lstm+tanh/Mon_Jan_18_15:47:57_2021", 
                     "Diversity":"lstm+tanh__diversity_weight_0.5/Mon_Jan_18_15:48:46_2021",
                     "Ortho":"ortho_lstm+tanh/Mon_Jan_18_15:48:19_2021"}

    plot_bar_pos_att_all_models(dataset_folder = dataset_name, model_folders=model_folders)

    # create permutations violin plots for all models
    model_folders = {"Vanilla":"lstm+tanh/Mon_Jan_18_16:24:22_2021", 
                     "Diversity":"lstm+tanh__diversity_weight_0.5/Mon_Jan_18_16:27:07_2021",
                     "Ortho":"ortho_lstm+tanh/Mon_Jan_18_16:25:41_2021"}
                        
    plot_permutations_all_models(dataset_folder=dataset_name, model_folders=model_folders)