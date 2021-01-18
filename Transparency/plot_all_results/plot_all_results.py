import os, pickle, numpy as np
from Transparency.common_code.plotting import * 
from datetime import datetime

def plot_importance_ranking_all_models(dataset_folder, model_folders):
    experiments_path = os.path.abspath('../experiments')
    dataset_path = os.path.join(experiments_path, dataset_folder)
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
    out_path = os.path.join(dataset_path, "all_models", str(now_to_unix_ts()))
    os.makedirs(out_path)
    save_axis_in_file(fig, ax, out_path, "importance_ranking_MAvDY_all_models")

'''
    Return current timestamp in unix format
'''
def now_to_unix_ts():
    time_now = datetime.now()
    return int((time_now - datetime(1970,1,1)).total_seconds())

model_folders = {"Vanilla":"lstm+tanh/Mon_Jan_18_09:21:26_2021", 
                    "Diversity":"lstm+tanh__diversity_weight_0.5/Mon_Jan_18_09:24:40_2021",
                    "Ortho":"ortho_lstm+tanh/Mon_Jan_18_09:23:03_2021"}
plot_importance_ranking_all_models(dataset_folder = '20News_sports', model_folders=model_folders)