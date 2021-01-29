[![KacperKubara](https://circleci.com/gh/KacperKubara/Transparency.svg?style=shield&circle-token=a5666e678dfb36927f320d07b004fd9ee6ae0a08)](https://app.circleci.com/pipelines/github/KacperKubara/Transparency)

# Towards Transparent and Explainable Attention Models
Code for [Towards Transparent and Explainable Attention Models](https://www.aclweb.org/anthology/2020.acl-main.387/) paper (ACL 2020)

Please note that the code is a modified copy of: https://github.com/akashkm99/Interpretable-Attention

When using this code, please cite:
```
@inproceedings{mohankumar-etal-2020-towards,
    title = "Towards Transparent and Explainable Attention Models",
    author = "Mohankumar, Akash Kumar  and
      Nema, Preksha  and
      Narasimhan, Sharan  and
      Khapra, Mitesh M.  and
      Srinivasan, Balaji Vasan  and
      Ravindran, Balaraman",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.387",
    pages = "4206--4216"
}
```

This codebase has been built based on this [repo](https://github.com/successar/AttentionExplanation) 

## Installation 
### Installing repository with a Docker image
Make sure that you have a [Docker](https://docs.docker.com/get-docker/) installed. If you use Windows10, you need to have [WSL2](https://docs.microsoft.com/en-gb/windows/wsl/install-win10#step-4---download-the-linux-kernel-update-package) as well.

Follow this steps to run the repo with Docker:

1)  `git clone  https://github.com/KacperKubara/Transparency.git`
2)  `cd Transparency`
3)  `docker build -t transparency .` (can take a while to install)
4)  `docker run -it transparency`
5)  `conda activate maka_paper`
6)  Now you have the exact setup necessary for running the package along with all necessary dependencies

### Manual installation
You need to use Linux to run these experiments. 

Add your present working directory, in which the Transparency folder is present, to your python path 

```export PYTHONPATH=$PYTHONPATH:$(pwd)```

To avoid having to change your python path variable each time, use: ``` echo 'PYTHONPATH=$PYTHONPATH:'$(pwd) >> ~/.bashrc```

#### Requirements 

```
torch==1.1.0
torchtext==0.4.0
pandas==0.24.2
nltk==3.4.5
tqdm==4.31.1
typing==3.6.4
numpy==1.16.2
allennlp==0.8.3
scipy==1.2.1
seaborn==0.9.0
gensim==3.7.2
spacy==2.1.3
matplotlib==3.0.3
ipython==7.4.0
scikit_learn==0.20.3
easy_dict==1.9
```

Install the required packages and download the spacy en model:
```
cd Transparency 
pip install -r requirements.txt
python -m spacy download en
```


#### (Optional) Use an isolated Conda environment
Run this instead of the lines above:
```
conda create -n "maka_paper" Python=3.7
conda activate maka_paper
pip install -r requirements.txt
python -m spacy download en
conda install -c anaconda jupyter
conda install -c anaconda pytest
```
---

Inside your terminal, open Python terminal to download NLTK taggers:
```
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
```

#### Set PYTHONPATH
To set the PYTHONPATH:
```
cd Transparency
echo 'PYTHONPATH=$PYTHONPATH:'$(pwd) >> ~/.bashrc
source ~/.bashrc
reset
```
You can check if the PYTHONPATH is set correctly by typing:
```
echo $PYTHONPATH
```

In my case, it outputs:
```
:/home/kacper/repos/Transparency
```

## Preparing the Datasets 

Each dataset has a separate ipython notebook in the `./preprocess` folder. Follow the instructions in the ipython notebooks to download and preprocess the datasets.

## Training & Running Experiments for Single Datasets

The below mentioned commands trains a given model on a dataset and performs all the experiments mentioned in the paper. 

### Text Classification datasets

#### LSTM-based models

```
python train_and_run_experiments_bc.py --dataset ${dataset_name} --data_dir . --output_dir ${output_path} --encoder ${model_name} --diversity ${diversity_weight}
```

```dataset_name``` can be any of the following: ```sst```, ```imdb```, ```amazon```,```yelp```,```20News_sports``` ,```tweet```, ```Anemia```, and ```Diabetes```.
```model_name``` can be ```vanilla_lstm```, or ```ortho_lstm```, ```diversity_lstm```. 
Only for the ```diversity_lstm``` model, the ```diversity_weight``` flag should be added. 

For example, to train and run experiments on the IMDB dataset with the Orthogonal LSTM, use:

```
dataset_name=imdb
model_name=ortho_lstm
output_path=./experiments
python train_and_run_experiments_bc.py --dataset ${dataset_name} --data_dir . --output_dir ${output_path} --encoder ${model_name} 
```

Similarly, for the Diversity LSTM, use

```
dataset_name=imdb
model_name=diversity_lstm
output_path=./experiments
diversity_weight=0.5
python train_and_run_experiments_bc.py --dataset ${dataset_name} --data_dir . --output_dir ${output_path} --encoder ${model_name} --diversity ${diversity_weight}
```
#### Transformer-based models
For the transformer-based experiments use:

```
python ExperimentsTransformer.py --model_type ${} --dataset_name ${} 
```

```dataset_name``` can be any of the following: ```sst```, ```imdb```, ```amazon```,```20News_sports``` ,```tweet```, ```cls_en```,```cls_de```,```cls_fr```,```cls_jp```
```model_type``` can be ```vanilla_transformer```, or ```diversity_transformer```

You can also easily run all the transformer related experiments from our paper with a simple command:

```
python transformer_results.py --num_runs 5
```

### Tasks with two input sequences (NLI, Paraphrase Detection, QA)

```
python train_and_run_experiments_qa.py --dataset ${dataset_name} --data_dir . --output_dir ${output_path} --encoder ${model_name} --diversity ${diversity_weight}
```

The ```dataset_name``` can be any of ```snli```, ```qqp```, ```cnn```, ```babi_1```, ```babi_2```, and ```babi_3```. 
As before, ```model_name``` can be ```vanilla_lstm```, ```ortho_lstm```, or ```diversity_lstm```. 

## Training & Running All Reproduced Experiments
To train and perform all experiments from the original paper which we reproduced, run the `run_all_datasets.sh` script. Beware that these experiments together took more than 24 hours on a GTX 1080Ti GPU, so you might prefer to run the datasets separately. That is why we don't add this part to the `results.ipynb` notebook. Often making all experiments on the trained model would take longer than training the model itself.

Once the script is done, the results are logged in the experiments folder. We did not attach all our experimtents as they surpass 1 GB of memory. This is how an example directory will look:
```
-- Transparency /
    |
    -- experiments / % folder with all experiments
       |
       -- babi_1 / % folder with all experiments of a given dataset
          |
          -- lstm+tanh / % folder with experiments of a given dataset and model
             |
             -- Wed_Jan_20_23:54:13_2021 / % folder with results of a particular run
        
```
Now you have to manually replace names of the log folders in the plot_all_results/plot_all_results.py as values of the `model_folders` attributes for each dataset in the `datasets` dictionary. To obtain plots and tables (in latex format) which you can see in our paper, run this Python script (takes about 30 mins to run):
```
python plot_all_results/plot_all_results.py
```
The outputs will appear in the `experiments` folder in the `all_datasets` folder. The name of the folder with the results is the creation date in Unix format (the highest number is the most recent). You can see an example in our `experiments` folder which overlaps with the results in our paper.









