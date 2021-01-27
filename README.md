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
Clone this repository into a folder named Transparency (This step is necessary)

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
#### WSL2 Additional Comments on Installation from Kacper
Run this instead of the lines above:
```
conda create -n "maka_paper" Python=3.7
conda activate maka_paper
pip install -r requirements.txt
python -m spacy download en
conda install -c anaconda jupyter
conda install -c anaconda pytest
```
#### GOLD COMMENT
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
:/home/kacpe/repos/Transparency
```

## Preparing the Datasets 

Each dataset has a separate ipython notebook in the `./preprocess` folder. Follow the instructions in the ipython notebooks to download and preprocess the datasets.

## Training & Running Experiments

The below mentioned commands trains a given model on a dataset and performs all the experiments mentioned in the paper. 

### Text Classification datasets

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

### Tasks with two input sequences (NLI, Paraphrase Detection, QA)

```
python train_and_run_experiments_qa.py --dataset ${dataset_name} --data_dir . --output_dir ${output_path} --encoder ${model_name} --diversity ${diversity_weight}
```

The ```dataset_name``` can be any of ```snli```, ```qqp```, ```cnn```, ```babi_1```, ```babi_2```, and ```babi_3```. 
As before, ```model_name``` can be ```vanilla_lstm```, ```ortho_lstm```, or ```diversity_lstm```. 











