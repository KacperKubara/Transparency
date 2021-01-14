import requests
import tarfile 
import os
import json
import pandas
import subprocess
import string

def download_zips(url):
    if isinstance(url, dict):
        download_url(url["url"], url["save_path"])
    else:
        raise TypeError
        
def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def untar(f_name):
    tar = tarfile.open(f_name)
    tar.extractall()
    tar.close()
    return f_name[:-7] # Works only for tar.gz

def get_preprocessed_text(dir_path, lang):
    categories = [
        "books",
        "dvd",
        "music"
    ]
    X = []
    labels = []

    for cat in categories:
        path = dir_path + "/" + cat
        with open(path + "/train.processed", 'r') as f:
            lines = f.readlines()
        X_train, labels_train = preprocess(lines, lang)
        with open(path + "/test.processed", 'r') as f:
            lines = f.readlines()
        X_test, labels_test = preprocess(lines, lang)

        X = X_train + X_test
        labels = labels_train + labels_test

    return X, labels

def preprocess(lines, lang):
    X = []
    labels = []
    target_to_int = {
        "positive": 1,
        "negative": 0
    }
    for line in lines:
        line = line.split()
        
        new_line = []
        for word in line[:-1]:
            # Split on the last delimeter
            new_line.append(word.rsplit(":")[0])
        
        # Split on the last delimeter
        target = line[-1].rsplit(":", 1)[1]
        
        # Preprocessing depending on the lang
        if lang == "en":
            with open("contraction_map.json") as f:
                contraction_map = json.load(f)
        else:
            contraction_map=None
        new_line = tokenize_and_stem(new_line, lang, contraction_map)
        
        labels.append(target_to_int[target])
        X.append(new_line)
 
    assert len(labels) == len(X)
    return X, labels

def tokenize_and_stem(line, lang, contraction_map=None):
    new_line = []
    for word in line:
        # 1) Map digit tokens to <num>
        word = "<num>" if word.isdigit() else word
        if word in string.punctuation:
            continue

        # 2) Lower case the words
        word = word.lower()
        
        # 3) (en only) normalize contactions, e.g., don't -> do not
        if lang == "en":
            if word in contraction_map:
                word = contraction_map[word]
        
        # 4) Tokenize the words (NLTK for de, en, fr, MECAB for jp)
        if lang != "jp":
            pass
        else:
            pass
    
        new_line.append(word)


    return new_line

if __name__ == "__main__":
    url_dict = {
        "url": "https://zenodo.org/record/3251672/files/cls-acl10-processed.tar.gz?download=1",
        "save_path": "./cls-acl10-processed.tar.gz"
    }
    language_dirs = [
        "de",
        "en",
        "fr",
        "jp"
    ]

    
    if not os.path.isfile(url_dict["save_path"]):
        download_zips(url_dict)
    if not os.path.isdir(url_dict["save_path"][:-7]):
        url_dict["save_path"] = untar(url_dict["save_path"])

    print("Preprocessing the files")
    for lang in language_dirs:
        dir_path = "./cls-acl10-processed/" + lang
        labels, X = get_preprocessed_text(dir_path, lang)
        length = len(labels)
        df = pandas.DataFrame({'text': X, 
                               'label': labels, 
                               'exp_split': ['train' for i in range(length)]
                               })
        df = df.sample(frac=1).reset_index(drop=True)
        ix_1 = length//2
        ix_2 = ix_1 + length//4
        df.loc["exp_split", ix_1: ix_2] = "test"
        df.loc["exp_split", ix_2:] = "dev"
        df.to_csv('amazon_dataset_' + lang + '.csv', index=False)
    print("All data has been preprocessed and saved")

    for lang in language_dirs:
        print(f"Calling External script")
        dataset_name = '--data_file amazon_dataset_' + lang + '.csv'
        vec_name = '--output_file amazon_dataset_' + lang + '.p'
        get_ipython().run_line_magic('run', f'"../preprocess_data_BC.py" {dataset_name} {vec_name} --word_vectors_type None --min_df 35')
