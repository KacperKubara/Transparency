import pandas as pd

print("### 20News")
print()

news20 = pd.read_csv("20News/20News_sports_dataset.csv")

print("train/dev/test ")
print(news20.exp_split.value_counts())

def label_dist(split):
    return split.label.value_counts()*100/split.shape[0]

print(news20.groupby("exp_split").apply(label_dist))

dataset_paths = ["20News/20News_sports_dataset.csv", "Amazon/amazon_dataset.csv", "Tweets/adr_dataset.csv", "SST/sst_dataset.csv", "IMDB/imdb_dataset.csv", "Babi/babi_qa1_single-supporting-fact_dataset.csv","Babi/babi_qa2_two-supporting-facts_dataset.csv","Babi/babi_qa3_three-supporting-facts_dataset.csv","QQP/QQP/qqp_dataset.csv","SNLI/snli_dataset.csv"]
for dataset in dataset_paths:
    print(f"### {dataset.split('/')[0]}")
    df = pd.read_csv(dataset)
    print("train/dev/test %")
    print(df.exp_split.value_counts())
    # print(df.groupby("exp_split").apply(label_dist).round(2))

