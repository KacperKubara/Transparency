
# general parameters
output_path=./experiments
diversity_weight=0.5
# TODO: When testing, change this (to 8, I reckon) or don't specify this parameter
n_epochs=1

# TODO: Once all datasets will be working, change to a higher number, for instance 5
num_runs=1
bc_datasets="sst imdb amazon yelp 20News_sports tweet Anemia Diabetes"
qa_datasets="snli qqp cnn babi_1 babi_2 babi_3"
models="vanilla_lstm ortho_lstm diversity_lstm"
for dataset in $datasets; do
    echo "Dataset "${dataset}
    for model in $models; do
        echo "Model "${model}
        for (( i=0; i<num_runs; ++i)); do 
            echo "Run "${i}
            # TODO add an actual run
        done
    done
done