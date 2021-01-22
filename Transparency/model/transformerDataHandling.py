import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from math import ceil
from tqdm import tqdm
from torchtext.vocab import pretrained_aliases
import torch

SOS = "<SOS>"
EOS = "<EOS>"
PAD = "<0>"
UNK = "<UNK>"

import spacy, re

nlp = spacy.load("en", disable=["parser", "tagger", "ner"])


class Vectorizer:
    def __init__(self, num_words=None, min_df=None):
        self.embeddings = None
        self.word_dim = 200
        self.num_words = num_words
        self.min_df = min_df

    def process_to_docs(self, texts):
        docs = [t.replace("\n", " ").strip() for t in texts]
        return docs

    def process_to_sentences(self, texts):
        docs = [t.split("\n") for t in texts]
        return docs

    def tokenizer(self, text):
        return text.split(" ")

    def fit(self, texts):
        if self.min_df is not None:
            self.cvec = CountVectorizer(tokenizer=self.tokenizer, min_df=self.min_df, lowercase=False)
        else:
            self.cvec = CountVectorizer(tokenizer=self.tokenizer, lowercase=False)

        bow = self.cvec.fit_transform(texts)

        self.word2idx = self.cvec.vocabulary_

        for word in self.cvec.vocabulary_:
            self.word2idx[word] += 4

        self.word2idx[PAD] = 0
        self.word2idx[UNK] = 1
        self.word2idx[SOS] = 2
        self.word2idx[EOS] = 3

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

        self.cvec.stop_words_ = None

    def add_word(self, word):
        if word not in self.word2idx:
            idx = max(self.word2idx.values()) + 1
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.vocab_size += 1

    def fit_docs(self, texts):
        docs = self.process_to_docs(texts)
        self.fit(docs)

    def convert_to_sequence(self, texts):
        texts_tokenized = map(self.tokenizer, texts)
        texts_tokenized = map(lambda s: [SOS] + [UNK if word not in self.word2idx else word for word in s] + [EOS], texts_tokenized)
        texts_tokenized = list(texts_tokenized)
        sequences = map(lambda s: [int(self.word2idx[word]) for word in s], texts_tokenized)
        return list(sequences)

    def texts_to_sequences(self, texts):
        unpad_X = self.convert_to_sequence(texts)
        return unpad_X

    def extract_embeddings(self, model):
        self.word_dim, self.vocab_size = model.vector_size, len(self.word2idx)
        self.embeddings = np.zeros([self.vocab_size, self.word_dim])
        in_pre = 0
        for i, word in sorted(self.idx2word.items()):
            if word in model:
                self.embeddings[i] = model[word]
                in_pre += 1
            else:
                self.embeddings[i] = np.random.randn(self.word_dim)

        self.embeddings[0] = np.zeros(self.word_dim)

        print("Found " + str(in_pre) + " words in model out of " + str(len(self.idx2word)))
        return self.embeddings

    def extract_embeddings_from_torchtext(self, model):
        vectors = pretrained_aliases[model](cache='../.vector_cache')
        self.word_dim = vectors.dim
        self.embeddings = np.zeros((len(self.idx2word), self.word_dim))
        in_pre = 0
        for i, word in self.idx2word.items():
            if word in vectors.stoi : in_pre += 1                
            self.embeddings[i] = vectors[word].numpy()

        self.embeddings[0] = np.zeros(self.word_dim)
        print("Found " + str(in_pre) + " words in model out of " + str(len(self.idx2word)))
        return self.embeddings

    def get_seq_for_docs(self, texts):
        docs = self.process_to_docs(texts)  # D
        seq = self.texts_to_sequences(docs)  # D x W

        return seq

    def get_seq_for_sents(self, texts):
        sents = self.process_to_sentences(texts)  # (D x S)
        seqs = []
        for d in tqdm(sents):
            seqs.append(self.texts_to_sequences(d))

        return seqs

    def map2words(self, sent):
        return [self.idx2word[x] for x in sent]

    def map2words_shift(self, sent):
        return [self.idx2word[x + 4] for x in sent]

    def map2idxs(self, words):
        return [self.word2idx[x] if x in self.word2idx else self.word2idx[UNK] for x in words]

    def add_frequencies(self, X):
        freq = np.zeros((self.vocab_size,))
        for x in X:
            for w in x:
                freq[w] += 1
        freq = freq / np.sum(freq)
        self.freq = freq


def sortbylength(X, y) :
    len_t = np.argsort([len(x) for x in X])
    X1 = [X[i] for i in len_t]
    y1 = [y[i] for i in len_t]
    return X1, y1
    
def filterbylength(X, y, min_length = None, max_length = None) :
    lens = [len(x)-2 for x in X]
    min_l = min(lens) if min_length is None else min_length
    max_l = max(lens) if max_length is None else max_length

    idx = [i for i in range(len(X)) if len(X[i]) > min_l+2 and len(X[i]) < max_l+2]
    X = [X[i] for i in idx]
    y = [y[i] for i in idx]

    return X, y

def set_balanced_pos_weight(dataset) :
    y = np.array(dataset.train_data.y)
    dataset.pos_weight = [len(y) / sum(y) - 1]

class DataHolder() :
  def __init__(self, X, y, padding_length, pad_token=0, batch_size=1):
    assert len(X) == len(y)
    self.padding_length = padding_length
    self.pad_token = pad_token
    self.batch_size = batch_size
    self.ix_max = len(X) // batch_size
    self.ix = 0
    self.X = X
    self.y = y
    self.attributes = ['X', 'y']

  def __iter__(self):
    return self

  def __next__(self):
    if self.ix >= self.ix_max:
      raise StopIteration
    
    X_padded = []
    X_unpadded_len= []
    X_max_length = max([len(self.X[self.ix + i]) for i in range(self.batch_size)]) # maxiumum unpadded length in a batch

    if self.padding_length == -1: # no limit on maximum sequence length
        trim = X_max_length
    else: 
        trim = min(X_max_length, self.padding_length)
    for i in range(self.batch_size):
      X_sample = self.X[self.ix + i][: trim]
      X_padded.append(X_sample + [self.pad_token] * (trim- len(X_sample)))
      X_unpadded_len.append(len(X_sample))
    
    X_batch = X_padded 
    y_batch = self.y[self.ix: self.ix + self.batch_size]
    self.ix += self.batch_size

    return torch.Tensor(X_batch), torch.Tensor(y_batch), torch.Tensor(X_unpadded_len)

  def get_stats(self, field='X') :
    assert field in self.attributes
    lens = [len(x) - 2 for x in getattr(self, field)]
    return {
        'min_length' : min(lens),
        'max_length' : max(lens),
        'mean_length' : np.mean(lens),
        'std_length' : np.std(lens)
    }
  
  def mock(self, n=200) :
    data_kwargs = { key: getattr(self, key)[:n] for key in self.attributes}
    return DataHolder(**data_kwargs)

  def filter(self, idxs) :
    data_kwargs = { key: [getattr(self, key)[i] for i in idxs] for key in self.attributes}
    return DataHolder(**data_kwargs)


class Dataset() :
  def __init__(self, name, vec, min_length=None, max_length=None, args=None) :
    self.name = name  
    self.vec = vec

    X, Xd, Xt = self.vec.seq_text['train'], self.vec.seq_text['dev'], self.vec.seq_text['test']
    y, yd, yt = self.vec.label['train'], self.vec.label['dev'], self.vec.label['test']

    X, y = filterbylength(X, y, min_length=min_length, max_length=max_length)
    Xt, yt = filterbylength(Xt, yt, min_length=min_length, max_length=max_length)
    Xt, yt = sortbylength(Xt, yt)
    
    Xd, yd = filterbylength(Xd, yd, min_length=min_length, max_length=max_length)
    Xd, yd = sortbylength(Xd, yd)

    self.train_data = DataHolder(X, y, args['padding_length'], args['padding_token'], args['batch_size'])
    self.dev_data = DataHolder(Xd, yd, args['padding_length'], args['padding_token'], args['batch_size'])
    self.test_data = DataHolder(Xt, yt, args['padding_length'], args['padding_token'], args['batch_size'])
    
    self.trainer_type = 'Single_Label'
    self.output_size = 1
    self.save_on_metric = 'roc_auc'
    self.keys_to_use = {
        'roc_auc' : 'roc_auc', 
        'pr_auc' : 'pr_auc'
    }

    self.bsize = 32
    if args is not None and hasattr(args, 'output_dir') :
      self.basepath = args.output_dir

  def display_stats(self) :
    stats = {}
    stats['vocab_size'] = self.vec.vocab_size
    stats['embed_size'] = self.vec.word_dim
    y = np.unique(np.array(self.train_data.y), return_counts=True)
    yt = np.unique(np.array(self.test_data.y), return_counts=True)

    stats['train_size'] = list(zip(y[0].tolist(), y[1].tolist()))
    stats['test_size'] = list(zip(yt[0].tolist(), yt[1].tolist()))
    stats.update(self.train_data.get_stats('X'))

    outdir = "datastats"
    os.makedirs('graph_outputs/' + outdir, exist_ok=True)

    json.dump(stats, open('graph_outputs/' + outdir + '/' + self.name + '.txt', 'w'))
    print(stats)

########################################## Dataset Loaders ################################################################################
def SST_dataset(vec=None, args=None) :
    dataset = Dataset(name='sst', vec=vec, min_length=5, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def CLS_dataset_en(vec=None, args=None) :
    dataset = Dataset(name='cls_en', vec=vec, min_length=5, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def CLS_dataset_de(vec=None, args=None) :
    dataset = Dataset(name='cls_de', vec=vec, min_length=5, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def CLS_dataset_fr(vec=None, args=None) :
    dataset = Dataset(name='cls_fr', vec=vec, min_length=5, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def CLS_dataset_jp(vec=None, args=None) :
    dataset = Dataset(name='cls_jp', vec=vec, min_length=5, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def IMDB_dataset(vec=None, args=None) :
    dataset = Dataset(name='imdb', vec=vec, min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def News20_dataset(vec=None, args=None) :
    dataset = Dataset(name='20News_sports', vec=vec, min_length=6, max_length=500, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def Yelp(vec=None, args=None) :
    dataset = Dataset(name='Yelp', vec=vec, min_length=6, max_length=150, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def Amazon(vec=None, args=None) :
    dataset = Dataset(name='Amazon', vec=vec, min_length=6, max_length=100, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def ADR_dataset(vec=None, args=None) :
    dataset = Dataset(name='tweet', vec=vec, min_length=5, max_length=100, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def Anemia_dataset(vec=None, args=None) :
    dataset = Dataset(name='anemia', vec=vec, max_length=4000, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def Diabetes_dataset(vec=None, args=None) :
    dataset = Dataset(name='diab', vec=vec, min_length=6, max_length=4000, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

datasets = {
    "sst" : SST_dataset,
    "cls_en": CLS_dataset_en,
    "cls_de": CLS_dataset_de,
    "cls_fr": CLS_dataset_fr,
    "cls_jp": CLS_dataset_jp,
    "imdb" : IMDB_dataset,
    'amazon': Amazon,
    'yelp': Yelp,
    "20News_sports" : News20_dataset,
    "tweet" : ADR_dataset,
    "Anemia" : Anemia_dataset,
    "Diabetes" : Diabetes_dataset,
}

def vectorize_data(data_file, min_df, word_vectors_type=None):
  vec = Vectorizer(min_df=min_df)

  df = pd.read_csv(data_file)
  
  assert 'text' in df.columns, "No Text Field"
  assert 'label' in df.columns, "No Label Field"
  assert 'exp_split' in df.columns, "No Experimental splits defined"

  texts = list(df[df.exp_split == 'train']['text'])
  vec.fit(texts)

  print("Vocabulary size : ", vec.vocab_size)

  vec.seq_text = {}
  vec.label = {}
  splits = df.exp_split.unique()
  
  for k in splits :
    split_texts = list(df[df.exp_split == k]['text'])
    vec.seq_text[k] = vec.get_seq_for_docs(split_texts)
    vec.label[k] = list(df[df.exp_split == k]['label'])
  
  vec.embeddings = None
  return vec

