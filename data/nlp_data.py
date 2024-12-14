import os
from sklearn.model_selection import train_test_split

import torch
from torchtext.datasets import CoLA, AG_NEWS
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

_NLP_DATASETS = {}

def _add_dataset(dataset_fn):
    _NLP_DATASETS[dataset_fn.__name__] = dataset_fn
    return dataset_fn

class NLPDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = [feat if isinstance(feat, torch.Tensor) else torch.tensor(feat,dtype=torch.float32) for feat in features]
        self.labels = [lab if isinstance(lab, torch.Tensor) else torch.tensor(lab,dtype=torch.long) for lab in labels]
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx].squeeze(), self.labels[idx].squeeze()

@_add_dataset
def cola(root):
    train_dataset, test_dataset = CoLA(root=root, split=('train', 'test'))

    train_sentences = [label[2] for label in train_dataset]
    train_labels = [label[1] for label in train_dataset]
    test_sentences = [label[2] for label in test_dataset]
    test_labels = [label[1] for label in test_dataset]

    vectorizer = TfidfVectorizer(max_features=10000)  # 你可以根据需要调整最大特征数
    X_train = vectorizer.fit_transform(train_sentences).toarray()
    X_test = vectorizer.transform(test_sentences).toarray()

    train_set = NLPDataset(X_train,train_labels)
    test_set = NLPDataset(X_test,test_labels)
    return train_set, test_set

@_add_dataset
def ag_news(root):
    train_dataset, test_dataset = AG_NEWS(root=root, split=('train', 'test'))

    train_sentences = [label[1] for label in train_dataset]
    train_labels = [label[0] - 1 for label in train_dataset]

    test_sentences = [label[1] for label in test_dataset]
    test_labels = [label[0] - 1 for label in test_dataset]

    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(train_sentences).toarray()
    X_test = vectorizer.transform(test_sentences).toarray()

    train_set = NLPDataset(X_train,train_labels)
    test_set = NLPDataset(X_test,test_labels)
    return train_set, test_set

@_add_dataset
def spam(root):
    try:
        dataset = pd.read_csv(f'{root}/SMSSpamCollection.tsv',sep='\t')
    except FileNotFoundError:
        raise FileNotFoundError("You need to download the csv from github in advance.")
    sentences = list(dataset['sentence'])
    labels = (dataset['label'] == 'ham').astype(int).values

    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(sentences).toarray()

    train_set = NLPDataset(X,labels)
    return train_set, None

def get_nlp_dataset(dataset_name, root=None):
    if root is None:
        root = 'data'
    return _NLP_DATASETS[dataset_name](root)