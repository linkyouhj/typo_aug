import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tokenization_kobert import KoBertTokenizer
import pandas as pd
import os


def get_dataset(model_name, sep, path, train_name, test_name, max_length, batch_size):
    tokenizer = KoBertTokenizer.from_pretrained(model_name) # monologg/distilkobert도 동일
    df_train = pd.read_csv(path+train_name, sep=sep)
    df_test = pd.read_csv(path+test_name, sep=sep)
    df_train.drop(columns=["id"],inplace = True)
    df_test.drop(columns=["id"],inplace = True)

    # remove Nan data 
    df_train = df_train.dropna(axis=0).reset_index(drop=True)
    df_test = df_test.dropna(axis=0).reset_index(drop=True)

    # tokenize sentence
    df_train["tokenized"] = df_train["document"].map(lambda x: tokenizer(x, padding='max_length', truncation=True, max_length=max_length))
    df_test["tokenized"] = df_test["document"].map(lambda x: tokenizer(x, padding='max_length', truncation=True, max_length=max_length))

    train_dataset = BERTDataset(df_train["tokenized"], df_train['label'],max_length)
    test_dataset = BERTDataset(df_test["tokenized"], df_test['label'],max_length)

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, num_workers=0, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, num_workers=0,shuffle=True)

    return train_dataloader, test_dataloader

class BERTDataset(Dataset):
  def __init__(self, dataset, label, max_length):
    self.dataset = dataset
    self.label = label
    self.max_length = max_length
  def __getitem__(self,idx):
    return (np.array(self.dataset[idx]["input_ids"]),
            self.max_length, 
            np.array(self.dataset[idx]["token_type_ids"]), 
            self.label[idx])

  def __len__(self):
    return len(self.label)

