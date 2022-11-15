import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tokenization_kobert import KoBertTokenizer
import pandas as pd
import os
import re
from soynlp.normalizer import repeat_normalize
import emoji


def encode_label(df,classes):
  df["encoded_label"] = df["label"].map(lambda x: classes[x])

def clean(x):
    # emojis = ''.join(emoji.EMOJI_DATA.keys())
    # pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣{emojis}]+')
    # url_pattern = re.compile(
    #   r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
    # x = pattern.sub(' ', x)
    # x = url_pattern.sub('', x)
    x = x.strip()
    x = repeat_normalize(x, num_repeats=2)
    return x

def get_dataset(model_name, sep, path, train_name, test_name, max_length, batch_size, sentence_column, classes):
    tokenizer = KoBertTokenizer.from_pretrained(model_name) # monologg/distilkobert도 동일
    df_train = pd.read_csv(path+train_name, sep=sep)
    df_test = pd.read_csv(path+test_name, sep=sep)
    # df_train.drop(columns=["id"],inplace = True)
    # df_test.drop(columns=["id"],inplace = True)

    # remove Nan data 
    
    df_train = df_train.dropna(axis=0).reset_index(drop=True)
    df_test = df_test.dropna(axis=0).reset_index(drop=True)
    df_train["clean_sentence"] = df_train[sentence_column].map(lambda x : clean(x))
    df_test["clean_sentence"] = df_test[sentence_column].map(lambda x : clean(x))  
    encode_label(df_train,classes)
    encode_label(df_test,classes)
    # tokenize sentence
    df_train["tokenized"] = df_train["clean_sentence"].map(lambda x: tokenizer(x, padding='max_length', truncation=True, max_length=max_length))
    df_test["tokenized"] = df_test["clean_sentence"].map(lambda x: tokenizer(x, padding='max_length', truncation=True, max_length=max_length))

    train_dataset = BERTDataset(df_train["tokenized"], df_train['encoded_label'],max_length)
    test_dataset = BERTDataset(df_test["tokenized"], df_test['encoded_label'],max_length)

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, num_workers=0, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, num_workers=0,shuffle=True)

    return train_dataloader, test_dataloader

def get_testset(path, sep, max_length,model_name,batch_size):
  tokenizer = KoBertTokenizer.from_pretrained(model_name) # monologg/distilkobert도 동일
  df_pred = pd.read_csv(path+"/data/test.hate.no_label.csv", sep=sep)
  df_pred["tokenized"] = df_pred["comments"].map(lambda x: tokenizer(x, padding='max_length', truncation=True, max_length=max_length))
  pred_data = TestDataset(df_pred["tokenized"],max_length)
  pred_dataloader = DataLoader(pred_data,batch_size = batch_size,num_workers=0,shuffle=False)
  return df_pred, pred_dataloader
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

class TestDataset(Dataset):
  def __init__(self, dataset, max_length):
    self.dataset = dataset
    self.max_length = max_length
  def __getitem__(self,idx):
    return (np.array(self.dataset[idx]["input_ids"]),
            self.max_length, 
            np.array(self.dataset[idx]["token_type_ids"]), 
            )

  def __len__(self):
    return len(self.dataset)