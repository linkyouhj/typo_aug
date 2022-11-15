import torch
from torch import nn
import os
import sys
from transformers import BertModel
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

class EarlyStopping():
    def __init__(self, patience=3, path='./checkpoint.pt'):
        self.patience = patience
        self.count = 0
        self.best_acc = 0
        self.early_stop = False
        self.path = path

    def __call__(self, acc, model):
        if acc >= self.best_acc:
            self.best_acc = acc
            self.count = 0

            torch.save(model, self.path)

            self.early_stop = False
        else:
            self.count += 1

            if self.count == self.patience:
                self.early_stop = True
        
        return self.early_stop

def get_model(train_dataloader, model_name, device, learning_rate, num_epochs, warmup_ratio,num_classes, early_stop_patience, path):
    bert_model = BertModel.from_pretrained(model_name)
    model = BERTClassifier(bert_model, num_classes=num_classes, dr_rate=0.5).to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    t_total = len(train_dataloader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
    early_stopping_callback = EarlyStopping(patience=early_stop_patience, path=path+"model.pt")

    return model, optimizer, loss_fn, scheduler, early_stopping_callback


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes = 2, 
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, 
                              token_type_ids = segment_ids.long(), 
                              attention_mask = attention_mask.float().to(token_ids.device),
                              return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)