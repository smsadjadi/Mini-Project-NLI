

import sys
import torch
import torch.nn as nn

sys.path.append('../')
from EDA import *

class BERTNLIModel(nn.Module):
  def __init__(self, bert_model, hidden_dim=250, output_dim=3):
      super().__init__()
      self.bert = bert_model
      embedding_dim = bert_model.config.to_dict()['hidden_size']
      self.head = nn.Sequential(
          nn.Linear(embedding_dim, output_dim))
  def forward(self, batch, tokenizer):
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') ; print('device:',device)
      encoded_input = tokenizer(batch['premise'], batch['hypothesis'], truncation=True, padding="max_length", max_length=150, return_tensors="pt").to(device)
      embedded = self.bert(**encoded_input)[1]
      output = self.head(embedded)
      return output

