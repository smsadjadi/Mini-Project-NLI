
import sys
import math
import torch
import subprocess
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from transformers.optimization import *
from sklearn.preprocessing import LabelEncoder
subprocess.check_call([sys.executable,'-m','pip','install','transformers']) # !pip install transformers
from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer
import transformers

from EDA import *
from dataloaders import *
from nets import *
from learning import *
from losses import *
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') ; print('device:',device)

df_train, df_dev, _ = ReadDataFrame()
train_dataset = Dataset(df_train)
val_dataset = Dataset(df_dev)

bert_model = AutoModel.from_pretrained("HooshvareLab/bert-base-parsbert-uncased") # BertModel.from_pretrained("bert-base-multilingual-cased")
tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased") # BertTokenizer.from_pretrained('bert-base-multilingual-cased')
config = bert_model.config
config.num_hidden_layers = 12
config.num_attention_heads = 12
bert_model = BertModel(config)

model = BERTNLIModel(bert_model, hidden_dim=250, output_dim=3)
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-6, correct_bias=False)
criterion = nn.CrossEntropyLoss()

N_EPOCHS = 5
warmup_percent = 0.2
batch_size = 16
train_data_len = len(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

total_steps = math.ceil(N_EPOCHS * train_data_len * 1./batch_size)
warmup_steps = int(total_steps*warmup_percent)
scheduler = get_constant_schedule_with_warmup(optimizer,num_warmup_steps=warmup_percent)
best_valid_loss = float('inf')

report = {'train_loss':[],'train_acc':[],'valid_loss':[],'valid_acc':[]}
label_encoder = LabelEncoder() ; label_encoder.fit(df_train['label'])
transformers.logging.set_verbosity_error()

for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, tokenizer, train_loader, optimizer, criterion, scheduler)
    valid_loss, valid_acc = evaluate(model, tokenizer, val_loader, criterion)
    report['train_loss'].append(train_loss) ; report['train_acc'].append(train_acc)
    report['valid_loss'].append(valid_loss) ; report['valid_acc'].append(valid_acc)
    if valid_loss < best_valid_loss:
      best_valid_loss = valid_loss
      torch.save(model.state_dict(),'bert-nli.pt') 
    print(f'Epoch {epoch+1}: Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f} | Valid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:.2f}%')

report = pd.DataFrame(report) ; report.to_csv("report.csv",index=False)

PlotReport(report)
cm = show_cm(model, tokenizer, test_loader, device)
plot_cm(cm, label_encoder.classes_)

