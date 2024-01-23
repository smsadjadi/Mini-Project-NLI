
import sys
import torch
import subprocess
from random import randint
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

_, _, df_test = ReadDataFrame()
test_dataset = Dataset(df_test)

bert_model = AutoModel.from_pretrained("HooshvareLab/bert-base-parsbert-uncased") # BertModel.from_pretrained("bert-base-multilingual-cased")
tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased") # BertTokenizer.from_pretrained('bert-base-multilingual-cased')
config = bert_model.config
config.num_hidden_layers = 12
config.num_attention_heads = 12
bert_model = BertModel(config)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

device = torch.device("cpu")
model = BERTNLIModel(bert_model, hidden_dim=512, output_dim=3).to(device)
model.load_state_dict(torch.load("bert-nli.pt",map_location=device)) ; model.eval()

sententce = df_test.iloc[randint(0,len(df_test))]
lbl = {0:'c', 1:'e', 2:'n'}
predictions = model(sententce, tokenizer).argmax(dim = 1, keepdim = True).detach().to('cpu').numpy()
pred = lbl[int(predictions)]

print('Premise:    ',sententce['premise'])
print('Hypothesis: ',sententce['hypothesis'])
print('True label: ',sententce['label'])
print('Pred label: ',pred)

