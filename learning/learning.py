
import sys
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.append('../')
from utils import *

def train(model, tokenizer, loader, optimizer, criterion, scheduler):
  epoch_loss = 0 ; epoch_acc = 0
  model = model.to(device)
  criterion = criterion.to(device)
  model.train()
  for batch in loader:
      label = batch['label_id'].to(device)
      torch.cuda.empty_cache()
      optimizer.zero_grad()
      predictions = model(batch, tokenizer)
      loss = criterion(predictions, label)
      acc = accuracy(predictions, label)
      loss.backward()
      optimizer.step()
      scheduler.step()
      epoch_loss += loss.item()
      epoch_acc += acc.item()
  return epoch_loss / len(loader), epoch_acc / len(loader)

def evaluate(model, tokenizer, loader, criterion):
  epoch_loss = 0 ; epoch_acc = 0
  model = model.to(device)
  criterion = criterion.to(device)
  model.eval()
  with torch.no_grad():
    for batch in loader:
        label = batch['label_id'].to(device)
        torch.cuda.empty_cache()
        predictions = model(batch, tokenizer)
        loss = criterion(predictions, label)
        acc = accuracy(predictions, label)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
  return epoch_loss / len(loader), epoch_acc / len(loader)

