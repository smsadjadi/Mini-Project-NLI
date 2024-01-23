

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') ; print('device:',device)

def accuracy(pred, y):
    max_preds = pred.argmax(dim = 1, keepdim = True)
    correct = (max_preds.squeeze(1)==y).float()
    return correct.sum() / len(y)

def show_cm(model, tokenizer, loader, device, labels=None):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            data = tokenizer(batch['premise'], batch['hypothesis'], truncation=True, padding="max_length", max_length=150, return_tensors="pt")
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            token_type_ids = data["token_type_ids"].to(device)
            label_id = batch['label_id'].to(device)
            now_batch_size = label_id.size(0)
            probs = nn.functional.softmax(model(batch, tokenizer),dim=1)
            outputs = torch.argmax(probs, axis=1)
            y_true.extend(label_id.tolist())
            y_pred.extend(outputs.tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm

def plot_cm(cm, labels):
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labels)
    cm_display.plot()
    plt.show() 

