

import torch

class Dataset(torch.utils.data.Dataset):
  def __init__(self, df):
      self.df = df
  def __len__(self):
      return len(self.df)
  def __getitem__(self, idx):
      row = self.df.iloc[idx]
      premise = row["premise"]
      hypothesis = row["hypothesis"]
      label = row["label"]
      label_id = row["label_id"]
      return {"premise": premise, "hypothesis": hypothesis, "label": label, "label_id": label_id}

