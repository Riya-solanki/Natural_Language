# Grammar_FineTune_Loader.py

from torch.utils.data import Dataset, DataLoader
import os
import torch

class GrammarDataset(Dataset):
    def __init__(self, data_dir, tokenizer):
        self.texts = []
        for file in os.listdir(data_dir):
            if file.endswith(".txt"):
                with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                    self.texts.append(f.read())
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": enc["input_ids"].squeeze(0),
        }
