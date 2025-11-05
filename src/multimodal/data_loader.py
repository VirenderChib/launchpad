import os
import torch
from datasets import Dataset

def load_multimodal_datasets(base_dir):
    train = torch.load(os.path.join(base_dir, "train_mm.pt"))
    val = torch.load(os.path.join(base_dir, "validation_mm.pt"))
    test = torch.load(os.path.join(base_dir, "test_mm.pt"))

    def to_tensor(ex):
        ex["input_ids"] = torch.tensor(ex["input_ids"], dtype=torch.long)
        ex["attention_mask"] = torch.tensor(ex["attention_mask"], dtype=torch.long)
        return ex

    train_ds = Dataset.from_list(train).map(to_tensor)
    val_ds = Dataset.from_list(val).map(to_tensor)
    test_ds = Dataset.from_list(test).map(to_tensor)

    return train_ds, val_ds, test_ds
