import os, torch
import pytest
import transformers
from src.data_prep.tokenize_dataset import tokenize_dataset

def test_tokenizer_loads_successfully(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "dummy_token")
    tokenize_dataset("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    assert os.path.exists("data/processed/tokenized/train.pt")

def test_missing_token_raises_error(monkeypatch):
    # Disable python-dotenv during this test
    monkeypatch.setenv("PYTHON_DOTENV_DISABLE", "1")
    # Ensure the var is absent/empty
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "")
    with pytest.raises(ValueError):
        tokenize_dataset()

def test_tokenized_files_created():
    for split in ["train", "validation", "test"]:
        path = f"data/processed/tokenized/{split}.pt"
        assert os.path.exists(path)

def test_tokenized_data_structure():
    with torch.serialization.safe_globals([transformers.tokenization_utils_base.BatchEncoding]):
        sample = torch.load("data/processed/tokenized/train.pt")
    assert isinstance(sample, list)
    first = sample[0]
    # If you implemented fix A (save dicts), this will be dict already.
    # If not, BatchEncoding behaves like a dict; access via keys works too.
    assert "input_ids" in first and "attention_mask" in first