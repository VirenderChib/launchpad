from src.data_prep.filter_dataset import filter_dataset
from datasets import Dataset
import os

def test_filter_dataset_loads_and_returns_dataset():
    ds = filter_dataset()
    assert isinstance(ds, Dataset)
    assert len(ds) > 0

def test_filtered_dataset_is_smaller():
    ds = filter_dataset()
    full_size = 20000  # Approx known size of CodeAlpaca dataset
    assert len(ds) <= full_size

def test_filtered_dataset_file_created():
    path = "data/processed/codealpaca_tech_filtered.json"
    assert os.path.exists(path)
    assert path.endswith(".json")
