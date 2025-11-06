from src.data_prep.inspect_dataset import inspect_dataset
from datasets import DatasetDict

def test_dataset_loads_successfully():
    data = inspect_dataset()
    assert isinstance(data, DatasetDict)
    assert "train" in data
    assert len(data["train"]) > 0

def test_dataset_has_required_columns():
    data = inspect_dataset()
    cols = data["train"].column_names
    expected = {"instruction", "input", "output"}
    assert expected.issubset(set(cols))

def test_dataset_examples_non_empty():
    data = inspect_dataset()
    sample = data["train"][0]
    assert all(sample[key].strip() != "" for key in ["instruction", "output"])
