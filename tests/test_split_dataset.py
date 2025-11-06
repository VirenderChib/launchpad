import os, json
from src.data_prep.split_dataset import split_dataset

def test_split_dataset_creates_output_dir():
    output_path = split_dataset()
    assert os.path.exists(output_path)
    assert all(os.path.exists(os.path.join(output_path, f"{name}.jsonl")) for name in ["train", "validation", "test"])

def test_split_dataset_files_contain_valid_json():
    output_path = "data/processed/splits"
    for name in ["train", "validation", "test"]:
        with open(os.path.join(output_path, f"{name}.jsonl"), "r", encoding="utf-8") as f:
            line = f.readline()
            assert json.loads(line) is not None

def test_split_dataset_returns_correct_path():
    path = split_dataset()
    assert path.endswith("data/processed/splits")
