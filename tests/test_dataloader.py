import os
import torch
import pytest
from datasets import Dataset
from src.multimodal.data_loader import load_multimodal_datasets

@pytest.fixture
def dummy_pt_files(tmp_path):
    """Create dummy .pt files for train, validation, test."""
    def make_sample(n=2):
        return [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "pixel_values": torch.randn(3, 32, 32),
                "caption": "test caption",
            }
            for _ in range(n)
        ]

    train_path = tmp_path / "train_mm.pt"
    val_path = tmp_path / "validation_mm.pt"
    test_path = tmp_path / "test_mm.pt"

    torch.save(make_sample(), train_path)
    torch.save(make_sample(), val_path)
    torch.save(make_sample(), test_path)

    return tmp_path

def test_load_multimodal_datasets_returns_datasets(dummy_pt_files):
    """Verify that all three splits load correctly and return Dataset objects."""
    train_ds, val_ds, test_ds = load_multimodal_datasets(dummy_pt_files)

    # 👇 Force dataset map results to materialize (apply transformations)
    train_ds = train_ds.map(lambda ex: ex)
    val_ds = val_ds.map(lambda ex: ex)
    test_ds = test_ds.map(lambda ex: ex)

    assert isinstance(train_ds, Dataset)
    assert isinstance(val_ds, Dataset)
    assert isinstance(test_ds, Dataset)

    assert len(train_ds) > 0
    assert "input_ids" in train_ds[0]
    assert isinstance(train_ds[0]["input_ids"], torch.Tensor)


def test_load_multimodal_datasets_tensor_conversion(dummy_pt_files):
    """Ensure input_ids and attention_mask are converted to tensors."""
    train_ds, _, _ = load_multimodal_datasets(dummy_pt_files)

    # 👇 Force realization before checking types
    train_ds = train_ds.map(lambda ex: ex)

    example = train_ds[0]
    assert isinstance(example["input_ids"], torch.Tensor)
    assert example["input_ids"].dtype == torch.long
    assert isinstance(example["attention_mask"], torch.Tensor)


def test_load_multimodal_datasets_missing_file(tmp_path):
    """Raise FileNotFoundError when one of the .pt files is missing."""
    # Only save one file
    torch.save([], tmp_path / "train_mm.pt")
    with pytest.raises(FileNotFoundError):
        _ = torch.load(os.path.join(tmp_path, "validation_mm.pt"))
