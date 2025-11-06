import os
import torch
import json
import pytest
from pathlib import Path
from PIL import Image

from src.multimodal.tokenize_mm import process_split

# ---------- FIXTURE HELPERS ----------
@pytest.fixture
def dummy_tokenizer():
    class DummyTokenizer:
        def __call__(self, text, **kwargs):
            return {
                "input_ids": [1, 2, 3, 4],
                "attention_mask": [1, 1, 1, 1]
            }
    return DummyTokenizer()

@pytest.fixture
def dummy_vision_proc():
    class DummyProcessor:
        def __call__(self, images, return_tensors="pt"):
            return {"pixel_values": torch.randn(1, 3, 32, 32)}
    return DummyProcessor()

@pytest.fixture
def dummy_jsonl(tmp_path):
    """Create a dummy multimodal JSONL file with image and caption fields."""
    img_path = tmp_path / "dummy.jpg"
    Image.new("RGB", (32, 32), color="blue").save(img_path)

    data = {
        "instruction": "Describe this image.",
        "response": "A small blue square.",
        "image": str(img_path)
    }
    file_path = tmp_path / "train_mm.jsonl"
    with open(file_path, "w") as f:
        f.write(json.dumps(data) + "\n")
    return file_path

# ---------- TEST 1 ----------
def test_process_split_creates_pt_file(tmp_path, dummy_tokenizer, dummy_vision_proc, dummy_jsonl, monkeypatch):
    """Verify process_split runs and creates .pt file."""
    from src.multimodal import tokenize_mm  # import inside test to access globals

    # Redirect INPUT_DIR and OUTPUT_DIR to tmp_path
    monkeypatch.setattr(tokenize_mm, "INPUT_DIR", str(tmp_path))
    monkeypatch.setattr(tokenize_mm, "OUTPUT_DIR", str(tmp_path))

    out_path = tmp_path / "train_mm.pt"

    tokenize_mm.process_split(
        split_name="train",
        in_fname=os.path.basename(dummy_jsonl),
        out_fname=os.path.basename(out_path),
        tokenizer=dummy_tokenizer,
        vision_proc=dummy_vision_proc,
        limit=None,
    )

    assert out_path.exists(), "Output .pt file was not created"
    data = torch.load(out_path)
    assert isinstance(data, list)
    assert "input_ids" in data[0] and "pixel_values" in data[0]

# ---------- TEST 2 ----------

def test_process_split_handles_missing_image(tmp_path, dummy_tokenizer, dummy_vision_proc, monkeypatch):
    """Should skip entries with missing image paths gracefully."""
    from src.multimodal import tokenize_mm

    # Patch directories to use temp dir
    monkeypatch.setattr(tokenize_mm, "INPUT_DIR", str(tmp_path))
    monkeypatch.setattr(tokenize_mm, "OUTPUT_DIR", str(tmp_path))

    bad_data = {"instruction": "Bad sample", "response": "Missing image"}
    file_path = tmp_path / "bad.jsonl"
    with open(file_path, "w") as f:
        f.write(json.dumps(bad_data) + "\n")

    out_path = tmp_path / "bad.pt"

    tokenize_mm.process_split(
        split_name="train",
        in_fname=os.path.basename(file_path),
        out_fname=os.path.basename(out_path),
        tokenizer=dummy_tokenizer,
        vision_proc=dummy_vision_proc,
    )

    data = torch.load(out_path)
    assert isinstance(data, list)
    assert len(data) == 0  # skipped entry since image missing


# ---------- TEST 3 ----------
def test_process_split_file_not_found(tmp_path, dummy_tokenizer, dummy_vision_proc):
    """Should raise FileNotFoundError if input JSONL not found."""
    with pytest.raises(FileNotFoundError):
        process_split(
            split_name="train",
            in_fname="does_not_exist.jsonl",
            out_fname="out.pt",
            tokenizer=dummy_tokenizer,
            vision_proc=dummy_vision_proc,
        )
