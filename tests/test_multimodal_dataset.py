import os
import json
import pytest
from src.multimodal.multimodal_dataset import (
    save_image_from_url, load_and_filter_coco, save_splits
)

# ---------- TEST 1: Image saving (mocked requests) ----------
def test_save_image_from_url_success(monkeypatch, tmp_path):
    """Simulate successful image download and saving."""
    import requests
    from PIL import Image
    from io import BytesIO

    # Create fake image bytes
    img = Image.new("RGB", (10, 10), color="blue")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    fake_content = buf.getvalue()

    class DummyResponse:
        def __init__(self): self.content = fake_content
        def raise_for_status(self): return None

    monkeypatch.setattr(requests, "get", lambda url, timeout=30: DummyResponse())

    img_path = tmp_path / "test_img.jpg"
    result = save_image_from_url("http://dummy.url/image.jpg", img_path)
    assert result is True
    assert os.path.exists(img_path)

# ---------- TEST 2: Image saving failure ----------
def test_save_image_from_url_failure(monkeypatch, tmp_path):
    """Simulate failed download."""
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **kw: (_ for _ in ()).throw(Exception("Network error")))
    result = save_image_from_url("http://bad.url/image.jpg", tmp_path / "fail.jpg", retries=1)
    assert result is False

# ---------- TEST 3: save_splits creates 3 JSONL files ----------
def test_save_splits_creates_files(tmp_path):
    """Ensure save_splits() correctly writes 3 files."""
    mock_data = [
        {"id": str(i), "instruction": f"instr{i}", "response": "", "image": f"/img/{i}.jpg"}
        for i in range(10)
    ]
    out_dir = tmp_path / "processed"
    save_splits(mock_data, output_dir=out_dir)

    for name in ["train_mm.jsonl", "val_mm.jsonl", "test_mm.jsonl"]:
        path = out_dir / name
        assert path.exists(), f"{name} file not created"
        with open(path) as f:
            line = json.loads(f.readline())
            assert "instruction" in line
            assert "image" in line

# ---------- TEST 4: load_and_filter_coco structure (mocked dataset) ----------
def test_load_and_filter_coco_structure(monkeypatch, tmp_path):
    """Mock COCO dataset to check output structure without real download."""
    from PIL import Image

    # Mock save_image_from_url -> always True
    monkeypatch.setattr("src.multimodal.prepare_multimodal_dataset.save_image_from_url", lambda *a, **kw: True)

    # Mock load_dataset -> iterable of dummy dicts
    def fake_load_dataset(*a, **kw):
        class DummySet(list):
            def __iter__(self_inner): return iter([
                {"question": "What is this?", "answer": ["A cat"], "coco_url": "http://fake.img"},
                {"question": "Where is the dog?", "answer": ["On grass"], "coco_url": "http://fake2.img"},
            ])
        return DummySet()
    monkeypatch.setattr("src.multimodal.prepare_multimodal_dataset.load_dataset", fake_load_dataset)

    data = load_and_filter_coco(limit=2, img_save_dir=tmp_path)
    assert isinstance(data, list)
    assert len(data) == 2
    assert all("instruction" in d and "image" in d for d in data)
