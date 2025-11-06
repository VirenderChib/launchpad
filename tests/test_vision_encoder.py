import pytest
import torch
from PIL import Image
from pathlib import Path
from src.multimodal.vision_encoder import VisionEncoder

# ---------- TEST 1: Initialization ----------
def test_vision_encoder_initializes(monkeypatch):
    """Checks if VisionEncoder initializes correctly with mocked model."""
    
    # Mock AutoProcessor and AutoModel to avoid actual CLIP download
    import transformers

    class DummyProcessor:
        def __call__(self, images, return_tensors="pt"):
            return {"pixel_values": torch.randn(1, 3, 32, 32)}

    class DummyModel:
        def eval(self): return self
        def __call__(self, **kwargs):
            return type("DummyOutput", (), {"last_hidden_state": torch.randn(1, 10, 512)})

    monkeypatch.setattr(transformers, "AutoProcessor", lambda *a, **kw: DummyProcessor())
    monkeypatch.setattr(transformers, "AutoModel", lambda *a, **kw: DummyModel())

    encoder = VisionEncoder()
    assert hasattr(encoder, "processor")
    assert hasattr(encoder, "model")
    assert callable(encoder.encode_image)

# ---------- TEST 2: Encode valid image ----------
def test_encode_image_returns_tensor(tmp_path, monkeypatch):
    """Ensures encode_image returns a 2D tensor."""
    
    img_path = tmp_path / "dummy.jpg"
    Image.new("RGB", (64, 64), color="red").save(img_path)

    # Mock model and processor to avoid downloading CLIP
    import transformers
    class DummyProcessor:
        def __call__(self, images, return_tensors="pt"):
            return {"pixel_values": torch.randn(1, 3, 32, 32)}

    class DummyModel:
        def eval(self): return self
        def __call__(self, **kwargs):
            return type("DummyOutput", (), {"last_hidden_state": torch.randn(1, 5, 512)})

    monkeypatch.setattr(transformers, "AutoProcessor", lambda *a, **kw: DummyProcessor())
    monkeypatch.setattr(transformers, "AutoModel", lambda *a, **kw: DummyModel())

    encoder = VisionEncoder()
    output = encoder.encode_image(str(img_path))

    assert isinstance(output, torch.Tensor)
    assert output.ndim == 2
    assert output.shape[0] == 1

# ---------- TEST 3: Invalid path ----------
def test_invalid_image_path(monkeypatch):
    """Checks if invalid image path raises an OSError."""
    import transformers

    class DummyProcessor:
        def __call__(self, images, return_tensors="pt"):
            return {"pixel_values": torch.randn(1, 3, 32, 32)}

    class DummyModel:
        def eval(self): return self
        def __call__(self, **kwargs):
            return type("DummyOutput", (), {"last_hidden_state": torch.randn(1, 5, 512)})

    monkeypatch.setattr(transformers, "AutoProcessor", lambda *a, **kw: DummyProcessor())
    monkeypatch.setattr(transformers, "AutoModel", lambda *a, **kw: DummyModel())

    encoder = VisionEncoder()
    with pytest.raises(OSError):
        encoder.encode_image("nonexistent.jpg")
