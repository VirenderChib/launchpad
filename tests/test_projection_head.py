import torch
import pytest
from src.multimodal.projection_head import ProjectionHead

# ---------- TEST 1: Initialization ----------
def test_projection_head_initializes_correctly():
    """Ensure ProjectionHead initializes with correct layers."""
    model = ProjectionHead(vision_dim=512, text_dim=2048, hidden_dim=768)
    assert isinstance(model.proj[0], torch.nn.Linear)
    assert isinstance(model.proj[1], torch.nn.ReLU)
    assert isinstance(model.proj[2], torch.nn.Linear)
    assert model.proj[0].in_features == 512
    assert model.proj[2].out_features == 2048

# ---------- TEST 2: Forward Pass Shape ----------
def test_projection_head_forward_shape():
    """Forward pass produces correct output shape."""
    model = ProjectionHead(vision_dim=512, text_dim=2048, hidden_dim=768)
    x = torch.randn(4, 512)  # batch_size=4, vision_dim=512
    out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (4, 2048)

# ---------- TEST 3: Forward Pass Grad Flow ----------
def test_projection_head_allows_backprop():
    """Model parameters should have gradients after backward()."""
    model = ProjectionHead()
    x = torch.randn(2, 512)
    out = model(x)
    loss = out.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)

# ---------- TEST 4: Handles Wrong Input Dim ----------
def test_projection_head_wrong_input_dimension():
    """Passing wrong input dimension should raise RuntimeError."""
    model = ProjectionHead()
    bad_input = torch.randn(1, 256)  # incorrect feature size
    with pytest.raises(RuntimeError):
        _ = model(bad_input)

# ---------- TEST 5: Works on CPU ----------
def test_projection_head_runs_on_cpu():
    """Ensure model can run purely on CPU (no CUDA required)."""
    model = ProjectionHead()
    x = torch.randn(3, 512)
    with torch.no_grad():
        out = model(x)
    assert out.device.type == "cpu"
