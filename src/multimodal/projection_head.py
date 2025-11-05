import torch
from torch import nn

class ProjectionHead(nn.Module):
    def __init__(self, vision_dim=512, text_dim=2048, hidden_dim=768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, text_dim)
        )

    def forward(self, x):
        return self.proj(x)
