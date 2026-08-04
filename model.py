import torch
import torch.nn as nn
import torch.nn.functional as F

class basic_model(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,2)
        )

    def forward(self , x):
        return self.model(x)
