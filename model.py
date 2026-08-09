import torch
import torch.nn as nn
import torch.nn.functional as F

class basic_model(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,2),
        )

    def forward(self , x):
        return self.model(x)
