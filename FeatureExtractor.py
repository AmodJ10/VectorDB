import torch
import torch.nn as nn



class FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(1,16,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16 , 32 , 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 , 16)
        )

    def forward(self , x):
        x = self.extractor(x)
        x = self.linear(x)
        return x
    
