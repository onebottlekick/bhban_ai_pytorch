import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, 7)
        self.fc2 = nn.Linear(7, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)
        
        self.sigmoid = nn.LogSigmoid()
        self.relu = nn.ReLU(True)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        # x = self.sigmoid(x)
        
        return x