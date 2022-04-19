import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, in_features=3, out_features=3, dropout=None):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, 3)
        self.fc2 = nn.Linear(3, 128)
        self.fc3 = nn.Linear(128, out_features)
        
        self.relu = nn.ReLU(True)
        self.softmax = nn.LogSoftmax(dim=1)
        
        if dropout:
            self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        
        if self.dropout:
            x = self.dropout(x)
            
        x = self.fc3(x)
        x = self.softmax(x)
        
        return x