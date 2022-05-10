import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, in_features=14, out_features=14, hidden_size=64):
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(in_features, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, 32)
        self.fc2 = nn.Linear(32, out_features)
        
    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        x = self.fc(output)
        x = self.fc2(x)
        
        return x
    
    
if __name__ == '__main__':
    x = torch.randn(10, 12, 14)
    model = LSTM()
    assert model(x).shape == (10, 12, 14)
    