import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_vocab):
        super(Model, self).__init__()
        
        self.embedding = nn.Embedding(n_vocab, 128)
        self.pooling = nn.AdaptiveAvgPool2d((1, 128))
        self.fc = nn.Linear(128, 32)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pooling(x)
        x = x.squeeze(1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x
    
    
if __name__ == '__main__':
    x = torch.randint(0, 8982, (10, 189))
    model = Model(8983)
    print(model(x).shape)