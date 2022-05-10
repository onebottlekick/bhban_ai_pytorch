import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import WeatherDataset
from model import LSTM
from utils import EarlyStopping, evaluate, plot_graph, train


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
TEST_RATIO = 0.8
NUM_EPOCHS = 50

dataset = WeatherDataset(normalize=True)
train_data, val_data = random_split(dataset, (int(len(dataset)*TEST_RATIO), len(dataset) - int(len(dataset)*TEST_RATIO)))

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

model = LSTM().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()

early_stopping = EarlyStopping(verbose=True)

train_losses = []
val_losses = []
for epoch in range(NUM_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion, DEVICE)
    train_losses.append(train_loss)
    val_loss = evaluate(model, val_loader, criterion, DEVICE)
    val_losses.append(val_loss)
    
    print('-'*30)
    print(f'Epoch [{epoch+1:02}/{NUM_EPOCHS}]')
    print()
    print(f'Train Loss: {train_loss:.3f}')
    print(f'Val Loss: {val_loss:.3f}')


    early_stopping(val_loss, model)
    print('-'*30)
    if early_stopping.early_stop:
        print('Early stopping')
        break
    
plot_graph(train_losses, val_losses)
