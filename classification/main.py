import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import HealthData
from models import FeedForward
from utils import evaluate, train, plot_loss

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
TEST_RATIO = 0.8
NUM_EPOCHS = 20


dataset = HealthData()
train_data, val_data = random_split(dataset, (int(len(dataset)*TEST_RATIO), len(dataset) - int(len(dataset)*TEST_RATIO)))

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

model = FeedForward().to(DEVICE)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters())

train_losses = []
val_losses = []
for epoch in range(NUM_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion, epoch, DEVICE)
    train_losses.append(train_loss)
    val_loss = evaluate(model, val_loader, criterion, DEVICE)
    val_losses.append(val_loss)
    
    print(f'Epoch {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\tVal Loss: {val_loss:.3f}')
    
plot_loss(train_losses, val_losses)