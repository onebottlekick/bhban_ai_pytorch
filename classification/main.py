import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import HealthData
from models import FeedForward
from utils import evaluate, train, plot_graph, EarlyStopping

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
TEST_RATIO = 0.8
NUM_EPOCHS = 1000


dataset = HealthData()
train_data, val_data = random_split(dataset, (int(len(dataset)*TEST_RATIO), len(dataset) - int(len(dataset)*TEST_RATIO)))

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

model = FeedForward().to(DEVICE)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters())

early_stopping = EarlyStopping(verbose=True)

train_losses = []
train_accs = []
val_losses = []
val_accs = []
for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, DEVICE)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print('-'*30)
    print(f'Epoch [{epoch+1:02}/{NUM_EPOCHS}]')
    print()
    print(f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}')
    print(f'Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.2f}')

    
    early_stopping(val_loss, model)
    print('-'*30)
    if early_stopping.early_stop:
        print('Early stopping')
        break
    
plot_graph(train_losses, val_losses, train_accs, val_accs)