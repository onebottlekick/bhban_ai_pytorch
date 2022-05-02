import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from dataset import PetDataset
from model import UNet
from utils import EarlyStopping, evaluate, plot, train


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
TEST_RATIO = 0.8
NUM_EPOCHS = 100

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128))
])

dataset = PetDataset(transform=transform)

train_data, val_data = random_split(dataset, (int(len(dataset)*TEST_RATIO), len(dataset) - int(len(dataset)*TEST_RATIO)))
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

model = UNet().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

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
    print(f'Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}')
    early_stopping(val_loss, model)
    print('-'*30)
    
    # TODO make result image
    plot(None, train_losses, val_losses)
    
    if early_stopping.early_stop:
        print('Early stopping')
        break
