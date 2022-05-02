import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import display


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for idx, (imgs, annos) in enumerate(train_loader):
        imgs = imgs.to(device)
        annos = annos.to(device)
        
        optimizer.zero_grad()        
        outputs = model(imgs)
        loss = criterion(outputs, annos)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
    return epoch_loss/len(train_loader)


def evaluate(model, val_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for idx, (imgs, annos) in enumerate(val_loader):
            imgs = imgs.to(device)
            annos = annos.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, annos)
            
            epoch_loss += loss.item()
            
    return epoch_loss/len(val_loader)


def plot(img, train_losses, val_losses):
    with plt.ion():
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()

        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow(np.random.randn(28, 28))

        plt.subplot(1, 2, 2)
        plt.xlabel('steps')
        plt.ylabel('Epochs')
        plt.plot(train_losses, label='train', color='blue')
        plt.text(len(train_losses) - 1, train_losses[-1], str(train_losses[-1]))
        plt.plot(val_losses, label='val', color='red')
        plt.text(len(val_losses) - 1, val_losses[-1], str(val_losses[-1]))
        plt.legend()
        plt.show()

        plt.pause(0.1)


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, save_path='checkpoints/cp.pt'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.save_path = save_path
        
        self.counter = 0
        self.best_score = None
        self.val_loss = np.Inf
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self._save_model(val_loss, model)
        
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EearlyStopping counter [{self.counter}/{self.patience}]')
            if self.counter >= self.patience:
                self.early_stop = True
                
        else:
            self.best_score = score
            self._save_model(val_loss, model)
            self.counter = 0
            
    def _save_model(self, val_loss, model):
        os.makedirs(self.save_path.split('/')[0], exist_ok=True)
        if self.verbose:
            print(f'Val loss decreased ({self.val_loss:.6f} --> {val_loss:.6f}).')
            print('Saved model.')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss = val_loss
