import os

import matplotlib.pyplot as plt
import torch


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_epoch_loss = 0
    train_epoch_acc = 0
    for idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)

        pred = model(inputs)

        loss = criterion(pred, targets)

        loss.backward()
        optimizer.step()

        train_epoch_loss += loss.item()
        train_epoch_acc += ((pred > 0.5) == targets).sum().item()

    return train_epoch_loss/len(train_loader), train_epoch_acc/len(train_loader.dataset)


def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_epoch_loss = 0
    val_epoch_acc = 0
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            pred = model(inputs)

            loss = criterion(pred, targets)

            val_epoch_loss += loss.item()
            val_epoch_acc += ((pred > 0.5) == targets).sum().item()

    return val_epoch_loss/len(val_loader), val_epoch_acc/len(val_loader.dataset)


def plot_graph(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(train_losses, 'red', label='train')
    plt.plot(val_losses, 'blue', label='val')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(train_accs, 'red', label='train')
    plt.plot(val_accs, 'blue', label='val')
    plt.legend()
    plt.show()


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, save_path='checkpoints/cp.pt'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.save_path = save_path

        self.counter = 0
        self.best_score = None
        self.val_loss = float('inf')
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
