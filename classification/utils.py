import torch


def train(model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    train_epoch_loss = 0
    for idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
                
        pred = model(inputs)
        
        loss = criterion(pred, targets)
        
        loss.backward()        
        optimizer.step()
        
        train_epoch_loss += loss.item()
    
    return train_epoch_loss/len(train_loader)


def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_epoch_loss = 0
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            pred = model(inputs)
            
            loss = criterion(pred, targets)
            
            val_epoch_loss += loss.item()
    
    return val_epoch_loss/len(val_loader)