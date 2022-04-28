import os

import torch
from torchvision import transforms
from torchvision.utils import save_image

from models import VGG
from utils import load_img, plot, img2gif


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VGG().to(device)

content_img = load_img('content.jpg', device=device)
style_img = load_img('style.jpg', device=device)
style_img = transforms.Resize(content_img.shape[2:])(style_img)

transfered = content_img.clone().requires_grad_(True)

num_epochs = 400
learning_rate = 0.001

alpha = 1
beta = 0.01

optimizer = torch.optim.Adam([transfered], lr=learning_rate)

losses = []
for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()
    
    transfered_feartures = model(transfered)
    content_features = model(content_img)
    style_features = model(style_img)
    
    content_loss = 0
    style_loss = 0
    
    for transfered_fearture, content_feature, style_feature in zip(transfered_feartures, content_features, style_features):
        N, C, H, W = transfered_fearture.shape
        
        content_loss += torch.mean((transfered_fearture - content_feature)**2)
        
        G = transfered_fearture.view(C, H*W).mm(transfered_fearture.view(C, H*W).t())
        A = style_feature.view(C, H*W).mm(style_feature.view(C, H*W).t())
        style_loss += torch.mean((G - A)**2)
        
    total_loss = alpha*content_loss + beta*style_loss
    total_loss.backward()
    optimizer.step()
    
    losses.append(total_loss.item())
    plot(transfered.detach().cpu().squeeze(0).permute(1, 2, 0).numpy(), losses)
    
    if epoch%5 == 0:
        os.makedirs('results', exist_ok=True)
        save_image(transfered, os.path.join('results', f'transfered_epoch[{epoch}].png'))

img2gif(remove_imgs=True)