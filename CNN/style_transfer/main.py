import os

import torch
import torch.nn as nn
from torchvision import transforms

from models import StyleTransfer
from utils import gram_matrix, load_img, plot, img2gif


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

style_transfer = StyleTransfer(DEVICE)
model = style_transfer.model

layers = style_transfer.layers
conv_idx = style_transfer.conv_idx

content_path = os.path.join('samples', 'content.jpg')
style_path = os.path.join('samples', 'style.jpg')

resize = (480, 640)
transform = transforms.Compose([
    transforms.Resize(resize),
    transforms.ToTensor()
])

content = load_img(content_path, transform, DEVICE).unsqueeze(0)
style = load_img(style_path, transform, DEVICE).unsqueeze(0)

noise = torch.rand(content.shape, device=DEVICE, requires_grad=True)

NUM_EPOCHS = 300
alpha = 1e+0
beta = 1e+6

optimizer = torch.optim.LBFGS([noise], max_iter=1)
criterion = nn.MSELoss()

content_losses = []
style_losses = []
total_losses = []
for epoch in range(NUM_EPOCHS):
    def closure():
        optimizer.zero_grad()
        style_loss = 0
        content_loss = 0
        with torch.no_grad():
            noise.data.clip_(0, 1)
        for idx in layers.keys():
            if 'c' in layers[idx]:
                target_content = model[:conv_idx[idx]+2](content).detach()
                noise_content = model[:conv_idx[idx]+2](noise)
                content_loss += criterion(noise_content, target_content)
            if 's' in layers[idx]:
                target_style = gram_matrix(model[:conv_idx[idx]+2](style)).detach()
                noise_style = gram_matrix(model[:conv_idx[idx]+2](noise))
                style_loss += criterion(noise_style, target_style)

        content_losses.append(content_loss.item())
        style_losses.append(style_loss.item())

        total_loss = alpha*content_loss + beta*style_loss
        total_losses.append(total_loss.item())
        total_loss.backward()

        print(f'Epoch [{epoch+1:02}/{NUM_EPOCHS}] content_loss={content_loss:6f}, style_loss={style_loss:.6f}, total_loss={total_loss:.6f}')
        plot(
            content.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0),
            style.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0),
            noise.data.clip_(0, 1).squeeze(0).cpu().detach().numpy().transpose(1, 2, 0),
            content_losses,
            style_losses,
            total_losses,
            savefig=True
        )

        return total_loss
    
    optimizer.step(closure)
    
img2gif(remove_imgs=True, duration=0.2)