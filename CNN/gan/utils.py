import datetime
import glob
import os

import imageio
import matplotlib.pyplot as plt
import torch
from IPython import display
from PIL import Image
from torchvision.utils import make_grid, save_image


def train(generator, discriminator, dataloader, g_optimizer, d_optimizer, criterion, device):
    g_losses = 0
    d_losses = 0
    for idx, (img, _) in enumerate(dataloader):
        real = img.to(device)
        
        valid = torch.ones(img.size(0), 1, requires_grad=False, device=device)
        fake = torch.zeros(img.size(0), 1, requires_grad=False, device=device)

        g_optimizer.zero_grad()
        z = torch.randn(img.shape[0], 100, device=device)
        gen = generator(z)
        g_loss = criterion(discriminator(gen), valid)
        g_loss.backward()
        g_optimizer.step()
        
        if idx % 30 == 0:
            plot(generator, device)
        
        d_optimizer.zero_grad()
        real_loss = criterion(discriminator(real), valid)
        fake_loss = criterion(discriminator(gen.detach()), fake)
        d_loss = (real_loss + fake_loss)/2
        d_loss.backward()
        d_optimizer.step()
        
        g_losses += g_loss.item()
        d_losses += d_loss.item()
        
    return g_losses/len(dataloader), d_losses/len(dataloader)


def plot(generator, device, n_imgs=100, save_imgs=True, save_path='results', show_imgs=False):
    z = torch.randn(n_imgs, 100, device=device)
    gen_imgs = generator(z)    
    
    if save_imgs:
        grid = make_grid(gen_imgs, nrow=10, pad_value=255)
        os.makedirs(save_path, exist_ok=True)
        save_image(grid, os.path.join(save_path, f'{datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")}.png'))
    
    if show_imgs:
        grid = make_grid(gen_imgs, nrow=10).permute(1, 2, 0).detach().cpu()
        with plt.ion():
            display.clear_output()
            display.display(plt.gcf())
            plt.clf()

            plt.axis('off')
            plt.imshow(grid, cmap='gray')
            plt.show()
            plt.pause(0.1)
            

def img2gif(root='results', to='result.gif', remove_imgs=False, duration=0.5):
    imgs = [Image.open(os.path.join(root, img)) for img in sorted(os.listdir(root)) if img.endswith('png')]    
    imageio.mimsave(os.path.join(root, to), imgs, duration=duration)
    
    if remove_imgs:
        del imgs
        for img in os.listdir(root):
            if not img.endswith(".gif"):
                os.remove(os.path.join(root, img))
