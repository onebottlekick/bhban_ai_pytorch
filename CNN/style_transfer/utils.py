import os

import imageio
import matplotlib.pyplot as plt
import torch
from IPython import display
from PIL import Image


def load_img(img, transform, device):
    img = Image.open(img).convert('RGB')
    img = transform(img)
    
    return img.to(device)


def gram_matrix(x):
    N, C, H, W = x.shape
    features = x.view(N*C, H*W)
    gram = torch.mm(features, features.t())
    return gram/(2*N*C*H*W)


def plot(content, style, noise, content_losses, style_losses, total_losses):
    with plt.ion():
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        
        plt.subplot(2, 2, 1)
        plt.axis('off')
        plt.title('Content')
        plt.imshow(content)
        
        plt.subplot(2, 2, 2)
        plt.axis('off')
        plt.title('Style')
        plt.imshow(style)
        
        plt.subplot(2, 2, 3)
        plt.axis('off')
        plt.title('Noise')
        plt.imshow(noise)

        plt.subplot(2, 2, 4)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot(content_losses, label='content')
        plt.text(len(content_losses) - 1, content_losses[-1], str(content_losses[-1]))
        plt.plot(style_losses, label='style')
        plt.text(len(style_losses) - 1, style_losses[-1], str(style_losses[-1]))
        # plt.plot(total_losses, label='total', color='red')
        # plt.text(len(total_losses) - 1, total_losses[-1], str(total_losses[-1]))
        plt.legend()
        plt.show()

        plt.pause(0.1)
        
        
def img2gif(root='results', to='result.gif', remove_imgs=False, duration=0.5):
    imgs = [Image.open(os.path.join(root, img)) for img in os.listdir(root) if img.endswith('png')]    
    imageio.mimsave(os.path.join(root, to), imgs, duration=duration)
    
    if remove_imgs:
        del imgs
        for img in os.listdir(root):
            if not img.endswith(".gif"):
                os.remove(os.path.join(root, img))
