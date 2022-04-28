import os

import imageio
import matplotlib.pyplot as plt
from IPython import display
from PIL import Image
from torchvision import transforms


def load_img(img, device, root='samples'):
    img = Image.open(os.path.join(root, img))
    img = transforms.ToTensor()(img).unsqueeze(0)
    
    return img.to(device)


def plot(img, losses):
    with plt.ion():
        display.clear_output()
        display.display(plt.gcf())
        plt.clf()
        
        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow(img)
        
        plt.subplot(1, 2, 2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(losses)
        
        plt.text(len(losses) - 1, losses[-1], str(losses[-1]))
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