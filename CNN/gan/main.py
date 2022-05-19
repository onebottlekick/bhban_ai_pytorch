import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid

from models import Discriminator, Generator
from utils import train, plot, img2gif


os.makedirs('checkpoints', exist_ok=True)
generator_path = os.path.join('checkpoints', 'generator_mnist.pt')
discriminator_path = os.path.join('checkpoints', 'discriminator_mnist.pt')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 5
BETAS = (0.5, 0.999)
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor()
])

dataloader = DataLoader(
    dataset=MNIST(
                root='data',
                train=True,
                download=True,
                transform=transform
            ),
    shuffle=True,
    batch_size=BATCH_SIZE
)

generator = Generator().to(DEVICE)
# if os.path.exists(generator_path):
#     generator.load_state_dict(torch.load(generator_path))
discriminator = Discriminator().to(DEVICE)
# if os.path.exists(discriminator_path):
#     discriminator.load_state_dict(torch.load(discriminator_path))

criterion = nn.BCELoss()
g_optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=BETAS)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=BETAS)

best_g_loss = float('inf')
for epoch in range(NUM_EPOCHS):
    g_loss, d_loss = train(generator, discriminator, dataloader, g_optimizer, d_optimizer, criterion, DEVICE)
    
    print(f'Epoch [{epoch+1:03}/{NUM_EPOCHS}] G_Loss: {g_loss:.4f}, D_Loss: {d_loss:.4f}')
    # plot(generator, DEVICE)
    
    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator.state_dict(), discriminator_path)

img2gif(duration=0.1, to='mnist_gan.gif')