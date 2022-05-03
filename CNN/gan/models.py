import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, img_size=32):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            *self._block(in_channels, 16, normalize=False),
            *self._block(16, 32),
            *self._block(32, 64),
            *self._block(64, 128)
        )
        
        self.fc = nn.Linear(128*(img_size//2**4)**2, 1)
        
        self.sigmoid = nn.Sigmoid()
        
    def _block(self, in_channels, out_channels, normalize=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2)
        ]
        
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
            
        return layers
        
    def forward(self, img):
        x = self.model(img)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
    
    
class Generator(nn.Module):
    def __init__(self, in_features=100, img_channels=1, img_size=32):
        super(Generator, self).__init__()
        self.img_size = img_size
        
        self.fc = nn.Linear(in_features, 128*((img_size//4)**2))
        
        self.model = nn.Sequential(
            nn.BatchNorm2d(128),
            *self._block(128, 128),
            *self._block(128, 64),
            nn.Conv2d(64, img_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def _block(self, in_channels, out_channels):
        layers = [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        return layers
        
    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.shape[0], 128, self.img_size//4, self.img_size//4)
        z = self.model(z)
        return z
    
    
if __name__ == '__main__':
    img = torch.randn(1, 1, 32, 32)
    z = torch.randn(1, 100)
    
    generator = Generator()
    discriminator = Discriminator()
    
    assert generator(z).shape == img.shape
    assert discriminator(img).shape == (1, 1)