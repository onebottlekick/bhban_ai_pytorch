import torch.nn as nn
import torchvision.models as models


class VGG(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = [0, 5, 10, 19, 28]
        self.model = models.vgg19(pretrained=True).features[:29]
        
    def forward(self, x):
        features = []
        for i_layer, layer in enumerate(self.model):
            x = layer(x)
            
            if i_layer in self.features:
                features.append(x)
                
        return features
