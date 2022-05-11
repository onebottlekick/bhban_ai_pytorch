import torch.nn as nn
import torchvision.models as models


from utils import gram_matrix


class StyleTransfer:
    def __init__(self, device, layers={1:'s', 2:'s', 3:'s', 4:'sc', 5:'s'}):
        self.layers = layers        
        self.model = self._get_model().to(device)
    
    def _get_model(self):
        vgg = nn.ModuleList(models.vgg19(pretrained=True).features)
        
        for param in vgg.parameters():
            param.requires_grad_(False)
        
        conv_idx = []   
        for idx, layer in enumerate(vgg):
            if isinstance(layer, nn.MaxPool2d):
                layer = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
            
            elif isinstance(layer, nn.Conv2d):
                conv_idx.append(idx)
        
        model = nn.Sequential(*vgg[:conv_idx[len(self.layers)]])
        self.conv_idx = conv_idx
        
        return model
