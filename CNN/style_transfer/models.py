import torch
import torch.nn as nn
import torchvision.models as models


from utils import gram_matrix


class StyleTransfer:
    def __init__(self, device, layers={1:'s', 2:'s', 3:'s', 4:'sc', 5:'s'}):
        self.layers = layers        
        self.normalization = Normalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device=device)
        self.model = self._get_model().to(device)
    
    def _get_model(self):
        vgg = nn.ModuleList(models.vgg19(pretrained=True).features)
        
        for param in vgg.parameters():
            param.requires_grad_(False)
        
        conv_idx = []   
        for idx, layer in enumerate(vgg):
            # if isinstance(layer, nn.MaxPool2d):
            #     layer = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
            
            if isinstance(layer, nn.Conv2d):
                conv_idx.append(idx)
            
            if isinstance(layer, nn.ReLU):
                layer = nn.ReLU()
        
        model = nn.Sequential(self.normalization, *vgg[:conv_idx[len(self.layers)]])
        self.conv_idx = conv_idx
        
        return model
    

class Normalization(nn.Module):
    def __init__(self, mean, std, device):
        super(Normalization, self).__init__()
        
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)
        
    def forward(self, img):
        return (img - self.mean)/self.std
