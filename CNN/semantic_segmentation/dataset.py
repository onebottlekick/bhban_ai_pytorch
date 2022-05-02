import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SegmentDataset(Dataset):
    def __init__(self, root='data', transform=transforms.ToTensor()):
        img_path = os.path.join(root, 'images')
        annotation_path = os.path.join(root, 'annotations')
        files = os.listdir(img_path)
        
        imgs = [transform(Image.open(os.path.join(img_path, file)).convert('RGB')) for file in files]
        annos = [transform(Image.open(os.path.join(annotation_path, file)).convert('RGB')) for file in files]
        
        self.imgs = imgs
        self.annos = annos
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        return self.imgs[idx], self.annos[idx]
