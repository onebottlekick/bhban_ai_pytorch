import glob
import os
import tarfile
import urllib

from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import DownloadProgressBar


class PetDataset(Dataset):
    def __init__(self, root='data', download=False, transform=None):
        self.root = root
        self.transform = transform
        os.makedirs(root, exist_ok=True)
        
        if download:
            if len(os.listdir(root)) == 0:
                self._download()
                        
        self.img_path = glob.glob(os.path.join(self.root, 'images', '*.jpg'))
        self.anno_path = glob.glob(os.path.join(self.root, 'annotations', 'trimaps', '*.png'))
        
        if len(self.img_path) != len(self.anno_path):
                print(f'Please try download=True or remove data dir')
                raise Exception
            
        if self.transform:
            self.img = [self.transform(Image.open(img).convert('RGB')) for img in self.img_path]
            self.anno = [self.transform(Image.open(anno).convert('L')) for anno in self.anno_path]
        else:
            self.img = [Image.open(img).convert('RGB') for img in self.img_path]
            self.anno = [Image.open(anno).convert('L') for anno in self.anno_path]
        
    def _download(self):
        img_url = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz'
        anno_url = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz'
        
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=img_url.split('/')[-1]) as t:
            file_path, _ = urllib.request.urlretrieve(img_url, reporthook=t.update_to)
        with tarfile.open(file_path, 'r') as f:
            for name in tqdm(iterable=f.getnames(), total=len(f.getnames())):
                f.extract(member=name, path=self.root)
                
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=anno_url.split('/')[-1]) as t:
            file_path, _ = urllib.request.urlretrieve(anno_url, reporthook=t.update_to)
        with tarfile.open(file_path, 'r') as f:
            for name in tqdm(iterable=f.getnames(), total=len(f.getnames())):
                f.extract(member=name, path=self.root)
    
    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, idx):
        # img_path = self.img_path[idx]
        # anno_path = self.anno_path[idx]
        
        # img = Image.open(img_path).convert('RGB')
        # anno = Image.open(anno_path).convert('RGB')
        
        # if self.transform:
        #     img, anno = self.transform(img), self.transform(anno)        
        
        return self.img[idx], self.anno[idx]
