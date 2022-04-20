import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ArmyDataset(Dataset):
    def __init__(self, root='dataset'):
        
        self.root = root
        
        self.inputs, self.targets = self.get_data()
        
    def get_data(self):
        data = pd.read_csv(glob.glob(os.path.join('dataset', '*.csv'))[0], encoding='cp949', low_memory=False)
        data = data.iloc(axis=1)[2:10]
        data.iloc(axis=1)[3] = data.iloc(axis=1)[3].apply(lambda x: x.split('(')[0] if str(x).endswith(')') else x)
        data = data.astype(np.float32)
        data = data/data.max()
        
        inputs, targets = data.iloc(axis=1)[:-1].to_numpy().astype(np.float32), data.iloc(axis=1)[-1].to_numpy()
        
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
