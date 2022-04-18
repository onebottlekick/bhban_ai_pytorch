import glob
import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class HealthData(Dataset):
    def __init__(self, root='dataset'):
        super().__init__()
        
        self.inputs, self.targets = self.get_data()
        
    def get_data(self):
        data = pd.read_csv(glob.glob(os.path.join('dataset', '*.csv'))[0], encoding='cp949')
        data = data[['학교명', '키', '몸무게', '성별']]
        data['학교명'] = data['학교명'].apply(lambda x: 0 if x.endswith('초등학교') else 1 if x.endswith('중학교') else 2)
        data['성별'] = data['성별'].apply(lambda x: 1 if x=='남' else 0)
        
        inputs, targets = data.drop(['학교명'], axis=1).to_numpy(), data['학교명'].to_numpy()
        
        return torch.tensor(inputs), torch.tensor(targets)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
