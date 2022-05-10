import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class WeatherDataset(Dataset):
    def __init__(self, root='data', normalize=False, window_size=12):
        self.root = root
        self.dataset_path = os.path.join(root, 'jena_climate_2009_2016.csv')
        self.normalize = normalize
        self.window_size = window_size
        
        self.inputs, self.targets = self._get_data()
        
    def _get_data(self):
        dataset = pd.read_csv(self.dataset_path).drop('Date Time', axis=1).to_numpy()
        
        if self.normalize:
            dataset = (dataset - dataset.min(axis=0))/(dataset.max(axis=0) + 0.0001)
        
        return self._windowing(dataset)
            
    def _windowing(self, array):
        X = []
        y = []        
        for i in range(len(array)-self.window_size*2 + 1):
            X.append(torch.tensor(array[i:i+self.window_size], dtype=torch.float32))
            y.append(torch.tensor(array[i+self.window_size:i+self.window_size*2], dtype=torch.float32))
        
        return X, y
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
    
    
if __name__ == '__main__':
    dataset = WeatherDataset()
    print(dataset[0][0].shape)
