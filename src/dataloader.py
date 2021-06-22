import numpy as np

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class NumpyDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x: np.ndarray = x
        self.y: np.ndarray = y
        sc = StandardScaler()
        self.x = sc.fit_transform(self.x)
        self.x, self.y = self.x.astype(np.float32), self.y.astype(np.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y
