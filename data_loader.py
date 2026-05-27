import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class WeatherDataset(Dataset):
    def __init__(self, data_path, n_input_steps=2):
        self.data = np.load(data_path)
        self.n_input_steps = n_input_steps

    def __len__(self):
        return self.data.shape[0] - self.n_input_steps

    def __getitem__(self, idx):
        frames = self.data[idx:idx+self.n_input_steps]
        x = np.concatenate(frames, axis=0)
        current = self.data[idx+self.n_input_steps-1]
        y = self.data[idx+self.n_input_steps] - current
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())
