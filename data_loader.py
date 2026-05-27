import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class WeatherDataset(Dataset):
    def __init__(self, data_path, mean_path, std_path, n_input_steps=2):
        self.data = np.load(data_path, mmap_mode='r')
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
        self.n_input_steps = n_input_steps

    def __len__(self):
        return self.data.shape[0] - self.n_input_steps

    def __getitem__(self, idx):
        # Normalize on the fly
        frames = (self.data[idx:idx+self.n_input_steps] - self.mean) / self.std
        x = np.concatenate(frames, axis=0).astype(np.float32)

        current = (self.data[idx+self.n_input_steps-1] - self.mean[0]) / self.std[0]
        next_state = (self.data[idx+self.n_input_steps] - self.mean[0]) / self.std[0]
        y = (next_state - current).astype(np.float32)

        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())