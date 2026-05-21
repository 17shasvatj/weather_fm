import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class WeatherDataset(Dataset):
    def __init__(self, data_path, mean_path, std_path, n_input_steps=2):
        data = np.load(data_path)
        mean = np.load(mean_path)
        std = np.load(std_path)

        self.data = ((data-mean)/std).astype(np.float32)
        self.n_input_steps = n_input_steps

    def __len__(self):
        return self.data.shape[0] - self.n_input_steps

    def __getitem__(self, idx):
        frames = self.data[idx:idx+self.n_input_steps] # (n_input_steps, C, H, W)
        x = np.concatenate(frames, axis=0)  # (n_input_steps * C, H, W)
        current = self.data[idx+self.n_input_steps-1]
        y = self.data[idx+self.n_input_steps] - current # (C, H, W)
        return torch.from_numpy(x), torch.from_numpy(y)

def get_dataloaders(train_path, test_path, mean_path, std_path,
                    n_input_steps=2, batch_size=16):
    train_set = WeatherDataset(train_path, mean_path, std_path, n_input_steps)
    test_set = WeatherDataset(test_path, mean_path, std_path, n_input_steps)

    num_workers = 0 if torch.backends.mps.is_available() else 2

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    print(f"Train samples: {len(train_set)}")
    print(f"Test samples: {len(test_set)}")
    print(f"Input channels: {n_input_steps * 6}")
    print(f"Target channels: 6")

    return train_loader, test_loader









