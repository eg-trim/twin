import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Optional

def get_filter_rows(data):
    reshaped_data = data.reshape(data.shape[0], -1)
    rows_with_nan = np.any(np.isnan(reshaped_data), axis=1)
    return rows_with_nan

def clean_data(Data):
    total_nan_rows = get_filter_rows(Data['u']) | get_filter_rows(Data['a'])
    Data['u'] = Data['u'][~total_nan_rows]
    Data['a'] = Data['a'][~total_nan_rows]
    return Data

def load_navier_stokes_tensor(
    mat_path: Path,
    *,
    time_points: Optional[int] = None) -> torch.Tensor:
    data = clean_data(loadmat(mat_path))
    u_np = data["u"]  # (N, H, W, T'-1)
    a_np = data["a"]  # (N, H, W)

    u = torch.from_numpy(u_np)[..., None].permute(0, 3, 1, 2, 4)  # (N, T'-1, H, W, Q))  Q = 1
    a = torch.from_numpy(a_np)[..., None, None].permute(0, 3, 1, 2, 4)  # (N, 1, H, W, Q))

    # ``a`` becomes the first time step
    data_tensor = torch.cat([a, u], dim=1)  # (N, T', H, W, Q)

    # Optionally subsample the temporal dimension to `time_points` frames
    if time_points is not None and time_points < data_tensor.shape[1]:
        idx = np.linspace(0, data_tensor.shape[-1] - 1, num=time_points, dtype=int)
        data_tensor = data_tensor[:, idx]
    return data_tensor.float()  # (N, T, H, W, Q)

class NavierStokesDataset(Dataset):
    def __init__(self, data: torch.Tensor):
        super().__init__()
        self.data = data  # (N, T, H, W, Q)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        trajectory = self.data[idx]  # (T, H, W, Q)
        return trajectory


def setup_dataloaders(tensor: torch.Tensor, *, batch_size: int, train_fraction: float = 0.8):
    dataset = NavierStokesDataset(tensor)
    n_train = int(len(dataset) * train_fraction)
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False),
    )