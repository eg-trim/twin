import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Optional, Tuple

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
    n_timesteps: Optional[int] = None) -> torch.Tensor:
    data = clean_data(loadmat(mat_path))
    u_np = data["u"]  # (N, H, W, T'-1)
    a_np = data["a"]  # (N, H, W)

    u = torch.from_numpy(u_np)[..., None].permute(0, 3, 1, 2, 4)  # (N, T'-1, H, W, Q))  Q = 1
    a = torch.from_numpy(a_np)[..., None, None].permute(0, 3, 1, 2, 4)  # (N, 1, H, W, Q))

    # ``a`` becomes the first time step

    # Optionally subsample the temporal dimension to `n_timesteps` frames
    if n_timesteps is not None and n_timesteps < u.shape[1]:
        idx = np.linspace(0, u.shape[1] - 1, num=n_timesteps, dtype=int)
        u = u[:, idx]
    return a, u  # (N, T, H, W, Q), (N, 1, H, W, Q)

class NavierStokesDataset(Dataset):
    """Dataset yielding (initial_condition, trajectory) pairs.

    a : (N, 1, H, W, Q)  -- initial conditions
    u : (N, T-1, H, W, Q) -- subsequent frames
    """

    def __init__(self, a: torch.Tensor, u: torch.Tensor):
        super().__init__()
        assert a.shape[0] == u.shape[0], "Mismatched sample counts"
        self.a = a.squeeze(1)  # (N, H, W, Q)
        self.u = u             # (N, T-1, H, W, Q)

    def __len__(self) -> int:  # number of samples
        return self.a.shape[0]

    def __getitem__(self, idx: int):
        return self.a[idx], self.u[idx]


def setup_dataloaders(a: torch.Tensor, u: torch.Tensor, batch_size: int, train_fraction: float = 0.8):
    dataset = NavierStokesDataset(a, u)
    n_train = int(len(dataset) * train_fraction)
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True),
    )


class WaveformDataset(Dataset):
    """Dataset for gravitational waveforms with conditional parameters."""
    def __init__(self, waveforms: torch.Tensor, params: torch.Tensor):
        super().__init__()
        assert waveforms.shape[0] == params.shape[0], "Number of waveforms and parameters must match"
        self.waveforms = waveforms  # (N, T, 1, 1, 2)
        self.params = params      # (N, n_params)

    def __len__(self) -> int:
        return self.waveforms.shape[0]

    def __getitem__(self, idx: int):
        waveform = self.waveforms[idx]  # (T, 1, 1, 2)
        param = self.params[idx]        # (n_params,)
        return waveform, param


def setup_waveform_dataloaders(
    waveforms: torch.Tensor,
    params: torch.Tensor,
    batch_size: int,
    train_fraction: float = 0.8
):
    """Sets up DataLoaders for waveform data with conditional parameters."""
    dataset = WaveformDataset(waveforms, params)
    n_train = int(len(dataset) * train_fraction)
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True),
    )

def load_bhpt_tensors(
    pt_path: Path,
    *,
    n_timesteps: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Loads waveform and parameter tensors from a .pt file."""
    print(f"Loading data from {pt_path}...")
    data = torch.load(pt_path)
    waveforms_tensor = data["waveforms"]
    params_tensor = data["params"]
    print(f"  Loaded 'waveforms' with shape: {waveforms_tensor.shape}")
    print(f"  Loaded 'params' with shape: {params_tensor.shape}")

    if n_timesteps is not None and n_timesteps < waveforms_tensor.shape[1]:
        original_timesteps = waveforms_tensor.shape[1]
        idx = torch.linspace(0, original_timesteps - 1, steps=n_timesteps, dtype=torch.long)
        waveforms_tensor = waveforms_tensor[:, idx]
        print(f"  Subsampled time from {original_timesteps} to {waveforms_tensor.shape[1]} steps.")
    
    print(f"Final tensor shapes: waveforms={waveforms_tensor.shape}, params={params_tensor.shape}")
    return waveforms_tensor, params_tensor