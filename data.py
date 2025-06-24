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
    time_points: Optional[int] = None,
) -> torch.Tensor:
    """Load a .mat file produced by the FNO Navier–Stokes generator and return a
    tensor with shape (N, H, W, T+1).

    The .mat files contain two variables:
        * ``u``  – velocity field of shape (H, W, T, N)
        * ``a``  – scalar forcing term of shape (H, W, N)

    We transpose the arrays such that the sample dimension comes first and we
    concatenate the forcing field ``a`` in front of the temporal dimension so
    that consumers can treat it as the *initial* channel.
    """
    data = clean_data(loadmat(mat_path))
    u_np = data["u"]  # (H, W, P-1, N)
    a_np = data["a"]  # (H, W, N)

    u = torch.from_numpy(u_np)  # (N, H, W, P-1))
    a = torch.from_numpy(a_np)[..., None]  # (N, H, W, 1))

    # ``a`` becomes the first time step
    data_tensor = torch.cat([a, u], dim=-1)  # (N, H, W, P)

    # Optionally subsample the temporal dimension to `time_points` frames
    if time_points is not None and time_points < data_tensor.shape[-1]:
        idx = np.linspace(0, data_tensor.shape[-1] - 1, num=time_points, dtype=int)
        data_tensor = data_tensor[..., idx]
    return data_tensor.float() # (N, H, W, T)


class NavierStokesDataset(Dataset):
    def __init__(self, data: torch.Tensor):
        super().__init__()
        self.data = data  # (N, H, W, T)

        # Pre-compute flattened spatial coordinates in [0,1]
        _, H, W, T = data.shape
        row_coord = torch.linspace(0, 1, H)  # (H,)
        col_coord = torch.linspace(0, 1, W)  # (W,)
        t_coord = torch.linspace(0, 1, T)  # (T,)

        # 3. Broadcast so every (query-token i, key-token j, time k) triple
        #    gets the right positional triplet (row_i, col_j, t_k)
        row_enc  = row_coord.view(H, 1, 1).expand(H, W, T)  # (H, W, T)
        col_enc  = col_coord.view(1, W, 1).expand(H, W, T)  # (H, W, T)
        time_enc = t_coord.view(1, 1, T).expand(H, W, T)  # (H, W, T)

        # 4. Stack →  (H, W, T, 3)
        self.encoding = torch.stack([row_enc, col_enc, time_enc], dim=-1)  # (H, W, T, 3)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        trajectory = self.data[idx]  # (H, W, T)
        return trajectory, self.encoding  # (H, W, T), (H, W, T, 3)


def setup_dataloaders(tensor: torch.Tensor, *, batch_size: int, train_fraction: float = 0.8):
    dataset = NavierStokesDataset(tensor)
    n_train = int(len(dataset) * train_fraction)
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False),
    )