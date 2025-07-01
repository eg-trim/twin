import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Optional, Tuple, Callable
import os
import torch.nn as nn
import torch.nn.functional as F

'''
This file contains the data loading and preprocessing for the double pendulum dataset.
The double pendulum dataset (currently) is 5000 trajectories, each separately stored as a .npy array.
The file names are "ic_{ic#}_dt_pow_{pow#}.npy". 
    ic# is the index of the initial condition, stored in ICs.npy
    dt refers to the timestep in the RK4 simulation, with dt = 2^(-pow#)
The trajectories in these files are for 32 seconds.
The np array rows correspond to time steps, with each row [theta1, theta2, theta1_d, theta2_d]
'''

def load_double_pendulum_data(dir: Path, time_trimmed: bool = False) -> torch.Tensor:
    # time_trimmed True means that the trajectories stored in dir have been trimmed and all timesteps are to be included.
    # time_trimmed False means that the trajectories stored in dir still need to be trimmed by this function.

    print("Loading double pendulum data...")

    # Load initial conditions
    ICs = np.load("ICs.npy")
    if ICs.ndim == 1:
        ICs = np.array([ICs])
    print("ICs Loaded")

    trajectory_files = [f for f in os.listdir(dir) if f.endswith('.npy')]

    snapshot_interval = 2**-5 # seconds # When changing, also change trimmed_dir name
    max_time = 2**-3 # seconds
    # t_coords is the time coordinates snapshot_interval, 2*snapshot_interval, ..., up to a maximum of max_time, not including max_time
    t_coords = torch.arange(snapshot_interval, max_time, snapshot_interval)
    print(f"t_coords: {t_coords}")

    # Temporarily set maximum IC because data is still being generated
    max_ic = 1

    # Load all trajectories
    # Change max_ic to len(trajectory_files) when data is done being generated
    trajectories = torch.zeros((max_ic, 1+len(t_coords), 8)) # (ic_idx, time (including initial condition), [theta1, theta2, theta1_d, theta2_d]) # Also pend_params as traj data for now
    ics_loaded = 0
    for file in trajectory_files:
        # Expect trajectory files to be named "ic_{ic#}_dt_pow_{pow#}.npy"
        ic_idx = int(file.split('_')[1])
        if ic_idx >= max_ic:
            continue
        dt_pow = int(file.split('_')[4].removesuffix('.npy'))
        data = np.load(os.path.join(dir, file))
        ics = ICs[ic_idx, :]
        pend_params = ics[4:]

        # TO DO: Include check to see if these are integers (without int function) and if not, provide warning and do not include these trajectories in the dataset.
        max_timestep = int(max_time * 2**dt_pow)
        snapshot_timestep = int(snapshot_interval * 2**dt_pow)

        # Trim the timestamps to once snapshot interval
        if not time_trimmed:
            data = data[:max_timestep:snapshot_timestep, :]
            # Create a folder for the trimmed trajectories and save each as a numpy array with the same name as the original file
            # Comment these out after saving the first time.
            # Lazy, I know.
            power = int(np.log2(max_time))
            trimmed_dir_name = f"traj_max_time_2_pow_{power}_fps_32_dt_pow_15_const_params_trimmed"
            trimmed_dir = os.path.join(dir, trimmed_dir_name)
            os.makedirs(trimmed_dir, exist_ok=True)
            np.save(os.path.join(trimmed_dir, file), data)
            print(f"Saved trimmed trajectory to {os.path.join(trimmed_dir, file)}")
        # Add pendulum parameters to the trajectory data
        data = torch.tensor(data)
        pend_params = torch.tensor(pend_params)
        pend_params = pend_params.unsqueeze(0).expand(data.shape[0], -1)
        data = torch.cat([data, pend_params], dim=1)
        trajectories[ic_idx, :, :] = data
        # Debug
        ics_loaded += 1
        print(f"Loaded {ics_loaded} initial conditions")

    # Assuming 'trajectories' has shape (N, T+1, 4), N = number of initial conditions, T = number of time steps (not including initial condition), Q = 4 (theta1, theta2, theta1_d, theta2_d)
    # Add H and W dimensions to get (N, T+1, 1, 1, 4)
    # Q should now actually be 8
    trajectories = trajectories.unsqueeze(2).unsqueeze(3)
    initial_conditions = trajectories[:, :1, :, :, :]
    trajectories = trajectories[:, 1:, :, :, :]

    print(f"initial_conditions: {initial_conditions.shape}")
    print(f"trajectories: {trajectories.shape}")
    print(f"t_coords: {t_coords.shape}")

    return initial_conditions, trajectories, t_coords # Probably eventually want to return param_coords as tuple of coords?

def make_parameter_encoding(*coords: torch.Tensor, device: torch.device) -> torch.Tensor:
    shape = [c.shape[0] for c in coords]
    
    encodings = []
    for i, coord in enumerate(coords):
        view_shape = [1] * len(shape)
        view_shape[i] = shape[i]
        enc = coord.view(*view_shape).expand(*shape)
        encodings.append(enc)

    out = torch.stack(encodings, dim=-1)
    return out.to(device)


''' The remainder of this file is copied from the Navier-Stokes data and training files. '''
def make_positional_encoding(T, H, W, device: torch.device) -> torch.Tensor:
    t_coord = torch.linspace(0, 1, T, device=device)  # (T,)
    row_coord = torch.linspace(0, 1, H, device=device)  # (H,)
    col_coord = torch.linspace(0, 1, W, device=device)  # (W,)

    time_enc = t_coord.view(T, 1, 1).expand(T, H, W)  # (H, W, T)
    row_enc  = row_coord.view(1, H, 1).expand(T, H, W)  # (H, W, T)
    col_enc  = col_coord.view(1, 1, W).expand(T, H, W)  # (H, W, T)

    out = torch.stack([time_enc, row_enc, col_enc], dim=-1)  # (H, W, T, P)
    return out.to(device)

class AcausalPipeline(nn.Module):
    '''
    Forward pass for acausal models (predict full trajectory).
    Identical to the Navier-Stokes training class.
    '''


    def __init__(self, model: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, init_cond: torch.Tensor, traj: torch.Tensor) -> torch.Tensor:
        enc_out = self.encoder(init_cond.unsqueeze(1).repeat(1, 1 + traj.shape[1], 1, 1, 1))  # (B,T,H',W',C)

        # Build token sequence with positional encodings ----------------------------------
        B, T, Hp, Wp, C = enc_out.shape
        pos = make_positional_encoding(T, Hp, Wp, device=enc_out.device)  # (T,Hp,Wp,3)
        pos = pos.unsqueeze(0).expand(B, -1, -1, -1, -1)
        tokens = torch.cat([pos, enc_out], dim=-1).reshape(B, T*Hp*Wp, C+3)

        model_out = self.model(tokens)  # (B,L,D)

        # Extract predicted encoded features and reshape for decoder -----------------------
        preds_enc = model_out[..., -C:]  # last C dims
        preds_spatial = preds_enc.view(B, T, Hp, Wp, C)
        dec_in = preds_spatial.reshape(B, T, Hp*Wp*C)
        dec_out = self.decoder(dec_in)  # (B,T,H,W,Q)

        return dec_out[:, 1:].reshape_as(traj)

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
    # n_train = int(len(dataset) * train_fraction)
    # n_val = len(dataset) - n_train
    # train_ds, val_ds = random_split(dataset, [n_train, n_val])
    
    # Deterministic split for debugging
    n_val = int(len(dataset) * (1.0 - train_fraction))
    val_ds = torch.utils.data.Subset(dataset, range(n_val))
    train_ds = torch.utils.data.Subset(dataset, range(n_val, len(dataset)))
    
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True),
    )