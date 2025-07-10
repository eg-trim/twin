from tkinter import FALSE
import matplotlib.pyplot as plt # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
from torch.utils.data import Dataset, DataLoader, random_split  # type: ignore
from pathlib import Path
from typing import Optional, Tuple, Callable
import os
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
import math

'''
This file contains the data loading and preprocessing for the double pendulum dataset.
The double pendulum dataset (currently) is 5000 trajectories, each separately stored as a .npy array.
The file names are "ic_{ic#}_dt_pow_{pow#}.npy". 
    ic# is the index of the initial condition, stored in ICs.npy
    dt refers to the timestep in the RK4 simulation, with dt = 2^(-pow#)
The trajectories in these files are for 32 seconds.
The np array rows correspond to time steps, with each row [theta1, theta2, theta1_d, theta2_d]
'''

def load_double_pendulum_data(dir: Path, time_trimmed: bool = False, const_pendulum_parameters: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # time_trimmed True means that the trajectories stored in dir have been trimmed and all timesteps are to be included.
    # time_trimmed False means that the trajectories stored in dir still need to be trimmed by this function.

    # Load initial conditions
    ICs = np.load("ICs.npy")
    if ICs.ndim == 1:
        ICs = np.array([ICs])
    print("ICs Loaded")

    trajectory_files = [f for f in os.listdir(dir) if f.endswith('.npy')]

    snapshot_interval = 2**-5 # seconds # When changing, also change trimmed_dir name
    max_time = 2 # seconds

    # Check in case future implementations allow snapshot_interval, max_time to depend on inputs load_double_pendulum_data
    if max_time/snapshot_interval != int(max_time/snapshot_interval):
        print("Warning: max_time not an multiple of snapshot_interval. Setting max_time to be the next multiple of snapshot_interval.")
        max_time = math.ceil(max_time/snapshot_interval) * snapshot_interval

    # t_coords is the time coordinates 0, snapshot_interval, 2*snapshot_interval, ..., up to a maximum of max_time, not including max_time
    t_coords = torch.arange(0, max_time, snapshot_interval)

    # Temporarily set maximum IC because data is still being generated
    max_ic = 1

    # Treating pendulum parameters as variables
    num_vars = 8

    # Load all trajectories
    # Change max_ic to len(trajectory_files) when data is done being generated
    trajectories = torch.zeros((max_ic, len(t_coords), num_vars)) # (ic_idx, time (including initial condition), [theta1, theta2, theta1_d, theta2_d]) # Also pend_params as traj data for now
    problem_ics = []
    for file in trajectory_files:
        # Expect trajectory files to be named "ic_{ic#}_dt_pow_{pow#}.npy"
        ic_idx = int(file.split('_')[1])
        if ic_idx >= max_ic:
            continue
        dt_pow = int(file.split('_')[4].removesuffix('.npy'))

        # TO DO: Include check to see if these are integers (without int function) and if not, provide warning and do not include these trajectories in the dataset.
        snapshot_timestep = snapshot_interval * 2**dt_pow
        if snapshot_timestep != int(snapshot_timestep):
            print(f"Warning: snapshot_timestep is not an integer for file {file}")
            problem_ics.append(ic_idx)
            continue
        max_timestep = max_time * 2**dt_pow
        
        data = np.load(os.path.join(dir, file))
        ics = ICs[ic_idx, :]
        if const_pendulum_parameters:
            pend_params = [1, 1, 1, 1]
        else:
            pend_params = ics[4:]

        # Trim the timestamps to once snapshot interval
        if not time_trimmed:
            data = data[:int(max_timestep):int(snapshot_timestep), :]
            # Create a folder for the trimmed trajectories and save each as a numpy array with the same name as the original file
            # If overwrite is true, overwrite preexisting files if present. If overwrite is false, do not overwrite.
            trimmed_dir_name = f"max_time_{max_time}_fps_{int(1/snapshot_interval)}_trimmed"
            trimmed_dir = os.path.join(dir, trimmed_dir_name)
            os.makedirs(trimmed_dir, exist_ok=True)
            
            overwrite = False
            output_path = os.path.join(trimmed_dir, file)
            if os.path.exists(output_path) and not overwrite:
                print(f"File not saved, data already exists at: {output_path}")
            else:
                np.save(output_path, data)
                print(f"Saved trimmed trajectory to {output_path}")

        # Add pendulum parameters to the trajectory data
        data = torch.tensor(data)
        pend_params = torch.tensor(pend_params)
        pend_params = pend_params.unsqueeze(0).expand(data.shape[0], -1)
        data = torch.cat([data, pend_params], dim=1)
        trajectories[ic_idx, :, :] = data

    if problem_ics:
        print(f"Warning: {len(problem_ics)} initial conditions were skipped due to problems with their trajectories.")
        print(f"Problem ICs: {problem_ics}")
        # Remove problem ICs from trajectories
        trajectories = trajectories[~np.isin(np.arange(max_ic), problem_ics)]
        N = trajectories.shape[0]
        print(f"Number of valid trajectories: {N}")

    # Assuming 'trajectories' has shape (N, T+1, Q), N = number of initial conditions, T = number of time steps (not including initial condition), Q = 8 (theta1, theta2, theta1_d, theta2_d, pend_params)
    # Add H and W dimensions to get (N, T+1, 1, 1, Q)
    trajectories = trajectories.unsqueeze(2).unsqueeze(3)
    initial_conditions = trajectories[:, :1, :, :, :]
    trajectories = trajectories[:, 1:, :, :, :]

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

def make_positional_encoding(T, H, W, device: torch.device) -> torch.Tensor:
    t_coord = torch.linspace(0, 1, T, device=device)  # (T,)
    row_coord = torch.linspace(0, 1, H, device=device)  # (H,)
    col_coord = torch.linspace(0, 1, W, device=device)  # (W,)

    time_enc = t_coord.view(T, 1, 1).expand(T, H, W)  # (T, H, W)
    row_enc  = row_coord.view(1, H, 1).expand(T, H, W)  # (T, H, W)
    col_enc  = col_coord.view(1, 1, W).expand(T, H, W)  # (T, H, W)

    out = torch.stack([time_enc, row_enc, col_enc], dim=-1)  # (T, H, W, P)
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

    def forward(self, init_cond: torch.Tensor, traj: torch.Tensor, t_coords: torch.Tensor) -> torch.Tensor:
        # Input to the encoder (B, T, H, W, Q)
        enc_out = self.encoder(init_cond.unsqueeze(1).repeat(1, len(t_coords), 1, 1, 1))  # (B,T+1,H',W',C=D-P)

        # Build token sequence with positional encodings ----------------------------------
        B, Tplus1, Hp, Wp, C = enc_out.shape
        pend_params = init_cond[:, 0, 0, 4:].to(enc_out.device)  # (B, 4)

        # Coordinate encodings ----------------------------------------------------------
        t_enc  = t_coords.to(enc_out.device).view(1, Tplus1, 1, 1, 1).expand(B, -1, Hp, Wp, 1)
        row_enc = torch.linspace(0, 1, Hp, device=enc_out.device).view(1, 1, Hp, 1, 1).expand(B, Tplus1, -1, Wp, 1)
        col_enc = torch.linspace(0, 1, Wp, device=enc_out.device).view(1, 1, 1, Wp, 1).expand(B, Tplus1, Hp, -1, 1)

        pend_params_exp = pend_params.view(B, 1, 1, 1, 4).expand(B, Tplus1, Hp, Wp, 4)

        parameter_encodings = torch.cat([t_enc, row_enc, col_enc, pend_params_exp], dim=-1)  # (B,Tplus1,Hp,Wp,7)

        tokens = torch.cat([parameter_encodings, enc_out], dim=-1).reshape(B, Tplus1*Hp*Wp, C + 7)
        model_out = self.model(tokens)  # (B,L,D)

        # Extract predicted encoded features and reshape for decoder -----------------------
        preds_enc = model_out[..., -C:]  # last C dims
        preds_spatial = preds_enc.view(B, Tplus1, Hp, Wp, C)
        dec_in = preds_spatial.reshape(B, Tplus1, Hp*Wp*C)
        dec_out = self.decoder(dec_in)  # (B,Tplus1,H,W,2) # 4

        return dec_out[:, 1:, ...].reshape_as(traj[..., :2]) # (B,T,H,W,2) # 4

class NavierStokesDataset(Dataset):
    """Dataset yielding (initial_condition, trajectory) pairs.

    a : (N, 1, H, W, Q)  -- initial conditions
    u : (N, T, H, W, Q) -- subsequent frames
    """

    def __init__(self, a: torch.Tensor, u: torch.Tensor):
        super().__init__()
        assert a.shape[0] == u.shape[0], "Mismatched sample counts"
        self.a = a.squeeze(1)  # (N, H, W, Q)
        self.u = u             # (N, T, H, W, Q)

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
    train_ds = torch.utils.data.Subset(dataset, range(n_val, len(dataset)))
    val_ds = torch.utils.data.Subset(dataset, range(n_val))

    
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True),
    )

class Trainer:
    """Generic trainer parameterised by `predict_fn`.
    DataLoader must yield `(initial_conditions, trajectory)` where:
        initial_conditions : (B, H, W, Q)
        trajectory         : (B, T, H, W, Q)  (frames 1 … T)
    The trainer compares `predict_fn(init, traj)` against the same `traj`.
    """

    def __init__(self,
                 predict_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
                 loss_fn: Callable = F.mse_loss):
        self.predict_fn = predict_fn
        self.loss_fn = loss_fn

    def _epoch(self, loader: DataLoader, t_coords: torch.Tensor, optim: torch.optim.Optimizer | None) -> Tuple[float, torch.Tensor]:
        device = next(self.predict_fn.model.parameters()).device  # type: ignore[attr-defined]
        
        if len(loader) == 0:
            return 0.0, torch.zeros(4, device=device)

        train_mode = optim is not None

        running_loss = 0.0
        running_per_dim_mse = None
        for init_cond, traj in loader:
            if train_mode:
                optim.zero_grad()

            # Move data to the same device as the model parameters (attribute set in build_trainer)
            init_cond = init_cond.to(device)
            traj = traj.to(device)

            preds = self.predict_fn(init_cond, traj, t_coords.to(device))

            loss = self.loss_fn(preds, traj[..., :2]) # Do not calculate loss on the pendulum parameters. # 2

            with torch.no_grad():
                per_dim_mse = torch.mean((preds - traj[..., :2])**2, dim=tuple(range(preds.ndim - 1)))
                if running_per_dim_mse is None:
                    running_per_dim_mse = per_dim_mse
                else:
                    running_per_dim_mse += per_dim_mse

            if train_mode:
                loss.backward()
                optim.step() 

            running_loss += loss.item()

        if running_per_dim_mse is None:
            # This path should not be taken if loader is not empty, but as a safeguard:
            running_per_dim_mse = torch.zeros(2, device=device) # 4

        return running_loss / len(loader), running_per_dim_mse / len(loader)

    def train_epoch(self, loader: DataLoader, t_coords: torch.Tensor, optim: torch.optim.Optimizer) -> Tuple[float, torch.Tensor]:
        return self._epoch(loader, t_coords, optim)

    def eval_epoch(self, loader: DataLoader, t_coords: torch.Tensor) -> Tuple[float, torch.Tensor]:
        with torch.no_grad():
            return self._epoch(loader, t_coords, None)

    def get_all_validation_losses(self, loader: DataLoader, t_coords: torch.Tensor) -> dict[int, float]:
        """
        Computes the loss for every sample in the validation set.

        Returns:
            A dictionary mapping the original sample index to its MSE loss.
        """
        self.predict_fn.model.eval() # type: ignore
        all_losses = {}
        
        with torch.no_grad():
            for batch_idx, (init_cond, traj) in enumerate(loader):
                # Move data to device
                device = next(self.predict_fn.model.parameters()).device # type: ignore
                init_cond = init_cond.to(device)
                traj = traj.to(device)

                # Get predictions
                preds = self.predict_fn(init_cond, traj, t_coords.to(device))

                # Calculate per-sample MSE loss, averaging over all dimensions except batch
                per_sample_loss = F.mse_loss(preds, traj[..., :2], reduction='none')
                per_sample_loss = per_sample_loss.mean(dim=tuple(range(1, per_sample_loss.ndim)))

                # Check each sample in the batch
                for i in range(len(per_sample_loss)):
                    sample_loss = per_sample_loss[i].item()
                    
                    # The validation loader is not shuffled, so we can recover the original index
                    if loader.batch_size is None:
                        raise ValueError("Dataloader batch size cannot be None for index recovery.")
                    original_index = batch_idx * loader.batch_size + i
                    all_losses[original_index] = sample_loss
        return all_losses

    def analyze_validation_set(self, loader: DataLoader, t_coords: torch.Tensor, k: int = 5) -> list:
        """
        Analyzes the validation set to find the k samples with the highest loss.

        Returns:
            A list of tuples, where each tuple contains:
            (loss, predicted_trajectory, actual_trajectory, sample_index)
            The list is sorted by loss in descending order.
        """
        self.predict_fn.model.eval() # type: ignore
        
        worst_samples = []
        best_samples = []
        
        with torch.no_grad():
            for batch_idx, (init_cond, traj) in enumerate(loader):
                # Move data to device
                device = next(self.predict_fn.model.parameters()).device # type: ignore
                init_cond = init_cond.to(device)
                traj = traj.to(device)

                # Get predictions
                preds = self.predict_fn(init_cond, traj, t_coords.to(device))

                # Calculate per-sample MSE loss, averaging over all dimensions except batch
                per_sample_loss = F.mse_loss(preds, traj[..., :2], reduction='none') # 4
                per_sample_loss = per_sample_loss.mean(dim=tuple(range(1, per_sample_loss.ndim)))

                # Check each sample in the batch
                for i in range(len(per_sample_loss)):
                    sample_loss = per_sample_loss[i].item()
                    
                    # The validation loader is not shuffled, so we can recover the original index
                    if loader.batch_size is None:
                        raise ValueError("Dataloader batch size cannot be None for index recovery.")
                    original_index = batch_idx * loader.batch_size + i
                    
                    # This is a simple sorting approach. A min-heap would be more efficient for large k.
                    if len(worst_samples) < k:
                        worst_samples.append((sample_loss, preds[i], traj[i], original_index))
                        worst_samples.sort(key=lambda x: x[0], reverse=True)
                    elif sample_loss > worst_samples[-1][0]:
                        worst_samples.pop()
                        worst_samples.append((sample_loss, preds[i], traj[i], original_index))
                        worst_samples.sort(key=lambda x: x[0], reverse=True)

        return worst_samples

def build_trainer(*,
                  encoder: nn.Module,
                  model: nn.Module,
                  decoder: nn.Module,
                  loss_fn: Callable = F.mse_loss,
                  ) -> Trainer:
    pipeline = AcausalPipeline(model, encoder, decoder)

    def predict_fn(init_cond: torch.Tensor, traj: torch.Tensor, t_coords: torch.Tensor):  # type: ignore[override]
        return pipeline(init_cond, traj, t_coords)

    # Expose underlying model so Trainer can infer device without relying on __self__
    setattr(predict_fn, "model", model)

    return Trainer(predict_fn, loss_fn)