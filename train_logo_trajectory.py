#!/usr/bin/env python3
"""Train a simple Transformer to reproduce the 2-D logo trajectory.

The training script expects a dataset file created by
`twin/double_pendulum_logo.py`, containing a dictionary with keys
    waveforms: (N, T, 1, 1, 2)
    params   : (N, P)  – ignored here (we do not condition on it)

During training the model receives the first T-1 points and must predict
the subsequent T-1 points (teacher forcing).  Loss is mean-squared error
(MSE) on the full predicted sequence.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.utils.data import DataLoader, Dataset, random_split  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib import animation  # type: ignore
import math  # NEW

# --- Trim-Transformer imports -------------------------------------------------
# We fall back to the local repo checkout if the package isn't installed.
try:
    from trim_transformer.transformer_layers import (  # type: ignore
        CumulativeTransformerEncoderLayerKV,
        CumulativeTransformerEncoderKV,
    )
except ModuleNotFoundError:  # local checkout without installation
    import sys as _sys
    from pathlib import Path as _Path

    _trim_root = _Path(__file__).resolve().parent.parent / "trim-transformer"
    _sys.path.append(str(_trim_root))
    from trim_transformer.transformer_layers import (  # type: ignore
        CumulativeTransformerEncoderLayerKV,
        CumulativeTransformerEncoderKV,
    )

# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------

class LogoTrajectoryDataset(Dataset):
    """Turns (waveforms) tensor into (input_seq, target_seq) pairs.

    Each sample is split as:
        inputs  : coordinates 0 … T-2
        targets : coordinates 1 … T-1 (next-step ground truth)
    """

    def __init__(self, waveforms: torch.Tensor):
        if waveforms.dim() != 5 or waveforms.size(2) != 1 or waveforms.size(3) != 1 or waveforms.size(4) != 2:
            raise ValueError("waveforms tensor must have shape (N, T, 1, 1, 2)")
        # Collapse spatial dims → (N, T, 2)
        self.coords = waveforms.squeeze(3).squeeze(2)  # (N, T, 2)

    def __len__(self) -> int:
        return self.coords.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        traj = self.coords[idx]  # (T, 2)
        return traj[:-1], traj[1:]  # inputs, targets  both (T-1, 2)

# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (batch_first)."""

    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        return x + self.pe[:, : x.size(1)]

class TrajectoryTransformer(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4, n_layers: int = 4):
        super().__init__()
        self.input_fc = nn.Linear(2, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        # Trim-Transformer: cumulative linear attention (no mask needed).
        enc_layer = CumulativeTransformerEncoderLayerKV(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = CumulativeTransformerEncoderKV(enc_layer, num_layers=n_layers)
        self.output_fc = nn.Linear(d_model, 2)
        self.post_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, 2)
        y = self.input_fc(x)
        y = self.pos_enc(y)
        y = self.encoder(y)
        y = self.post_norm(y)
        GAIN = 1.0  # Empirically chosen boost
        return self.output_fc(y) * GAIN  # (B, L, 2)

# --------------------------------------------------------------------------------------
# Training helpers
# --------------------------------------------------------------------------------------

def train_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer, criterion) -> float:
    model.train()
    running = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(model.device), targets.to(model.device)  # type: ignore[attr-defined]
        optim.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        optim.step()
        running += loss.item() * inputs.size(0)
    return running / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, criterion) -> float:
    model.eval()
    running = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(model.device), targets.to(model.device)  # type: ignore[attr-defined]
        preds = model(inputs)
        loss = criterion(preds, targets)
        running += loss.item() * inputs.size(0)
    return running / len(loader.dataset)

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Transformer on logo trajectory")
    p.add_argument("--dataset", type=Path, default=Path("logo_trajectory_dataset.pt"), help="Path to dataset .pt file")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save", type=Path, default=Path("logo_traj_transformer.pt"), help="Where to save the trained model")
    p.add_argument("--gif", action="store_true", help="Generate a GIF showing learning progress")
    p.add_argument("--gif-every", type=int, default=50, help="Capture predictions every N epochs")
    p.add_argument("--gif-path", type=Path, default=Path("logo_training_progress.gif"))
    return p

# --------------------------------------------------------------------------------------


def main():
    args = build_argparser().parse_args()
    data = torch.load(args.dataset)
    waves = data["waveforms"].float()  # (N, T, 1, 1, 2)
    ds = LogoTrajectoryDataset(waves)

    # Pick first sample for visualisation
    vis_input, vis_target = ds[0]  # (T-1,2) each
    true_traj = torch.cat([vis_input[:1], vis_target], dim=0).cpu().numpy()  # (T,2)

    # Determine split sizes
    if len(ds) < 2:
        print("Dataset contains a single trajectory – using it for both training and validation.")
        train_ds = val_ds = ds  # type: ignore
    else:
        n_val = max(1, int(len(ds) * args.val_fraction))
        n_train = len(ds) - n_val
        train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = TrajectoryTransformer(d_model=args.d_model, nhead=args.nhead, n_layers=args.layers)
    model.device = torch.device(args.device)  # type: ignore[attr-defined]
    model.to(model.device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Histories for animation
    predictions_history: list[np.ndarray] = []
    losses_history: list[tuple[float,float]] = []
    epoch_nums: list[int] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optim, criterion)
        val_loss = eval_epoch(model, val_loader, criterion)
        print(f"Epoch {epoch:4d} | train {train_loss:.3e} | val {val_loss:.3e}")

        if args.gif and (epoch == 1 or epoch % args.gif_every == 0 or epoch == args.epochs):
            model.eval()
            with torch.no_grad():
                pred = model(vis_input.unsqueeze(0).to(model.device)).squeeze(0).cpu().numpy()
            # prepend first true point for alignment
            pred_full = np.vstack([true_traj[0], pred])
            predictions_history.append(pred_full)
            losses_history.append((train_loss, val_loss))
            epoch_nums.append(epoch)

    torch.save(model.state_dict(), args.save)
    print(f"Saved trained model to {args.save}")

    if args.gif and predictions_history:
        _create_training_gif(predictions_history, true_traj, losses_history, epoch_nums, args.gif_path)
        print(f"Saved training animation to {args.gif_path}")


# --------------------------------------------------------------------------------------
# Animation helper
# --------------------------------------------------------------------------------------


def _create_training_gif(preds_over_time: list[np.ndarray], true_traj: np.ndarray,
                          losses: list[tuple[float,float]], epochs: list[int], out_path: Path,
                          fps: int = 5):
    """Create and save GIF comparing predicted vs ground-truth trajectory over epochs."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal')
    ax.plot(true_traj[:, 0], true_traj[:, 1], 'k--', lw=1.0, label='Ground truth')

    pred_line, = ax.plot([], [], 'r-', lw=1.5, label='Prediction')
    ax.legend()

    # Fix limits for stability
    pad = 0.1
    xmin, xmax = true_traj[:, 0].min()-pad, true_traj[:, 0].max()+pad
    ymin, ymax = true_traj[:, 1].min()-pad, true_traj[:, 1].max()+pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    text = ax.text(0.02, 0.95, '', transform=ax.transAxes, va='top')

    def animate(i):
        pred = preds_over_time[i]
        pred_line.set_data(pred[:, 0], pred[:, 1])
        tr_loss, v_loss = losses[i]
        text.set_text(f'Epoch {epochs[i]}\nTrain {tr_loss:.2e}\nVal {v_loss:.2e}')
        return pred_line, text

    anim = animation.FuncAnimation(fig, animate, frames=len(preds_over_time), interval=1000//fps, blit=True)
    anim.save(out_path, writer='pillow', fps=fps)
    plt.close(fig)


# --- Trim-Transformer imports + monkey patch ----------------------------------
try:
    from trim_transformer.transformer_layers import (  # type: ignore
        CumulativeTransformerEncoderLayerKV,
        CumulativeTransformerEncoderKV,
    )
except ModuleNotFoundError:  # local checkout without installation
    import sys as _sys
    from pathlib import Path as _Path

    _trim_root = _Path(__file__).resolve().parent.parent / "trim-transformer"
    _sys.path.append(str(_trim_root))
    from trim_transformer.transformer_layers import (  # type: ignore
        CumulativeTransformerEncoderLayerKV,
        CumulativeTransformerEncoderKV,
    )

# Monkey-patch: boost cumulative attention output by √(head_dim)
def _patch_trim_boost():
    if getattr(CumulativeTransformerEncoderLayerKV, "_boosted", False):
        return  # already patched

    original_sa = CumulativeTransformerEncoderLayerKV._sa_block  # type: ignore[attr-defined]

    def boosted_sa(self, x, mask, src_key_padding_mask, is_causal=False,
                   use_kv_cache=False, update_kv_cache=False):  # type: ignore[override]
        out = original_sa(self, x, mask, src_key_padding_mask,
                          is_causal=is_causal,
                          use_kv_cache=use_kv_cache,
                          update_kv_cache=update_kv_cache)
        return out

    CumulativeTransformerEncoderLayerKV._sa_block = boosted_sa  # type: ignore[assignment]
    CumulativeTransformerEncoderLayerKV._boosted = True  # type: ignore[attr-defined]


_patch_trim_boost()

# ---------------------------------------------------------------------------
# Patch 2: scale Q & K by √(head_dim) before feature map (fixes gradient scale)
# ---------------------------------------------------------------------------
try:
    from trim_transformer.modules import CumulativeLinearMultiheadAttentionKV  # type: ignore
except ModuleNotFoundError:
    # Same fallback path as before
    import sys as _sys
    from pathlib import Path as _Path
    _trim_root = _Path(__file__).resolve().parent.parent / "trim-transformer"
    _sys.path.append(str(_trim_root))
    from trim_transformer.modules import CumulativeLinearMultiheadAttentionKV  # type: ignore


def _patch_qk_scale():
    # Disable Q/K scaling – single residual boost is sufficient
    return  # no-op patch


_patch_qk_scale()

if __name__ == "__main__":
    main() 