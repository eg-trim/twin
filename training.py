import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

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
    """Forward pass for acausal models (predict full trajectory)."""

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
        dec_out = self.decoder(dec_in)  # (B,T,H*W*Q)

        return dec_out[:, 1:].reshape_as(traj)


class CausalOSPipeline(nn.Module):
    def __init__(self, model: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, init_cond: torch.Tensor, traj: torch.Tensor) -> torch.Tensor:
        traj_full = torch.cat([init_cond.unsqueeze(1), traj], dim=1)
        enc_full = self.encoder(traj_full)

        # Use frames 0 … T-1 as context
        enc_ctx  = enc_full[:, :-1]  # (B,T,Hp,Wp,C)
        B, T, Hp, Wp, C = enc_ctx.shape

        # Tokens
        pos = make_positional_encoding(T, Hp, Wp, device=enc_ctx.device)
        pos = pos.unsqueeze(0).expand(B, -1, -1, -1, -1)
        tokens = torch.cat([pos, enc_ctx], dim=-1).reshape(B, T*Hp*Wp, C+3)

        model_out = self.model(tokens)
        preds_enc = model_out[..., -C:]
        preds_spatial = preds_enc.view(B, T, Hp, Wp, C)
        dec_in = preds_spatial.reshape(B, T, Hp*Wp*C)
        dec_out = self.decoder(dec_in)  # (B,T,H*W*Q)

        preds = dec_out.reshape_as(traj)
        return preds


class _CausalMSStepPipeline(nn.Module):
    def __init__(self, model: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, last_frame: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # current_sequence shape: (B, t, H, W, Q)
        enc_out = self.encoder(last_frame)  # (B,t,Hp,Wp,C)

        B, t, Hp, Wp, C = enc_out.shape
        pos = make_positional_encoding(t, Hp, Wp, device=enc_out.device)
        pos = pos.unsqueeze(0).expand(B, -1, -1, -1, -1)
        tokens = torch.cat([pos, enc_out], dim=-1).reshape(B, t*Hp*Wp, C+3)

        model_out = self.model(tokens, use_kv_cache=True, update_kv_cache=True)
        preds_enc = model_out[..., -C:]
        preds_spatial = preds_enc.view(B, t, Hp, Wp, C)
        dec_in = preds_spatial.reshape(B, t, Hp*Wp*C)
        dec_out = self.decoder(dec_in)  # (B,t,H*W*Q)

        next_frame = dec_out.reshape_as(last_frame)
        return next_frame

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.train(mode)
        self.encoder.train(mode)
        self.decoder.train(mode)
        return self

class CausalMSPipeline(nn.Module):
    def __init__(self, model: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.step = _CausalMSStepPipeline(model, encoder, decoder)

    def forward(self, init_cond: torch.Tensor, traj: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        frames = [init_cond.unsqueeze(1)]  # list of (B,1,H,W,Q)

        for _ in range(traj.shape[1]):
            last_frame = frames[-1]
            next_frame = self.step(last_frame)
            frames.append(next_frame)

        pred_traj = torch.cat(frames[1:], dim=1)  # (B,T,H,W,Q)
        return pred_traj

class Trainer:
    """Generic trainer parameterised by `predict_fn`.

    DataLoader must yield `(initial_conditions, trajectory)` where:
        initial_conditions : (B, H, W, Q)
        trajectory         : (B, T, H, W, Q)  (frames 1 … T)

    The trainer compares `predict_fn(init, traj)` against the same `traj`.
    """

    def __init__(self,
                 predict_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 loss_fn: Callable = F.mse_loss,
                 grad_clip_norm: float = 1.0):
        self.predict_fn = predict_fn
        self.loss_fn = loss_fn
        self.grad_clip_norm = grad_clip_norm

    def _epoch(self, loader: DataLoader, optim: torch.optim.Optimizer | None) -> float:
        train_mode = optim is not None

        running_loss = 0.0
        for init_cond, traj in loader:
            if train_mode:
                optim.zero_grad()

            preds = self.predict_fn(init_cond, traj)
            loss = self.loss_fn(preds, traj)

            if train_mode:
                loss.backward()
                # Apply gradient clipping if specified
                if self.grad_clip_norm > 0:
                    clip_grad_norm_(
                        [p for p in optim.param_groups[0]['params'] if p.grad is not None],
                        self.grad_clip_norm
                    )
                optim.step() 

            running_loss += loss.item()

        return running_loss / len(loader)

    def train_epoch(self, loader: DataLoader, optim: torch.optim.Optimizer) -> float:
        return self._epoch(loader, optim)

    def eval_epoch(self, loader: DataLoader) -> float:
        return self._epoch(loader, None)


def build_pipeline(kind: str,
                  encoder: nn.Module,
                  model: nn.Module,
                  decoder: nn.Module,
                  ) -> nn.Module:
    if kind == 'acausal':
        pipeline = AcausalPipeline(model, encoder, decoder)
    elif kind == 'causal_one_step':
        pipeline = CausalOSPipeline(model, encoder, decoder)
    elif kind == 'causal_many_steps':
        pipeline = CausalMSPipeline(model, encoder, decoder)
    else:
        raise ValueError(f"Unknown trainer kind '{kind}'.")
    return pipeline