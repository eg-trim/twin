import torch
import torch.nn as nn
from typing import Callable, Optional
from torch.utils.data import DataLoader
import torch.nn.functional as F
from architectures import model_is_causal

# -----------------------------
# Positional + Conditional Encodings
# -----------------------------

def make_positional_encoding(T: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    """Create a simple 3-channel spatio-temporal positional encoding.

    The encoding channels are normalised \(t, y, x\) coordinates in \[0,1\].
    """
    t_coord = (torch.linspace(-10, 10, T, device=device))
    row_coord = torch.linspace(0, 1, H, device=device)  # (H,)
    col_coord = torch.linspace(0, 1, W, device=device)  # (W,)

    time_enc = t_coord.view(T, 1, 1).expand(T, H, W)
    row_enc = row_coord.view(1, H, 1).expand(T, H, W)
    col_enc = col_coord.view(1, 1, W).expand(T, H, W)
    out = torch.stack([time_enc, row_enc, col_enc], dim=-1)  # (T, H, W, 3)
    return out.to(device)


def prepend_zero_sos(encoder_outputs: torch.Tensor) -> torch.Tensor:
    """Prepend a zero-filled start-of-sequence frame and drop the last frame."""
    sos = torch.zeros_like(encoder_outputs[:, :1])
    return torch.cat([sos, encoder_outputs[:, :-1]], dim=1)


# -----------------------------
# I/O helpers
# -----------------------------

def format_input_for_model(
    trajectories: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
    """Turn a 5-D trajectory tensor + conditioning parameters into a 2-D token sequence.

    Args:
        trajectories: (B, T, H, W, Q) waveform tensor.
        params:       (B, n_params) tensor of global conditioning parameters.

    Returns:
        (B, T·H·W, P+Q+n_params) tensor suitable for a Transformer with
        ``batch_first=True``.
    """
    B, T, H, W, _ = trajectories.shape
    _B, n_params = params.shape
    assert B == _B, "Batch size mismatch between trajectories and params"

    # Positional encoding (T, H, W, P) -> broadcast to batch
    pos_enc = make_positional_encoding(T, H, W, trajectories.device)  # (T, H, W, P)
    pos_enc = pos_enc.unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B, T, H, W, P)

    tokens = torch.cat([pos_enc, trajectories], dim=-1)  # (B, T, H, W, P+Q)
    flat_tokens = tokens.flatten(start_dim=1, end_dim=3)  # (B, T*H*W, P+Q)

    # Broadcast params to every token
    seq_len = flat_tokens.size(1)
    expanded_params = params.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T*H*W, n_params)

    return torch.cat([flat_tokens, expanded_params], dim=-1)  # (B, T*H*W, P+Q+n_params)


def reformat_output_from_model(
    model_outputs: torch.Tensor,  # (B, T*H'*W', P+Q')
    encoder_outputs: torch.Tensor,  # (B, T, H', W', Q')
) -> torch.Tensor:
    """Extract the value tokens (last Q' dims) and reshape for the decoder."""
    B, T, Hp, Wp, Qp = encoder_outputs.shape
    estimated = model_outputs[..., -Qp:]  # (B, T*H'*W', Q')
    return estimated.view(B, T, Hp * Wp * Qp)


def reformat_output_from_decoder(decoder_outputs: torch.Tensor, trajectories: torch.Tensor) -> torch.Tensor:
    """Reshape decoder output back into the original (B,T,H,W,Q) format."""
    return decoder_outputs.reshape_as(trajectories)


# -----------------------------
# Forward pipeline
# -----------------------------

class RefinementPipeline(nn.Module):
    """Forward pass for conditional waveform refinement."""

    def __init__(self,
                 model: nn.Module,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 process_trajectory: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.decoder = decoder
        self.process_trajectory = process_trajectory
        self.is_causal = model_is_causal(model)

    def forward(self, trajectories: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """End-to-end forward pass for conditional waveform refinement."""
        encoder_inputs = self.process_trajectory(trajectories)
        enc_out = self.encoder(encoder_inputs)
        if self.is_causal:
            enc_out = prepend_zero_sos(enc_out)

        model_in = format_input_for_model(enc_out, params)
        model_out = self.model(model_in)

        dec_in = reformat_output_from_model(model_out, enc_out)
        dec_out = self.decoder(dec_in)

        return reformat_output_from_decoder(dec_out, trajectories)


# -----------------------------
# Training / evaluation loops
# -----------------------------

class Trainer:
    """Generic trainer for conditional refinement models."""
    def __init__(self,
                 pipeline: RefinementPipeline,
                 loss_fn: Callable = F.mse_loss):
        self.pipeline = pipeline
        self.loss_fn = loss_fn
        self.device = next(pipeline.model.parameters()).device

    def _epoch(self, loader: DataLoader, optim: Optional[torch.optim.Optimizer] = None) -> float:
        is_train = optim is not None
        self.pipeline.train(is_train)
        
        running_loss = 0.0
        for waveforms, params in loader:
            waveforms, params = waveforms.to(self.device), params.to(self.device)
            if is_train:
                optim.zero_grad()
            
            out = self.pipeline(waveforms, params)
            loss = self.loss_fn(out, waveforms)
            
            if is_train:
                loss.backward()
                optim.step()
            
            running_loss += loss.item()
            
        return running_loss / len(loader)

    def train_epoch(self, loader: DataLoader, optim: torch.optim.Optimizer) -> float:
        return self._epoch(loader, optim)

    def eval_epoch(self, loader: DataLoader) -> float:
        with torch.no_grad():
            return self._epoch(loader, None)

    def eval_autoregressive(self, loader: DataLoader) -> float:
        """Autoregressive evaluation for *causal* conditional models."""
        assert self.pipeline.is_causal, "Autoregressive evaluation only for causal models"
        self.pipeline.train(False)
        running_loss = 0.0
        
        with torch.no_grad():
            for waveforms, params in loader:
                waveforms, params = waveforms.to(self.device), params.to(self.device)
                _, T, _, _, _ = waveforms.shape
                generated = torch.zeros_like(waveforms)
                generated[:, 0] = waveforms[:, 0]
                
                for t in range(1, T):
                    cur_in = generated.clone()
                    out = self.pipeline(cur_in, params)
                    generated[:, t] = out[:, t]
                
                running_loss += self.loss_fn(generated, waveforms).item()
                
        return running_loss / len(loader)
