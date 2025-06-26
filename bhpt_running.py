import torch
import torch.nn as nn
from typing import Callable
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
    t_coord = torch.linspace(0, 1, T, device=device)  # (T,)
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

def pipeline(
    model: nn.Module,
    encoder: nn.Module,
    decoder: nn.Module,
    process_trajectory: Callable[[torch.Tensor], torch.Tensor],
    trajectories: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
    """End-to-end forward pass for conditional waveform refinement."""
    encoder_inputs = process_trajectory(trajectories)  # (B, T, H, W, Q)
    enc_out = encoder(encoder_inputs)  # (B, T, H', W', Q')
    if model_is_causal(model):
        enc_out = prepend_zero_sos(enc_out)

    model_in = format_input_for_model(enc_out, params)  # (B, T*H'*W', *)
    model_out = model(model_in)  # (B, T*H'*W', *)

    dec_in = reformat_output_from_model(model_out, enc_out)  # (B, T, H'*W'*Q')
    dec_out = decoder(dec_in)  # (B, T, H, W, Q)

    return reformat_output_from_decoder(dec_out, trajectories)


# -----------------------------
# Training / evaluation loops
# -----------------------------

def train_one_epoch(
    model: nn.Module,
    encoder: nn.Module,
    decoder: nn.Module,
    process_trajectory: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    optim: torch.optim.Optimizer,
) -> float:
    model.train(); encoder.train(); decoder.train()
    running = 0.0
    device = next(model.parameters()).device

    for waveforms, params in loader:
        waveforms, params = waveforms.to(device), params.to(device)
        optim.zero_grad()
        out = pipeline(model, encoder, decoder, process_trajectory, waveforms, params)
        loss = F.mse_loss(out, waveforms)
        loss.backward(); optim.step()
        running += loss.item()
    return running / len(loader)


def evaluate_refinement(
    model: nn.Module,
    encoder: nn.Module,
    decoder: nn.Module,
    process_trajectory: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
) -> float:
    model.eval(); encoder.eval(); decoder.eval()
    running = 0.0
    device = next(model.parameters()).device

    with torch.no_grad():
        for waveforms, params in loader:
            waveforms, params = waveforms.to(device), params.to(device)
            out = pipeline(model, encoder, decoder, process_trajectory, waveforms, params)
            running += F.mse_loss(out, waveforms).item()
    return running / len(loader)


def evaluate_autoregressive(
    model: nn.Module,
    encoder: nn.Module,
    decoder: nn.Module,
    process_trajectory: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
) -> float:
    """Autoregressive evaluation for *causal* conditional models."""
    model.eval(); encoder.eval(); decoder.eval()
    running = 0.0
    device = next(model.parameters()).device

    with torch.no_grad():
        for waveforms, params in loader:
            waveforms, params = waveforms.to(device), params.to(device)
            _, T, _, _, _ = waveforms.shape
            generated = torch.zeros_like(waveforms)
            generated[:, 0] = waveforms[:, 0]
            for t in range(1, T):
                cur_in = generated.clone()
                out = pipeline(model, encoder, decoder, process_trajectory, cur_in, params)
                generated[:, t] = out[:, t]
            running += F.mse_loss(generated, waveforms).item()
    return running / len(loader)
