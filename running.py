import torch
import torch.nn as nn
from typing import Callable
from torch.utils.data import DataLoader
import torch.nn.functional as F

def make_positional_encoding(T, H, W, device: torch.device) -> torch.Tensor:
    t_coord = torch.linspace(0, 1, T, device=device)  # (T,)
    row_coord = torch.linspace(0, 1, H, device=device)  # (H,)
    col_coord = torch.linspace(0, 1, W, device=device)  # (W,)

    time_enc = t_coord.view(T, 1, 1).expand(T, H, W)  # (H, W, T)
    row_enc  = row_coord.view(1, H, 1).expand(T, H, W)  # (H, W, T)
    col_enc  = col_coord.view(1, 1, W).expand(T, H, W)  # (H, W, T)

    out = torch.stack([time_enc, row_enc, col_enc], dim=-1)  # (H, W, T, P)
    return out.to(device)

def format_input_for_model(trajectories: torch.Tensor) -> torch.Tensor:
    """Concatenate spatial–temporal tokens with their positional encodings and
    flatten the (T, H, W) grid into a 1-D token sequence.

    Args:
        trajectories: Tensor of shape (B, T, H, W, Q).

    Returns:
        Tensor of shape (B, T·H·W, Q+P) suitable for ``CausalTransformer`` with
        ``batch_first=True``.
    """
    B, T, H, W, _ = trajectories.shape
    positional_encodings = make_positional_encoding(T, H, W, device=trajectories.device)
    positional_encodings = positional_encodings.unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B, T, H, W, P)
    return (
        torch.cat([positional_encodings, trajectories], dim=-1)
        .flatten(start_dim=1, end_dim=3)  # (B, T·H·W, P+Q)
    )

def reformat_output_from_model(model_outputs: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
    # (B, T*H'*W', P+Q'), (B, T, H', W', Q') -> (B, T, H'*W'*Q')
    B, T, H_prime, W_prime, Q_prime = encoder_outputs.shape
    estimated_trajectories = model_outputs[..., -Q_prime:]
    return estimated_trajectories.reshape(B,T,H_prime*W_prime*Q_prime)

def reformat_output_from_decoder(decoder_outputs: torch.Tensor, trajectories: torch.Tensor) -> torch.Tensor:
    return decoder_outputs.reshape_as(trajectories)

def pipeline(model: nn.Module,
             encoder: nn.Module,
             decoder: nn.Module,
             process_trajectory: Callable,
             trajectories: torch.Tensor) -> torch.Tensor:
    encoder_inputs = process_trajectory(trajectories)  # (B, T, H, W, Q)
    encoder_outputs = encoder(encoder_inputs)  # (B, T, H', W', Q')
    model_inputs = format_input_for_model(encoder_outputs)  # (B, T*H'*W', P+Q')
    model_outputs = model(model_inputs)  # (B, T*H'*W', P+Q')
    decoder_inputs = reformat_output_from_model(model_outputs, encoder_outputs)  # (B, T, H'*W'*Q')
    decoder_outputs = decoder(decoder_inputs)  # (B, T, H, W, Q)
    outputs = reformat_output_from_decoder(decoder_outputs, trajectories)
    return outputs

def train_one_epoch(model: nn.Module,
                    encoder: nn.Module,
                    decoder: nn.Module,
                    process_trajectory: Callable,
                    loader: DataLoader,
                    optim: torch.optim.Optimizer,
                    ) -> float:
    model.train()
    running_loss = 0.0

    for trajectories in loader:
        optim.zero_grad()
        outputs = pipeline(model,
                           encoder,
                           decoder,
                           process_trajectory,
                           trajectories)
        loss = F.mse_loss(outputs, trajectories)
        loss.backward()
        optim.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model: nn.Module,
             encoder: nn.Module,
             decoder: nn.Module,
             process_trajectory: Callable,
             loader: DataLoader,
             ) -> float:
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for trajectories in loader:
            outputs = pipeline(model,
                               encoder,
                               decoder,
                               process_trajectory,
                               trajectories)
            loss = F.mse_loss(outputs, trajectories)
            running_loss += loss.item()
    return running_loss / len(loader)