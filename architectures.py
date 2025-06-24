import torch
import torch.nn as nn
from typing import List

class PicardIterations(nn.Module):
    """
    Regarding a model as an operator, this module performs Picard iterations of that operator.
    Precisely, at each step, the indices before -q are kept constant, and the indices from -q to -1 are updated
    by the rule r * Id + (1-r) * T[f], where T is the model.
    """
    def __init__(self, modules: List[nn.Module], q: int, r: float = 0.5):
        """
        Args:
            modules (List[nn.Module]): The list of modules to apply Picard iterations to.
            q (int): The output dimension of the function.
            r (float): The relaxation parameter for the Picard step.
        """
        super().__init__()
        self.modules = nn.Sequential(nn.ModuleList([PicardStep(module, q, r) for module in modules]))

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.modules(x, *args, **kwargs)

class PicardStep(nn.Module):
    """
    Regarding a model as an operator, this module performs a Picard iteration of that operator.
    Precisely, the indices before -q are kept constant, and the indices from -q to -1 are updated
    by the rule r * Id + (1-r) * T[f], where T is the model.
    """
    def __init__(self, model: nn.Module, q: int, r: float = 0.5):
        """
        Args:
            model (nn.Module): The model to apply the Picard step to.
            q (int): The number of previous steps to use for the Picard step.
            r (float): The relaxation parameter for the Picard step.
        """
        super().__init__()
        self.model = model
        self.q = q
        self.r = r

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        z = self.model(x, *args, **kwargs)
        y = self.r * x[..., -self.q:] + (1-self.r) * z[..., -self.q:]
        y_ = torch.cat([x[..., :-self.q], y], dim=-1)
        return y_

class ArbitraryIterations(nn.Module):
    """
    Wrapper for nn.Sequential.
    """
    def __init__(self, modules: List[nn.Module]):
        """
        Args:
            modules (List[nn.Module]): The list of modules to apply.
        """
        super().__init__()
        self.modules = nn.Sequential(modules)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.modules(x, *args, **kwargs)

def broadcast_initial_conditions(trajectories: torch.Tensor) -> torch.Tensor:
    """
    Broadcast the initial condition across the time dimension to make the trajectory constant over time.
    It is assumed that the time dimension is the second dimension of the tensor.

    Args:
        trajectories (torch.Tensor): A tensor of shape (batch_size, time_steps, ...).

    Returns:
        torch.Tensor: A tensor with the same shape as the input, with the initial condition
                      broadcast across the time dimension.
    """
    initial_conditions = trajectories[:, 0, None, ...]
    return initial_conditions.expand_as(trajectories)

def identity(x: torch.Tensor) -> torch.Tensor:
    return x

def make_weight_shared_modules(make_module, n_modules: int) -> nn.Module:
    module = make_module()
    return nn.ModuleList([module for _ in range(n_modules)])

def make_weight_unshared_modules(make_module, n_modules: int) -> nn.Module:
    return nn.ModuleList([make_module() for _ in range(n_modules)])

class CausalTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, n_layers: int):
        super().__init__()
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout), num_layers=n_layers)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.transformer(x, is_causal=True, *args, **kwargs)

class AcausalTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, n_layers: int):
        super().__init__()
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout), num_layers=n_layers)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.transformer(x, is_causal=False, *args, **kwargs)