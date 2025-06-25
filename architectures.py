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
        self.steps = nn.ModuleList([PicardStep(module, q, r) for module in modules])

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for step in self.steps:
            x = step(x, *args, **kwargs)
        return x

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
    Apply a list of modules in sequence.
    """
    def __init__(self, modules: List[nn.Module]):
        """
        Args:
            modules (List[nn.Module]): The list of modules to apply.
        """
        super().__init__()
        self.steps = nn.ModuleList(list(modules))

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for step in self.steps:
            x = step(x, *args, **kwargs)
        return x

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

class BlockCausalTransformer(nn.Module):
    """A Transformer encoder whose causal mask is block-causal with respect to
    temporal blocks. Each sequence is assumed to consist of ``n_timesteps``
    contiguous time blocks, where each time block contributes an equal number
    of spatial tokens. During self-attention a token may attend to all tokens
    in the current and previous time steps, but not to any token in a future
    timestep.

    Args:
        n_timesteps (int): Number of temporal blocks (time steps) in the
            sequence.
        d_model (int): Same as in ``nn.TransformerEncoderLayer``.
        nhead (int): Same as in ``nn.TransformerEncoderLayer``.
        dim_feedforward (int): Same as in ``nn.TransformerEncoderLayer``.
        dropout (float): Same as in ``nn.TransformerEncoderLayer``.
        n_layers (int): Number of encoder layers.
    """

    def __init__(self, n_timesteps: int, d_model: int, nhead: int, dim_feedforward: int, dropout: float, n_layers: int):
        super().__init__()
        self.n_timesteps = n_timesteps
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    @staticmethod
    def _generate_block_causal_mask(seq_len: int, n_timesteps: int, device: torch.device) -> torch.BoolTensor:
        """Return a boolean attention mask of shape *(seq_len, seq_len)* where
        *True* entries indicate **masked** (not visible) positions following
        the semantics of PyTorch's Transformer modules. The sequence is
        assumed to consist of ``n_timesteps`` contiguous temporal blocks of
        equal size.
        """
        if seq_len % n_timesteps != 0:
            raise ValueError(
                f"Sequence length {seq_len} is not divisible by the provided number of time steps {n_timesteps}."
            )

        block_size = seq_len // n_timesteps

        idx = torch.arange(seq_len, device=device)
        # Identify the timestep index of every token.
        block_idx = idx // block_size  # (seq_len,)
        # A token i may not attend to tokens whose timestep is strictly in the future.
        # Hence positions with block_j > block_i must be masked.
        mask = block_idx.unsqueeze(1) < block_idx.unsqueeze(0)  # (seq_len, seq_len)
        return mask

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Expect input of shape (B, L, D) with batch_first=True
        if x.dim() != 3:
            raise ValueError("Input must be a 3-D tensor of shape (B, L, D)")

        _, seq_len, _ = x.size()

        attn_mask = self._generate_block_causal_mask(seq_len, self.n_timesteps, x.device)
        return self.transformer(x, mask=attn_mask, *args, **kwargs)

class AcausalTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, n_layers: int):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.transformer(x, is_causal=False, *args, **kwargs)

def model_is_causal(model: nn.Module) -> bool:
    """Return True if model or any of its sub-modules is a BlockCausalTransformer.
    """
    if isinstance(model, BlockCausalTransformer):
        return True
    for m in model.modules():
        if isinstance(m, BlockCausalTransformer):
            return True
    return False

class SingleConvNeuralNet(nn.Module):
    def __init__(self, dim, hidden_dim=32, out_dim=32,hidden_ff=64,K=[4,4],S=[4,4]):
        super(SingleConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(dim, hidden_dim,
                                     kernel_size=K,
                                     stride=S)

        self.fc1 = nn.Linear(hidden_dim, hidden_ff)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ff, out_dim)

    def forward(self, x):
        B, T, H, W, Q = x.shape

        out = x.permute(0, 1, 4, 2, 3).reshape(B * T, Q, H, W)  # (B*T, Q, H, W)
        out = self.conv_layer1(out)  # (B*T, hidden_dim, H', W')
        out = out.permute(0, 2, 3, 1)  # (B*T, H', W', hidden_dim)

        out = self.fc1(out)  # (B*T, H', W', hidden_ff)
        out = self.relu2(out)  # (B*T, H', W', hidden_ff)
        out = self.fc2(out)  # (B*T, H', W', out_dim)

        _BT, H_prime, W_prime, C_out = out.shape
        out = out.view(B, T, H_prime, W_prime, C_out)  # (B, T, H', W', out_dim)
        return out