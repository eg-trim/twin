# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import architectures as arch
from functools import partial
import argparse
from pathlib import Path
from torchvision.ops import MLP
from data import load_navier_stokes_tensor, setup_dataloaders
from training import build_pipeline
from training import Trainer
from architectures import SingleConvNeuralNet
from trim_transformer.transformer_layers import TrimTransformerEncoderLayer, TrimTransformerEncoder
import time


parser = argparse.ArgumentParser(description="Navier–Stokes training script.")
parser.add_argument("--name", type=str, default="test")
parser.add_argument("--data", type=Path, default=Path("ns_data.mat"), help="Path to the .mat dataset.")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (L2 penalty) for Adam optimizer.")
parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping norm (set to 0 to disable).")
parser.add_argument("--min-lr", type=float, default=1e-9, help="Minimum learning rate for cosine annealing scheduler.")
parser.add_argument("--T-max", type=int, default=101, help="Maximum number of iterations for cosine annealing scheduler.")
parser.add_argument("--n-timesteps", type=int, default=11, help="Number of temporal frames to sample from the raw data (consistent with notebook).")

parser.add_argument("--share", action="store_true", help="Share weights between modules.")
parser.add_argument("--no-share", dest="share", action="store_false", help="Don't share weights between modules.")
parser.set_defaults(share=True)

parser.add_argument("--refinement", action="store_true", help="Use refinement.")
parser.add_argument("--no-refinement", dest="refinement", action="store_false", help="Don't use refinement.")
parser.set_defaults(refinement=True)

parser.add_argument("--picard", action="store_true", help="Use Picard iterations.")
parser.add_argument("--no-picard", dest="picard", action="store_false", help="Don't use Picard iterations.")
parser.set_defaults(picard=True)

parser.add_argument("--d_model", type=int, default=64)
parser.add_argument("--nhead", type=int, default=4)
parser.add_argument("--dim_feedforward", type=int, default=64)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--n_layers", type=int, default=4)
parser.add_argument("--n_modules", type=int, default=4)
parser.add_argument("--r", type=float, default=0.5)

# Encoder arguments
parser.add_argument("--encoder-hidden-dim", type=int, default=None, 
                    help="Hidden dimension for encoder (default: d_model - P)")
parser.add_argument("--encoder-hidden-ff", type=int, default=128,
                    help="Hidden feedforward dimension for encoder")
parser.add_argument("--patch_shape", type=int, nargs=2, default=[4, 4],
                    help="A token is a patch of size patch_shape")

# Decoder arguments
parser.add_argument("--decoder-hidden-channels", type=int, nargs="+", default=[64, 256],
                    help="Hidden channels for decoder MLP (excluding final output channel)")

parser.add_argument("--train-kind", choices=["acausal", "causal_one_step", "causal_many_steps"], default="acausal",
                    help="Pipeline kind to use during training")
parser.add_argument("--val-kind", choices=["acausal", "causal_one_step", "causal_many_steps"], default="acausal",
                    help="Pipeline kind to use during validation")

args = parser.parse_args()

# %%
# Create directory structure
base_dir = Path("runs/" + args.name)
base_dir.mkdir(exist_ok=True)

# Find the next available run number
run_num = 0
while True:
    run_dir = base_dir / f"run{run_num}"
    if not run_dir.exists():
        break
    run_num += 1

# Create the run directory
run_dir.mkdir(exist_ok=True)
print(f"Created run directory: {run_dir}")

# Save hyperparameters/config
config_dict = {
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'lr': args.lr,
    'weight_decay': args.weight_decay,
    'grad_clip_norm': args.grad_clip_norm,
    'min_lr': args.min_lr,
    'T_max': args.T_max,
    'n_timesteps': args.n_timesteps,
    'share': args.share,
    'refinement': args.refinement,
    'picard': args.picard,
    'd_model': args.d_model,
    'nhead': args.nhead,
    'dim_feedforward': args.dim_feedforward,
    'dropout': args.dropout,
    'n_layers': args.n_layers,
    'n_modules': args.n_modules,
    'r': args.r,
    'encoder_hidden_dim': args.encoder_hidden_dim,
    'encoder_hidden_ff': args.encoder_hidden_ff,
    'patch_shape': args.patch_shape,
    'decoder_hidden_channels': args.decoder_hidden_channels,
    'train_kind': args.train_kind,
    'val_kind': args.val_kind,
}
np.save(run_dir / "config.npy", config_dict)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

init_conds, trajs = load_navier_stokes_tensor(args.data, n_timesteps=args.n_timesteps)
init_conds = init_conds.to(device)
trajs = trajs.to(device)

train_loader, val_loader = setup_dataloaders(init_conds, trajs, batch_size=args.batch_size)
P = 3
N, T, H, W, Q = trajs.shape

# %%
def galerkin_init(param, gain=0.01, diagonal_weight=0.01):
    nn.init.xavier_uniform_(param, gain=gain)
    param.data += diagonal_weight * torch.diag(torch.ones(param.size(-1), dtype=torch.float))

class TrimTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, n_layers: int, mask: torch.Tensor | None = None, scale: float | None = None):
        super().__init__()

        norm_k = nn.LayerNorm(d_model//nhead)
        norm_v = nn.LayerNorm(d_model//nhead)
        encoder_layer = TrimTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_k=norm_k,
            norm_v=norm_v,
            q_weight_init=galerkin_init,
            k_weight_init=galerkin_init,
            v_weight_init=galerkin_init,
            scale=scale,
        )
        self.mask = mask
        self.transformer = TrimTransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, use_kv_cache: bool = False, update_kv_cache: bool = False) -> torch.Tensor:
        if self.mask is None:
            mask = None
        else:
            mask = self.mask[:x.shape[1]]
        return self.transformer(x, mask=mask, use_kv_cache=use_kv_cache, update_kv_cache=update_kv_cache)

    def clear_kv_cache(self):
        self.transformer.clear_kv_cache()

def make_block_mask_after(n_tokens, block_size):
    idx = torch.arange(n_tokens, dtype=torch.long)
    mask_after = block_size * ((idx // block_size) + 1)-1
    return mask_after

# %%
# Set encoder hidden_dim: use provided value or default to d_model - P
encoder_hidden_dim = args.encoder_hidden_dim if args.encoder_hidden_dim is not None else args.d_model - P

encoder = SingleConvNeuralNet(dim=Q,
                                hidden_dim=encoder_hidden_dim,
                                out_dim=args.d_model-P,
                                hidden_ff=args.encoder_hidden_ff,
                                K=args.patch_shape,
                                S=args.patch_shape)
encoder = encoder.to(device)

# Dummy forward pass to get shapes
with torch.no_grad():
    _, _, H_prime, W_prime, _ = encoder.forward(trajs[0, None, ...].to(device)).shape
block_size = H_prime * W_prime


if args.refinement:
    n_tokens = (args.n_timesteps + 1) * H_prime * W_prime
    scale = 1 / n_tokens
    mask = None
    make_module = partial(TrimTransformer,
                    d_model=args.d_model,
                    nhead=args.nhead,
                    dim_feedforward=args.dim_feedforward,
                    dropout=args.dropout,
                    n_layers=args.n_layers,
                    mask=mask,
                    scale=scale)
else:
    n_tokens = args.n_timesteps * H_prime * W_prime
    scale = 1 / n_tokens
    mask = make_block_mask_after(n_tokens, block_size).to(device)
    make_module = partial(TrimTransformer,
                          d_model=args.d_model,
                          nhead=args.nhead,
                          dim_feedforward=args.dim_feedforward,
                          dropout=args.dropout,
                          n_layers=args.n_layers,
                          mask=mask,
                          scale=scale)
if args.share:
    modules = arch.make_weight_shared_modules(make_module, n_modules=args.n_modules)
else:
    modules = arch.make_weight_unshared_modules(make_module, n_modules=args.n_modules)
if args.picard:
    model = arch.PicardIterations(modules, q=Q, r=args.r)
else:
    model = arch.ArbitraryIterations(modules)
model = model.to(device)

# Build decoder hidden channels: user-specified channels + fixed output channel
decoder_hidden_channels = args.decoder_hidden_channels + [H*W*Q]

decoder = MLP(
    in_channels=H_prime*W_prime*(args.d_model-P),
    hidden_channels=decoder_hidden_channels,
    activation_layer=nn.ELU,
)
decoder = decoder.to(device)

# %%
loss_fn = F.mse_loss
optim = torch.optim.Adam(
    list(model.parameters()) + list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.T_max, eta_min=args.min_lr)

train_pipeline = build_pipeline(args.train_kind,
                                encoder=encoder,
                                model=model,
                                decoder=decoder)

val_pipeline = build_pipeline(args.val_kind,
                              encoder=encoder,
                              model=model,
                              decoder=decoder)

train_trainer = Trainer(train_pipeline, loss_fn, grad_clip_norm=args.grad_clip_norm)

val_trainer = Trainer(val_pipeline, loss_fn)

# Initialize lists to track losses and times
train_losses = []
val_losses = []
epoch_times = []

for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    
    train_loss = train_trainer.train_epoch(train_loader, optim)
    with torch.no_grad():
        val_loss   = val_trainer.eval_epoch(val_loader)
    scheduler.step()

    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time

    # Append losses and times to tracking lists
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    epoch_times.append(epoch_time)

    print(f"{args.name} | Epoch {epoch:3d} | train loss: {train_loss:.6f} | val loss: {val_loss:.6f} | time: {epoch_time:.2f}s")
    
    # Save losses and times as numpy arrays every epoch in run directory
    np.save(run_dir / "train_loss.npy", np.array(train_losses))
    np.save(run_dir / "val_loss.npy", np.array(val_losses))
    np.save(run_dir / "epoch_times.npy", np.array(epoch_times))

# Save model weights in run directory
torch.save({"state_dict": model.state_dict()}, run_dir / "model_weights.pt")

# Save final loss arrays and times in run directory
np.save(run_dir / "train_loss.npy", np.array(train_losses))
np.save(run_dir / "val_loss.npy", np.array(val_losses))
np.save(run_dir / "epoch_times.npy", np.array(epoch_times))

print(f"\nTraining completed! All files saved to: {run_dir}")
print(f"Final train loss: {train_losses[-1]:.6f}")
print(f"Final val loss: {val_losses[-1]:.6f}")
print(f"Average epoch time: {np.mean(epoch_times):.2f}s")
print(f"Total training time: {np.sum(epoch_times):.2f}s")