# %%
import time, math, gc, random
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.io import loadmat
from trim_transformer.transformer_layers import CumulativeTransformerEncoderLayerKV, CumulativeTransformerEncoderKV
from torchvision.ops import MLP


# Reproducibility ----------------------------------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Device -------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
DATA_PATH = "./ns_data_visc_8e-4.mat"
data_dict = loadmat(DATA_PATH)
u = torch.tensor(data_dict["u"][:625]).to(device)  # (N, Nx, Ny, T)
a = torch.tensor(data_dict["a"][:625]).to(device)  # (N, Nx, Ny)

# %%
print(u[0])

class TokensDataset(Dataset):
    """Flat token sequence dataset for next-step prediction."""
    def __init__(self, u, a, n_timesteps=None):
        N, Nx, Ny, T = u.shape
        u = u.permute(0, 3, 1, 2)
        a = a.unsqueeze(1)
        if n_timesteps is not None and n_timesteps < T:
            idx = np.linspace(0, u.shape[1] - 1, num=n_timesteps, dtype=int)
            u = u[:, idx]
        self.data = torch.cat([a, u], dim=1).reshape(N, n_timesteps+1, Nx, Ny, 1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

N_TIMESTEPS = 20
full_ds = TokensDataset(u, a, n_timesteps=N_TIMESTEPS)
train_size = int(0.8 * len(full_ds))
val_size = len(full_ds) - train_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size])

BATCH_SIZE = 8
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
print(f"Train/Val samples: {len(train_ds)} / {len(val_ds)}")

N, Nx, Ny, T = u.shape
Q = 1
X_COMPRESSION = 4
Y_COMPRESSION = 4
Nx_ = Nx // X_COMPRESSION
Ny_ = Ny // Y_COMPRESSION
n_tokens = N_TIMESTEPS * Nx_ * Ny_  
block_size = Nx_ * Ny_

# %%
def make_block_mask_after(n_tokens, block_size):
    idx = torch.arange(n_tokens, dtype=torch.long)
    mask_after = block_size * ((idx // block_size) + 1)-1
    return mask_after

def mask_after_to_dense_mask(mask_after):
    n_tokens = mask_after.shape[0]
    col_indices = torch.arange(n_tokens)
    return (col_indices > mask_after.unsqueeze(1)).float()

mask_after = make_block_mask_after(n_tokens, block_size)
dense_mask = mask_after_to_dense_mask(mask_after)
mask_after = mask_after.to(device)
dense_mask = dense_mask.to(device)

# %%
a = make_block_mask_after(10, 2)
b = mask_after_to_dense_mask(a)

print(a)
print(b)

# %%
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
        out = out.contiguous().view(B, T, H_prime, W_prime, C_out)  # (B, T, H', W', out_dim)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)
    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

# %%
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.1

encoder = SingleConvNeuralNet(
    1,
    EMBED_DIM,
    EMBED_DIM,
    K=[X_COMPRESSION, Y_COMPRESSION],
    S=[X_COMPRESSION, Y_COMPRESSION]
)

decoder = MLP(
    in_channels=Nx_*Ny_*(EMBED_DIM),
    hidden_channels=[64, 256, Nx*Ny*Q],
    activation_layer=nn.ELU,
)

pos_enc = PositionalEncoding(EMBED_DIM, max_len=T*Nx*Ny)

baseline_layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=NUM_HEADS, batch_first=True)
baseline_model = nn.TransformerEncoder(baseline_layer, num_layers=NUM_LAYERS)

cumulative_layer = CumulativeTransformerEncoderLayerKV(d_model=EMBED_DIM, nhead=NUM_HEADS, batch_first=True)
cumulative_model = CumulativeTransformerEncoderKV(cumulative_layer, num_layers=NUM_LAYERS)


class Pipeline(nn.Module):
    def __init__(self, encoder, decoder, pos_enc, transformer, mask):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pos_enc = pos_enc
        self.transformer = transformer
        self.mask = mask

    def forward(self, x):
        B, T, _, _, _ = x.shape
        y = self.encoder(x)
        y = y.flatten(1, 3)
        y = self.pos_enc(y)
        y = self.transformer(y, mask=self.mask)
        y = y.reshape(B, T, -1)
        y = self.decoder(y)
        return y.reshape_as(x)

baseline_pipeline = Pipeline(encoder, decoder, pos_enc, baseline_model, dense_mask).to(device)
cumulative_pipeline = Pipeline(encoder, decoder, pos_enc, cumulative_model, mask_after).to(device)

# %%
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    start = time.time()
    running = 0.0
    for traj in loader:
        optimizer.zero_grad()
        pred = model(traj[:, :-1])
        loss = criterion(pred, traj[:, 1:])
        loss.backward()
        optimizer.step()
        running += loss.item() * traj.size(0)
    elapsed = time.time() - start
    return running / len(loader.dataset), elapsed

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running = 0.0
    for traj in loader:
        pred = model(traj[:, :-1])
        loss = criterion(pred, traj[:, 1:])
        running += loss.item() * traj.size(0)
    return running / len(loader.dataset)

def peak_mem():
    if device.type == "cuda":
        torch.cuda.synchronize()
        m = torch.cuda.max_memory_allocated() / 1024**2
        torch.cuda.reset_peak_memory_stats()
        return m
    return 0.0

EPOCHS = 1000
criterion = nn.MSELoss()

results = {}
for name, model in [("baseline", baseline_pipeline), ("cumulative", cumulative_pipeline)]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist = {"train": [], "val": [], "time": [], "mem": []}
    for ep in range(1, EPOCHS+1):
        train_loss, t = train_epoch(model, train_loader, optimizer, criterion)
        if device.type == "cuda":
            optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache(); gc.collect()
        val_loss = evaluate(model, val_loader, criterion)
        if device.type == "cuda":
            optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache(); gc.collect()
        mem = peak_mem()
        hist["train"].append(train_loss)
        hist["val"].append(val_loss)
        hist["time"].append(t)
        hist["mem"].append(mem)
        print(f"{name:10s} | epoch {ep}/{EPOCHS} | train {train_loss:.3e} | val {val_loss:.3e} | {t:.2f}s | mem {mem:.1f}MB")
    results[name] = hist
    if device.type == "cuda":
        optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache(); gc.collect()

# %%
summary = pd.DataFrame.from_dict({
    k: {
        "train_loss": v["train"][-1],
        "val_loss": v["val"][-1],
        "time/epoch (s)": np.mean(v["time"]),
        "peak_mem (MB)": max(v["mem"]),
    } for k, v in results.items()
}, orient="index")
print(summary) 

# %%