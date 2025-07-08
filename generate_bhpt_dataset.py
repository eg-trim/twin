"""Generate a (waveforms, params) dataset from the BHPTNRSur1dq1e4 surrogate.

The script samples random mass-ratio values ``q`` and produces tensors

    waveforms: (N, T, 1, 1, 2)  # plus & cross
    params:    (N, 1)           # scalar mass ratio

The file can be used as a library function *or* executed directly to
produce a ``.pt`` file on disk.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

PATH_TO_BHPTNRSur = "/home/ubuntu/EG-UT/BHPTNRSurrogate"
import sys
sys.path.append(PATH_TO_BHPTNRSur)
from surrogates import BHPTNRSur1dq1e4 as bhptsur

# The surrogate lives in the repository cloned in the notebook path
from surrogates import BHPTNRSur1dq1e4 as bhptsur

def _synthesize_polarisation(
    h_dict: dict[Tuple[int, int], np.ndarray],
    modes: Tuple[Tuple[int, int], ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """Sum the requested modes and split into (h_+, h_×)."""
    h = sum(h_dict[m] for m in modes)  # complex
    h_plus = np.real(h)
    h_cross = np.imag(h)
    return h_plus, h_cross


def generate_bhpt_dataset(
    n_samples: int,
    *,
    q_min: float = 2.5,
    q_max: float = 10.0,
    M_tot: float = 60.0,
    dist_mpc: float = 100.0,
    n_timesteps: Optional[int] = None,
    modes: Tuple[Tuple[int, int], ...] = (
        (2, 2),
    ),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a synthetic dataset of gravitational waveforms.

    Returns
    -------
    waveforms : torch.Tensor
        Shape ``(N, T, 1, 1, 2)`` where the last channel is (h_+, h_×).
    params : torch.Tensor
        Shape ``(N, 1)`` containing the mass-ratio for each sample.
    """
    rng = np.random.default_rng()
    # Sample in the open interval (q_min, q_max) to stay strictly within bounds
    eps = np.finfo(float).eps
    q_low = q_min + eps
    q_high = q_max - eps
    # Sample q log-uniformly as we have a wide range
    log_q_low = np.log(q_low)
    log_q_high = np.log(q_high)
    log_q_values = rng.uniform(log_q_low, log_q_high, size=n_samples)
    q_values = np.exp(log_q_values)  # np.ndarray[float64]

    print("First 5 q values:", q_values[:5])

    waveforms: List[np.ndarray] = []

    # First call to get canonical timeline length
    t_ref, _ = bhptsur.generate_surrogate(q=q_values[0], M_tot=None, dist_mpc=None)
    full_T = len(t_ref)
    # Pre-compute subsample indices if the user requests it
    if n_timesteps is not None and n_timesteps < full_T:
        idx = np.linspace(0, full_T - 1, num=n_timesteps, dtype=int)
    else:
        idx = slice(4096)
        n_timesteps = full_T

    for q in tqdm(q_values, desc="Generating waveforms"):
        t, h_dict = bhptsur.generate_surrogate(q=q, M_tot=None, dist_mpc=None)
        assert len(t) == full_T, "Inconsistent timeline length returned by surrogate"
        h_plus, h_cross = _synthesize_polarisation(h_dict, modes)
        sample = np.stack([h_plus[idx], h_cross[idx]], axis=-1) * 20
        sample = sample[..., None, None, :]  # (T, 1, 1, 2)
        waveforms.append(sample.astype(np.float32))

    waveforms_np = np.stack(waveforms, axis=0)  # (N, T, 1, 1, 2)
    params_np = q_values.astype(np.float32)[:, None]  # (N,1)

    return torch.from_numpy(waveforms_np), torch.from_numpy(params_np)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate BHPT waveform dataset")
    p.add_argument("--n-samples", type=int, default=128, help="Number of waveforms to generate")
    p.add_argument("--output", type=Path, default=Path("bhpt_dataset.pt"), help="Path to write the dataset (.pt)")
    p.add_argument("--n-timesteps", type=int, default=None, help="Optional temporal subsampling")
    p.add_argument("--q-min", type=float, default=1.0)
    p.add_argument("--q-max", type=float, default=10.0)
    return p


def main():
    args = _build_argparser().parse_args()
    waves, params = generate_bhpt_dataset(
        args.n_samples,
        q_min=args.q_min,
        q_max=args.q_max,
        n_timesteps=args.n_timesteps,
    )
    torch.save({"waveforms": waves, "params": params}, args.output)
    print(f"Saved dataset to {args.output} with shapes: waveforms {waves.shape}, params {params.shape}")


if __name__ == "__main__":
    main() 